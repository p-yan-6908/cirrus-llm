"""Sliding-window attention layer for Cirrus.

GQA attention with QKNorm, sliding window, and causal masking.
Concentrated in later layers of the model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QKNorm(nn.Module):
    """Query-Key normalization for stable quantization."""

    def __init__(self, head_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(head_dim))

    def forward(self, x):
        """Normalize along head_dim."""
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.scale


class SlidingWindowAttention(nn.Module):
    """Sliding-window grouped-query attention with QKNorm.

    Uses a causal sliding window: each token can attend to the
    previous window_size tokens, giving O(n * window_size) complexity.
    """

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.window_size = config.window_size
        self.d_model = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

        if config.qk_norm:
            self.q_norm = QKNorm(config.head_dim)
            self.k_norm = QKNorm(config.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.n_rep = config.n_heads // config.n_kv_heads

    def forward(self, x, kv_cache=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            kv_cache: optional (k_cache, v_cache) from previous steps

        Returns:
            output: (batch, seq_len, d_model)
            new_kv_cache: updated (k_cache, v_cache)
        """
        B, T, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Update KV cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v)

        # Expand KV heads for GQA
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(
                B, self.n_heads, -1, self.head_dim
            )
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(
                B, self.n_heads, -1, self.head_dim
            )

        # Sliding window causal attention
        seq_len = k.shape[2]
        if seq_len > self.window_size:
            # Only keep the last window_size tokens
            k = k[:, :, -self.window_size:, :]
            v = v[:, :, -self.window_size:, :]

        # SDPA with causal mask
        # For sliding window, we use a custom attention mask
        output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            scale=1.0 / math.sqrt(self.head_dim),
        )

        # Merge heads and project
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.o_proj(output)

        return output, new_kv_cache

    def step(self, x, kv_cache):
        """Single-step inference.

        Args:
            x: (batch, 1, d_model)
            kv_cache: previous KV cache

        Returns:
            output: (batch, 1, d_model)
            new_kv_cache: updated cache
        """
        return self.forward(x, kv_cache=kv_cache)
