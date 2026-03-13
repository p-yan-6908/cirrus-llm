"""Mixture of Depths (MoD) for Cirrus.

Easy tokens get zero contribution from the layer (skip connection).
Combined with MoE for double sparsity (~6-8x compute reduction).

The gate produces a per-token scale factor:
  output = mask * layer_output + (1 - mask) * input

This is correct for both SSM and attention layers, since the full
sequence is always processed (required for SSM recurrence), but
skipped tokens pass through unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoDGate(nn.Module):
    """Learned gate that decides which tokens to process vs skip.

    For each token, produces a score. Only the top-k tokens
    (by score) get the layer's output; the rest pass through unchanged.
    """

    def __init__(self, d_model, capacity=0.5):
        super().__init__()
        self.gate = nn.Linear(d_model, 1, bias=False)
        self.capacity = capacity

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            mask: (batch, seq_len, 1) float — 1.0 for process, 0.0 for skip
            scores: (batch, seq_len) — raw gate scores (for loss)
        """
        B, T, _ = x.shape
        scores = self.gate(x).squeeze(-1)  # (B, T)

        # Number of tokens to process
        k = max(1, int(T * self.capacity))

        # Get top-k indices
        _, topk_indices = scores.topk(k, dim=-1)

        # Create float mask (for gradient flow)
        mask = torch.zeros(B, T, device=x.device, dtype=x.dtype)
        mask.scatter_(1, topk_indices, 1.0)
        mask = mask.unsqueeze(-1)  # (B, T, 1)

        return mask, scores


class MoDWrapper(nn.Module):
    """Wraps a layer (SSM or Attention) with MoD gating.

    All tokens are processed through the layer (required for SSM recurrence),
    but the gate masks the output contribution:

        output = mask * layer(x) + (1 - mask) * x

    Skipped tokens receive identity (no computation effect).
    """

    def __init__(self, layer, d_model, capacity=0.5):
        super().__init__()
        self.layer = layer
        self.gate = MoDGate(d_model, capacity)

    def forward(self, x, **kwargs):
        """
        Args:
            x: (batch, seq_len, d_model)
            **kwargs: passed through to the wrapped layer

        Returns:
            output: (batch, seq_len, d_model)
            gate_scores: for auxiliary loss
            layer_out: raw layer output tuple for state/kv_cache extraction
        """
        # Run layer on ALL tokens (SSMs need the full sequence)
        layer_result = self.layer(x, **kwargs)

        if isinstance(layer_result, tuple):
            layer_out = layer_result[0]
            rest = layer_result[1:]
        else:
            layer_out = layer_result
            rest = ()

        # Gate: which tokens get the layer's output vs skip
        mask, scores = self.gate(x)  # (B, T, 1)

        # Blend: selected tokens get layer output, skipped get identity
        output = mask * layer_out + (1 - mask) * x

        # Return along with any state/kv_cache from the inner layer
        if rest:
            return output, scores, rest
        return output, scores, ()
