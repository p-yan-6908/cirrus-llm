"""Cirrus: ToolMoE-SSM v2 model.

A hybrid SSM/Attention architecture with Mixture-of-Experts,
Mixture-of-Depths, and tool-native agentic capabilities.
Designed for 8GB RAM inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CirrusConfig
from .ssm import Mamba2Layer
from .attention import SlidingWindowAttention
from .moe import CirrusFFN
from .mod import MoDWrapper
from .memory import ScratchpadMemory, ToolResultCache
from .tools import (
    ToolSchemaEncoder,
    GrammarConstrainedDecoder,
    MODE_TOKENS,
)


class CirrusLayer(nn.Module):
    """Single transformer block for Cirrus.

    Heterogeneous backbone: SSM or Attention + FFN (dense or MoE) + MoD.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_attention = (
            layer_idx >= config.attention_start_layer
            and layer_idx < config.attention_start_layer + config.n_attention_layers
        )

        # Pre-norm
        self.norm1 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # Core layer: SSM or Attention
        if self.is_attention:
            self.core = SlidingWindowAttention(config)
        else:
            self.core = Mamba2Layer(config)

        # MoD wrapper (applies to both SSM and Attention)
        if config.mod_enabled:
            self.core = MoDWrapper(self.core, config.d_model, config.mod_capacity)

        # FFN: Dense or MoE
        self.ffn = CirrusFFN(config, layer_idx)

    def forward(self, x, state=None, kv_cache=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            state: SSM state (for SSM layers)
            kv_cache: KV cache (for attention layers)

        Returns:
            output: (batch, seq_len, d_model)
            new_state: updated SSM state
            new_kv_cache: updated KV cache
            aux_loss: MoE load balancing loss
        """
        aux_loss = torch.tensor(0.0, device=x.device)
        new_state = state
        new_kv_cache = kv_cache

        residual = x
        x_norm = self.norm1(x)

        if isinstance(self.core, MoDWrapper):
            # MoD-gated: pass kwargs through to inner layer
            mod_kwargs = {}
            if not self.is_attention and state is not None:
                mod_kwargs["state"] = state
            elif self.is_attention and kv_cache is not None:
                mod_kwargs["kv_cache"] = kv_cache

            x_core, gate_scores, rest = self.core(x_norm, **mod_kwargs)

            # Extract state/kv_cache from inner layer's output
            if rest:
                if not self.is_attention:
                    new_state = rest[0]
                else:
                    new_kv_cache = rest[0]
        else:
            # Direct layer (no MoD)
            if self.is_attention:
                x_core, new_kv_cache = self.core(x_norm, kv_cache=kv_cache)
            else:
                x_core, new_state = self.core(x_norm, state=state)

        x = residual + x_core

        # FFN
        residual = x
        x_norm = self.norm2(x)
        ffn_result = self.ffn(x_norm)

        if isinstance(ffn_result, tuple):
            ffn_out, aux_loss = ffn_result
        else:
            ffn_out = ffn_result

        x = residual + ffn_out

        return x, new_state, new_kv_cache, aux_loss


class CirrusModel(nn.Module):
    """Cirrus: ToolMoE-SSM v2.

    Architecture:
        - 48 layers: 42 SSM + 6 Attention (7:1 ratio, attention at the end)
        - MoE with adaptive expert groups (layers 16-48)
        - Mixture of Depths (all layers, 50% capacity)
        - 3-layer memory: SSM state + scratchpad + tool cache
        - Tool schema encoder (1 token per tool)
        - Mode tokens + grammar-constrained decoding
    """

    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = CirrusConfig(**kwargs)
        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Layers
        self.layers = nn.ModuleList([
            CirrusLayer(config, i) for i in range(config.n_layers)
        ])

        # Final norm
        self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # Language model head (unified — handles modes via special tokens)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embedding and lm_head weights
        self.lm_head.weight = self.embedding.weight

        # Tool schema encoder
        self.tool_encoder = ToolSchemaEncoder(config)

        # Scratchpad memory
        self.scratchpad = ScratchpadMemory(config)

        # Runtime tool cache (not a module parameter)
        self.tool_cache = ToolResultCache(config.tool_cache_max_tokens)

        # Mode token IDs (to be set after tokenization)
        self._mode_token_ids = {}

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def set_mode_token_ids(self, token_ids):
        """Set mode token vocabulary IDs after tokenizer setup.

        Args:
            token_ids: dict mapping mode token string → vocab id
        """
        self._mode_token_ids = token_ids

    def register_tools(self, tools):
        """Register tool schemas and encode them as tokens.

        Args:
            tools: list of ToolSchema objects

        Returns:
            tool_tokens: (1, n_tools, d_model)
        """
        return self.tool_encoder(tools)

    def get_grammar_decoder(self, tool_names):
        """Get a grammar-constrained decoder for inference."""
        return GrammarConstrainedDecoder(tool_names, self._mode_token_ids)

    def forward(
        self,
        input_ids,
        tool_tokens=None,
        scratchpad_update=None,
        states=None,
        kv_caches=None,
    ):
        """
        Args:
            input_ids: (batch, seq_len)
            tool_tokens: optional (1, n_tools, d_model) from tool encoder
            scratchpad_update: optional (batch, n_update, d_model) for memory write
            states: list of SSM states for each SSM layer
            kv_caches: list of KV caches for each attention layer

        Returns:
            logits: (batch, seq_len, vocab_size)
            new_states: updated SSM states
            new_kv_caches: updated KV caches
            aux_loss: total MoE load balancing loss
        """
        B, T = input_ids.shape

        # Token embedding
        x = self.embedding(input_ids)  # (B, T, d_model)

        # Prepend scratchpad memory
        memory_tokens = self.scratchpad(x, update_info=scratchpad_update)
        if memory_tokens is not None:
            x = torch.cat([memory_tokens, x], dim=1)

        # Prepend tool tokens
        if tool_tokens is not None:
            n_tools = tool_tokens.shape[1]
            # Expand for batch
            tool_tok = tool_tokens.expand(B, -1, -1)
            x = torch.cat([tool_tok, x], dim=1)

        # Track which positions are tool tokens and memory tokens
        n_prefix = 0
        if tool_tokens is not None:
            n_prefix += n_tools
        if memory_tokens is not None:
            n_prefix += self.config.scratchpad_n_tokens

        # Initialize states/caches
        if states is None:
            states = [None] * self.config.n_layers
        if kv_caches is None:
            kv_caches = [None] * self.config.n_layers

        # Forward through layers
        total_aux_loss = torch.tensor(0.0, device=x.device)
        new_states = []
        new_kv_caches = []

        for i, layer in enumerate(self.layers):
            x, new_state, new_kv_cache, aux_loss = layer(
                x, state=states[i], kv_cache=kv_caches[i]
            )
            new_states.append(new_state)
            new_kv_caches.append(new_kv_cache)
            total_aux_loss = total_aux_loss + aux_loss

        # Final norm
        x = self.norm(x)

        # Remove prefix tokens for LM head
        if n_prefix > 0:
            x_main = x[:, n_prefix:, :]
        else:
            x_main = x

        # LM head
        logits = self.lm_head(x_main)

        return logits, new_states, new_kv_caches, total_aux_loss

    def generate_step(
        self,
        input_ids,
        tool_tokens=None,
        states=None,
        kv_caches=None,
        grammar_decoder=None,
        temperature=1.0,
        top_p=0.9,
    ):
        """Single-step autoregressive generation.

        Args:
            input_ids: (batch, 1) — single token
            tool_tokens: optional pre-encoded tool tokens
            states: SSM states
            kv_caches: KV caches
            grammar_decoder: optional GrammarConstrainedDecoder
            temperature: sampling temperature
            top_p: nucleus sampling threshold

        Returns:
            next_token: (batch, 1)
            new_states: updated states
            new_kv_caches: updated caches
        """
        logits, new_states, new_kv_caches, _ = self.forward(
            input_ids,
            tool_tokens=tool_tokens,
            states=states,
            kv_caches=kv_caches,
        )

        # Get logits for last position
        next_logits = logits[:, -1, :]  # (B, vocab_size)

        # Apply grammar constraints
        if grammar_decoder is not None:
            mask = grammar_decoder.get_valid_tokens(
                self.config.vocab_size,
                tool_name_token_ids=self._mode_token_ids,
            )
            mask = mask.to(next_logits.device)
            next_logits = next_logits.masked_fill(~mask, float("-inf"))

        # Temperature
        if temperature > 0:
            next_logits = next_logits / temperature

            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        return next_token, new_states, new_kv_caches

    def count_parameters(self):
        """Count total and active parameters."""
        total = sum(p.numel() for p in self.parameters())
        # Rough estimate of active params per token
        # SSM layers + active experts + attention (when processing)
        active = 0
        for layer in self.layers:
            # Core layer params
            if isinstance(layer.core, MoDWrapper):
                core_params = sum(p.numel() for p in layer.core.layer.parameters())
            else:
                core_params = sum(p.numel() for p in layer.core.parameters())
            active += core_params

            # FFN (dense: all, MoE: only active experts)
            if layer.ffn.is_moe:
                m = layer.ffn.ffn
                if hasattr(m, 'experts'):
                    # Avg 1.8 experts active
                    expert_params = sum(p.numel() for p in m.experts[0].parameters())
                    active += int(expert_params * 1.8)
                active += sum(p.numel() for p in m.router.parameters())
            else:
                active += sum(p.numel() for p in layer.ffn.ffn.parameters())

        # Embedding, norm, lm_head
        active += sum(p.numel() for p in self.embedding.parameters())
        active += sum(p.numel() for p in self.norm.parameters())

        return {
            "total": total,
            "active_estimate": active,
            "total_gb": total * 2 / (1024**3),  # bf16
            "active_gb": active * 2 / (1024**3),
        }

    def clear_memory(self):
        """Clear scratchpad and tool cache for a new conversation."""
        self.scratchpad.reset()
        self.tool_cache.clear()
