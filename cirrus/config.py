"""Cirrus model configuration.

Cirrus (ToolMoE-SSM v2) is designed for 8GB RAM inference with:
- 7:1 SSM:Attention ratio (Mamba-2 + sliding window attention)
- Adaptive Mixture-of-Experts (3 expert groups, 1-3 active per token)
- Mixture of Depths (50% token skipping)
- 3-layer memory system (SSM state + scratchpad + tool cache)
- Tool-native integration (schema encoder, mode tokens, grammar decoding)

Target: ~10.5B total params, ~2B active per token, ~5GB at 4-bit.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CirrusConfig:
    """Configuration for the Cirrus architecture.

    Can be constructed from a dict via CirrusConfig.from_dict() or
    via keyword args: CirrusConfig(d_model=1024, n_layers=24).

    Named presets:
        CirrusConfig.tiny()      — ~25M params, for testing
        CirrusConfig.small()     — ~500M params, for development
        CirrusConfig.base_10b()  — ~10.5B params, default target
    """

    # --- Model dimensions ---
    d_model: int = 3072
    vocab_size: int = 32000
    n_layers: int = 48
    rms_norm_eps: float = 1e-5

    # --- Backbone: 7:1 SSM:Attention ratio ---
    # Attention layers are the LAST 6 layers (layers 42-48)
    n_attention_layers: int = 6
    attention_start_layer: int = 42  # 0-indexed

    # --- Sliding window attention ---
    window_size: int = 2048
    n_heads: int = 24
    head_dim: int = 128  # d_model // n_heads = 3072/24 = 128
    n_kv_heads: int = 6  # GQA: 6 KV heads for 24 Q heads (4:1 ratio)

    # --- Mamba-2 SSM ---
    ssm_d_state: int = 128
    ssm_d_conv: int = 4
    ssm_expand: float = 2.0  # d_inner = d_model * expand = 6144

    # --- MoE: Staged expert placement ---
    # Layers 0-15:  dense FFN (shared features)
    # Layers 16-31: MoE with 8 experts (moderate specialization)
    # Layers 32-47: MoE with 12 experts in 3 groups (full)
    moe_start_layer: int = 16
    moe_mid_experts: int = 8
    moe_full_start_layer: int = 32
    moe_full_experts: int = 12

    # --- Expert groups (full MoE layers only) ---
    # Group A (general): E0-E3, always active, top-1
    # Group B (reasoning): E4-E7, conditional, top-1
    # Group C (tool): E8-E11, conditional, top-2
    expert_groups: tuple = (
        ("general", 0, 4),
        ("reasoning", 4, 8),
        ("tool", 8, 12),
    )
    expert_dim: int = 2560  # FFN hidden dim per expert
    group_gate_threshold: float = 0.15

    # --- Mixture of Depths ---
    mod_capacity: float = 0.5  # process 50% of tokens per layer
    mod_enabled: bool = True

    # --- 3-layer memory system ---
    scratchpad_n_tokens: int = 32
    tool_cache_max_tokens: int = 2048

    # --- Tool schema encoder ---
    tool_encoder_layers: int = 2
    tool_encoder_dim: int = 3072
    max_tools: int = 64

    # --- Mode tokens ---
    mode_tokens: tuple = (
        "<THINK>", "<TOOL_CALL>", "<TOOL_NAME>",
        "<TOOL_ARGS>", "<TOOL_END>", "<CONFIDENCE=",
        "<DONE>",
    )

    # --- Quantization-friendly ---
    qk_norm: bool = True
    clip_activation: float = 8.0  # clipped SiLU ceiling

    # --- Training ---
    n_expert_phases: tuple = (2, 4, 12)
    expert_phase_epochs: tuple = (2, 2, None)
    tool_data_fraction: float = 0.1

    def __post_init__(self):
        # Auto-derive head_dim if not set correctly
        if self.head_dim != self.d_model // self.n_heads:
            self.head_dim = self.d_model // self.n_heads

    def _validate(self):
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        assert self.attention_start_layer + self.n_attention_layers <= self.n_layers
        assert self.moe_full_start_layer > self.moe_start_layer

    @classmethod
    def from_dict(cls, d: dict) -> "CirrusConfig":
        """Create config from a dictionary, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Export config as a dictionary."""
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    @classmethod
    def tiny(cls) -> "CirrusConfig":
        """~25M params — for unit tests and debugging."""
        return cls(
            d_model=256, vocab_size=1000, n_layers=12,
            n_attention_layers=2, attention_start_layer=10,
            n_heads=4, head_dim=64, n_kv_heads=2,
            ssm_d_state=32, ssm_d_conv=4,
            moe_start_layer=4, moe_mid_experts=4,
            moe_full_start_layer=8, moe_full_experts=6,
            expert_dim=128, scratchpad_n_tokens=8,
            tool_cache_max_tokens=128, tool_encoder_dim=256,
        )

    @classmethod
    def small(cls) -> "CirrusConfig":
        """~500M params — for development and experimentation."""
        return cls(
            d_model=1024, vocab_size=32000, n_layers=24,
            n_attention_layers=3, attention_start_layer=21,
            n_heads=8, head_dim=128, n_kv_heads=2,
            ssm_d_state=64, ssm_d_conv=4,
            moe_start_layer=8, moe_mid_experts=6,
            moe_full_start_layer=16, moe_full_experts=8,
            expert_dim=512, scratchpad_n_tokens=16,
            tool_cache_max_tokens=1024, tool_encoder_dim=1024,
        )

    @classmethod
    def base_10b(cls) -> "CirrusConfig":
        """~10.5B params — the default 8GB target."""
        return cls()  # uses all defaults
