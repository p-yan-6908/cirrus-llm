"""Cirrus: ToolMoE-SSM v2

A hybrid SSM/Attention architecture with Mixture-of-Experts,
Mixture-of-Depths, and tool-native agentic capabilities.
Designed for 8GB RAM inference.

Usage:
    from cirrus import CirrusModel, CirrusConfig

    config = CirrusConfig()
    model = CirrusModel(config)

    # Or with custom dimensions:
    model = CirrusModel(CirrusConfig(d_model=1024, n_layers=24))
"""

from .config import CirrusConfig
from .model import CirrusModel
from .ssm import Mamba2Layer
from .attention import SlidingWindowAttention
from .moe import MoELayer, CirrusFFN
from .mod import MoDWrapper, MoDGate
from .memory import ScratchpadMemory, ToolResultCache
from .tools import ToolSchema, ToolSchemaEncoder, GrammarConstrainedDecoder
from .training import CirrusTrainer, SyntheticToolTrajectoryGenerator

__all__ = [
    "CirrusConfig",
    "CirrusModel",
    "Mamba2Layer",
    "SlidingWindowAttention",
    "MoELayer",
    "CirrusFFN",
    "MoDWrapper",
    "MoDGate",
    "ScratchpadMemory",
    "ToolResultCache",
    "ToolSchema",
    "ToolSchemaEncoder",
    "GrammarConstrainedDecoder",
    "CirrusTrainer",
    "SyntheticToolTrajectoryGenerator",
]
