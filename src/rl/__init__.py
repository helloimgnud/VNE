"""
src/rl
======
RL-based VNR ordering module.

Provides a Graph Pointer Network agent (adapted from GPRL) that learns to
order VNRs in a time window before feeding them to the HPSO solver.

Public API
----------
>>> from src.rl import VNRSchedulerAgent, PPOTrainer, SchedulerEnv
>>> from src.rl import build_vnr_dgl, build_substrate_dgl
"""

from src.rl.networks import (
    GATEncoder,
    SubstrateEncoder,
    VNREncoder,
    ContextMLP,
    KeyProjection,
    PointerDecoder,
    CriticHead,
    VNRSchedulerNetwork,
)
from src.rl.agent import VNRSchedulerAgent
from src.rl.env import SchedulerEnv
from src.rl.trainer import PPOTrainer
from src.rl.utils import build_vnr_dgl, build_substrate_dgl, DEFAULT_CFG

__all__ = [
    # Network components
    "GATEncoder",
    "SubstrateEncoder",
    "VNREncoder",
    "ContextMLP",
    "KeyProjection",
    "PointerDecoder",
    "CriticHead",
    "VNRSchedulerNetwork",
    # High-level
    "VNRSchedulerAgent",
    "SchedulerEnv",
    "PPOTrainer",
    # Utilities
    "build_vnr_dgl",
    "build_substrate_dgl",
    "DEFAULT_CFG",
]
