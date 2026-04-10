"""
src/scheduler
=============
GCN-RL VNR Ordering Scheduler — PyG-based implementation.

Architecture
------------
  SubstrateGCN  (GATv2Conv × 2)  → h_s [128]
  VNRGCN        (GATv2Conv × 2)  → h_v [64]
  BatchContextEncoder (Transformer) → h_ctx [64]  (optional)
  ScoringMLP    → score per VNR   [1]

The full pipeline is exposed through VNRScheduler.

Public API
----------
>>> from src.scheduler import VNRScheduler
>>> from src.scheduler.features import substrate_to_pyg, vnr_to_pyg
>>> from src.scheduler.environment import VNEOrderingEnv
"""

from src.scheduler.features import substrate_to_pyg, vnr_to_pyg
from src.scheduler.encoders import SubstrateGCN, VNRGCN, BatchContextEncoder
from src.scheduler.model import VNRScheduler, ScoringMLP
from src.scheduler.environment import VNEOrderingEnv
from src.scheduler.rewards import RewardMode

__all__ = [
    "substrate_to_pyg",
    "vnr_to_pyg",
    "SubstrateGCN",
    "VNRGCN",
    "BatchContextEncoder",
    "ScoringMLP",
    "VNRScheduler",
    "VNEOrderingEnv",
    "RewardMode",
]
