"""
src/scheduler/model.py
======================
Full VNRScheduler — the top-level scoring network.

Architecture (see network_encoder_rl.md §5.4–5.5):

  substrate_data  ──► SubstrateGCN  ──► h_s  [128]
  vnr_data_list   ──► VNRGCN (×B)   ──► h_vs [B, 64]
                  ──► BatchContextEncoder ──► h_ctx [B, 64]  (if use_batch_context)

  concat([h_s_expanded, h_vs, h_ctx])  ──► ScoringMLP ──► scores [B]

Usage
-----
# Inference (no grad, sorted descending = highest priority first)
>>> scheduler = VNRScheduler.load("path/to/checkpoint.pt")
>>> scores = scheduler.predict(substrate_pyg, vnr_pyg_list)
>>> order = scores.argsort(descending=True).tolist()

# Training (grad enabled)
>>> scores = scheduler(substrate_pyg, vnr_pyg_list)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from src.scheduler.encoders import SubstrateGCN, VNRGCN, BatchContextEncoder


# ---------------------------------------------------------------------------
# Scoring MLP
# ---------------------------------------------------------------------------

class ScoringMLP(nn.Module):
    """
    Two-hidden-layer MLP that maps a combined embedding to a scalar score.

    Input dimension depends on whether batch context is used:
      - with context   : 128 + 64 + 64 = 256
      - without context: 128 + 64       = 192
    """

    def __init__(self, in_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),    nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_dim] → [B, 1]"""
        return self.net(x)


# ---------------------------------------------------------------------------
# Full VNRScheduler
# ---------------------------------------------------------------------------

class VNRScheduler(nn.Module):
    """
    GCN-RL VNR Ordering Scheduler.

    Parameters
    ----------
    use_batch_context : bool
        If True, includes BatchContextEncoder (Phase 2+).
        Phase 1 should set this to False for faster iteration.
    substrate_kwargs : dict
        Extra kwargs forwarded to SubstrateGCN (e.g. to change hidden dim).
    vnr_kwargs : dict
        Extra kwargs forwarded to VNRGCN.
    context_kwargs : dict
        Extra kwargs forwarded to BatchContextEncoder.
    """

    def __init__(
        self,
        use_batch_context: bool = True,
        substrate_kwargs: Optional[dict] = None,
        vnr_kwargs:       Optional[dict] = None,
        context_kwargs:   Optional[dict] = None,
    ):
        super().__init__()
        self.use_batch_context = use_batch_context

        sub_kw = substrate_kwargs or {}
        vnr_kw = vnr_kwargs       or {}
        ctx_kw = context_kwargs   or {}

        self.substrate_encoder = SubstrateGCN(**sub_kw)   # → [G_s, 128]
        self.vnr_encoder       = VNRGCN(**vnr_kw)         # → [B, 64]

        if use_batch_context:
            self.context_encoder: Optional[BatchContextEncoder] = BatchContextEncoder(**ctx_kw)
        else:
            self.context_encoder = None

        # Scoring MLP input dimension
        s_dim   = sub_kw.get("out_dim", 128)
        v_dim   = vnr_kw.get("out_dim", 64)
        ctx_dim = ctx_kw.get("out_dim", 64) if use_batch_context else 0
        in_dim  = s_dim + v_dim + ctx_dim

        self.scorer = ScoringMLP(in_dim=in_dim)

    # ------------------------------------------------------------------

    def forward(
        self,
        substrate_data: Data,
        vnr_data_list: List[Data],
    ) -> torch.Tensor:
        """
        Compute a score for each VNR in the current batch.

        Parameters
        ----------
        substrate_data : PyG Data
            Single substrate graph (batch dimension = 1).
        vnr_data_list : list of PyG Data
            One PyG Data per remaining VNR (can vary in size per step).

        Returns
        -------
        scores : Tensor  shape [B]
            Scalar score per VNR (higher = higher priority).
        """
        if not vnr_data_list:
            device = next(self.parameters()).device
            return torch.zeros(0, device=device)

        device = next(self.parameters()).device
        substrate_data = substrate_data.to(device)

        # --- Encode substrate (single graph → [1, 128]) ---
        h_s = self.substrate_encoder(substrate_data)   # [1, 128]

        # --- Encode all VNRs as a batch → [B, 64] ---
        vnr_batch = Batch.from_data_list(vnr_data_list).to(device)
        h_vs = self.vnr_encoder(vnr_batch)              # [B, 64]

        B = h_vs.size(0)

        # --- Optional batch context → [B, 64] ---
        if self.use_batch_context and self.context_encoder is not None:
            h_ctx = self.context_encoder(h_vs)          # [B, 64]
        else:
            h_ctx = None

        # --- Expand substrate embedding to match batch ---
        h_s_exp = h_s.expand(B, -1)                    # [B, 128]

        # --- Concatenate and score ---
        if h_ctx is not None:
            combined = torch.cat([h_s_exp, h_vs, h_ctx], dim=-1)  # [B, 256]
        else:
            combined = torch.cat([h_s_exp, h_vs], dim=-1)          # [B, 192]

        scores = self.scorer(combined).squeeze(-1)      # [B]
        return scores

    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        substrate_data: Data,
        vnr_data_list: List[Data],
    ) -> torch.Tensor:
        """
        Inference helper: evaluates the model without gradients.

        Returns
        -------
        scores : Tensor  shape [B]
        """
        self.eval()
        return self.forward(substrate_data, vnr_data_list)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path], extra_meta: Optional[dict] = None) -> None:
        """
        Save model state dict and config to ``path``.

        The saved dict has the keys:
          ``state_dict``, ``use_batch_context``, and optionally ``meta``.
        """
        ckpt = {
            "state_dict":        self.state_dict(),
            "use_batch_context": self.use_batch_context,
        }
        if extra_meta:
            ckpt["meta"] = extra_meta
        torch.save(ckpt, path)
        print(f"[VNRScheduler] Saved checkpoint → {path}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        **model_kwargs,
    ) -> "VNRScheduler":
        """
        Load a checkpoint saved with ``VNRScheduler.save()``.

        Parameters
        ----------
        path         : path to .pt file
        device       : target device (defaults to CPU)
        model_kwargs : override any constructor kwargs

        Returns
        -------
        VNRScheduler  (in eval mode)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(path, map_location=device)
        use_ctx = ckpt.get("use_batch_context", True)
        model_kwargs.setdefault("use_batch_context", use_ctx)

        model = cls(**model_kwargs)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        print(f"[VNRScheduler] Loaded checkpoint ← {path}  (device={device})")
        return model
