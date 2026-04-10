"""
src/scheduler/encoders.py
=========================
GNN encoder modules for the VNR ordering scheduler.

Architecture (see network_encoder_rl.md §5):

SubstrateGCN
  GATv2Conv(5→64, heads=4, edge_dim=2) + ReLU + LayerNorm
  GATv2Conv(256→64, heads=4, edge_dim=2) + ReLU + LayerNorm
  global_mean_pool + global_max_pool → concat → Linear → h_s [128]

VNRGCN
  GATv2Conv(4→32, heads=4, edge_dim=1) + ReLU + LayerNorm
  GATv2Conv(128→32, heads=4, edge_dim=1) + ReLU + LayerNorm
  global_mean_pool + global_max_pool → concat → Linear → h_v [64]

BatchContextEncoder  (optional, Phase 2+)
  Stack of VNR embeddings [B, 64]
  TransformerEncoder (1 layer, 4 heads) → mean pool → broadcast h_ctx [B, 64]

Design rationale
----------------
- GATv2Conv over plain GCN / GraphSAGE: attends to edge features (BW)
  which are critical for resource-aware graph reasoning.
- LayerNorm over BatchNorm: graphs have variable sizes; LayerNorm is
  batch-size-independent.
- mean+max readout: captures both average and peak node characteristics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


# ---------------------------------------------------------------------------
# Substrate encoder
# ---------------------------------------------------------------------------

class SubstrateGCN(nn.Module):
    """
    Encodes the substrate network into a fixed-size embedding h_s ∈ ℝ^{out_dim}.

    Parameters
    ----------
    in_dim   : number of node features  (default 5, see features.py)
    hidden   : attention head width     (default 64)
    heads    : number of GAT heads      (default 4)
    edge_dim : edge feature dimension   (default 2, see features.py)
    out_dim  : output embedding size    (default 128)
    """

    def __init__(
        self,
        in_dim:   int = 5,
        hidden:   int = 64,
        heads:    int = 4,
        edge_dim: int = 2,
        out_dim:  int = 128,
    ):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim,         hidden, heads=heads, edge_dim=edge_dim, concat=True)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=heads, edge_dim=edge_dim, concat=False)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.norm2 = nn.LayerNorm(hidden)
        self.act   = nn.ReLU()
        # mean + max readout → concat → project to out_dim
        self.proj  = nn.Linear(hidden * 2, out_dim)

    def forward(self, data) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch_geometric.data.Data  (or Batch)
            Expected fields: x, edge_index, edge_attr, batch

        Returns
        -------
        Tensor  shape [num_graphs, out_dim]
        """
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else \
                torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.act(self.norm1(self.conv1(x, ei, ea)))
        x = self.act(self.norm2(self.conv2(x, ei, ea)))

        g_mean = global_mean_pool(x, batch)       # [G, hidden]
        g_max  = global_max_pool(x, batch)        # [G, hidden]
        g      = torch.cat([g_mean, g_max], dim=-1)  # [G, hidden*2]
        return self.proj(g)                           # [G, out_dim]


# ---------------------------------------------------------------------------
# VNR encoder
# ---------------------------------------------------------------------------

class VNRGCN(nn.Module):
    """
    Encodes a single VNR graph into a fixed-size embedding h_v ∈ ℝ^{out_dim}.
    Handles VNRs with *no edges* (uses mean/max over node features directly).

    Parameters
    ----------
    in_dim   : number of VNR node features  (default 4, see features.py)
    hidden   : attention head width          (default 32)
    heads    : number of GAT heads           (default 4)
    edge_dim : VNR edge feature dimension    (default 1)
    out_dim  : output embedding size         (default 64)
    """

    def __init__(
        self,
        in_dim:   int = 4,
        hidden:   int = 32,
        heads:    int = 4,
        edge_dim: int = 1,
        out_dim:  int = 64,
    ):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim,         hidden, heads=heads, edge_dim=edge_dim, concat=True)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=heads, edge_dim=edge_dim, concat=False)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.norm2 = nn.LayerNorm(hidden)
        self.act   = nn.ReLU()
        self.proj  = nn.Linear(hidden * 2, out_dim)

        self._hidden = hidden

    def forward(self, data) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch_geometric.data.Data  (or Batch of VNR graphs)

        Returns
        -------
        Tensor  shape [num_graphs, out_dim]
        """
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else \
                torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # --- Fallback for graphs with no edges ---
        if ei.size(1) == 0:
            # Cannot run GATv2Conv; fall back to simple mean/max pooling over nodes
            g_mean = global_mean_pool(x, batch)   # [G, in_dim]
            g_max  = global_max_pool(x, batch)    # [G, in_dim]
            # Pad to expected hidden*2 size by projecting via a cheap linear
            hidden = self._hidden
            g = torch.cat([g_mean, g_max], dim=-1)   # [G, in_dim*2]
            # Use proj even if size mismatch — add an extra linear if needed
            if g.size(-1) != hidden * 2:
                device = x.device
                if not hasattr(self, "_fallback_proj") or \
                   self._fallback_proj.in_features != g.size(-1):
                    self._fallback_proj = nn.Linear(g.size(-1), hidden * 2).to(device)
                g = self._fallback_proj(g)
            return self.proj(g)

        x = self.act(self.norm1(self.conv1(x, ei, ea)))
        x = self.act(self.norm2(self.conv2(x, ei, ea)))

        g_mean = global_mean_pool(x, batch)
        g_max  = global_max_pool(x, batch)
        g      = torch.cat([g_mean, g_max], dim=-1)
        return self.proj(g)   # [num_graphs, out_dim]


# ---------------------------------------------------------------------------
# Batch context encoder (Phase 2+)
# ---------------------------------------------------------------------------

class BatchContextEncoder(nn.Module):
    """
    Encodes the context of ALL remaining VNRs in the current batch.

    A small Transformer lets each VNR attend to its peers so the model can
    learn inter-VNR relationships (e.g. "VNR_i is a good first pick because
    it doesn't compete with the others").

    Parameters
    ----------
    vnr_dim    : dimension of each VNR embedding  (default 64)
    out_dim    : output dimension per VNR         (default 64)
    nhead      : number of Transformer heads      (default 4)
    num_layers : number of Transformer layers     (default 1)
    """

    def __init__(
        self,
        vnr_dim:    int = 64,
        out_dim:    int = 64,
        nhead:      int = 4,
        num_layers: int = 1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vnr_dim,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj        = nn.Linear(vnr_dim, out_dim)

    def forward(self, vnr_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        vnr_embeddings : Tensor  shape [B, vnr_dim]
            Stacked embeddings of the B remaining VNRs in the batch.

        Returns
        -------
        Tensor  shape [B, out_dim]
            Per-VNR context-aware embeddings (each VNR's embedding enriched
            with global batch context).
        """
        # Transformer expects [batch, seq, dim]; treat VNRs as the sequence
        x   = vnr_embeddings.unsqueeze(0)       # [1, B, vnr_dim]
        x   = self.transformer(x).squeeze(0)    # [B, vnr_dim]
        ctx = x.mean(0, keepdim=True)           # [1, vnr_dim]  global context
        # Add global context to each VNR embedding, then project
        ctx_broad = ctx.expand(x.size(0), -1)  # [B, vnr_dim]
        return self.proj(ctx_broad)             # [B, out_dim]
