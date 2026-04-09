"""
src/rl/networks.py
==================
Graph Pointer Network for VNR Scheduling — architecture adapted from GPRL.

Differences from original GPRL (GPRL assigns physical nodes to virtual nodes):
- This network operates at *request level*: pointer points to a VNR index.
- Substrate encoder  : 3-layer GAT → mean-pool → h_p  ∈ R^128
- VNR encoder        : 3-layer GAT → max-pool  → h_vi ∈ R^64  (shared weights)
- Context MLP        : [h_p || mean(h_vi)] → context ∈ R^256  (initial GRU state)
- Pointer Decoder    : GRU cell + attention → logits over active VNR queue
- Critic Head        : [h_p || mean(h_vi)] → scalar V(s)

Key design choices (following idea.md §4):
- GAT attention weights follow GPRL eq. (α_ij with LeakyReLU)
- Max-pool VNR readout to capture "hardest" resource constraint
- Pointer attention: u^t_i = v^T tanh(W1·h_vi + W2·d_t)  (GPRL eq. 18-19)
- Masking: processed / padding slots → -inf before softmax
- Shared encoder backbone between actor and critic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch.conv as dglnn
from typing import List, Optional, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# GAT Encoder  (replicates GPRL EncoderNet with modern multi-head GAT)
# ---------------------------------------------------------------------------

class GATEncoder(nn.Module):
    """
    3-layer stacked Graph Attention Network encoder.

    Follows idea.md §4.1 / §4.2 structure:
      Layer 1: GATConv(in_feat,  hidden, heads[0], concat=True)  → hidden*h0
      Layer 2: GATConv(hidden*h0, hidden, heads[1], concat=True) → hidden*h1
      Layer 3: GATConv(hidden*h1, hidden, heads[2], mean)        → hidden

    Uses ELU after layers 1-2 (activation inside GATConv), no activation
    on layer 3 (raw embedding before pooling).

    Returns per-node embeddings of shape (total_nodes, hidden).
    """

    def __init__(self, in_feat: int, hidden: int, heads: List[int]):
        super().__init__()
        assert len(heads) == 3, "Exactly 3 head specifications required."
        h0, h1, h2 = heads

        # Layer 1: concat heads → in_feat → hidden*h0
        self.gat1 = dglnn.GATConv(
            in_feat, hidden, h0,
            allow_zero_in_degree=True,
            activation=F.elu,
        )
        # Layer 2: concat heads → hidden*h0 → hidden*h1
        self.gat2 = dglnn.GATConv(
            hidden * h0, hidden, h1,
            allow_zero_in_degree=True,
            activation=F.elu,
        )
        # Layer 3: average heads → hidden*h1 → hidden
        self.gat3 = dglnn.GATConv(
            hidden * h1, hidden, h2,
            allow_zero_in_degree=True,
            activation=None,
        )
        self.out_dim = hidden

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        g    : DGLGraph (can be a batched graph)
        feat : (total_nodes, in_feat)

        Returns
        -------
        h    : (total_nodes, hidden)
        """
        h = self.gat1(g, feat)    # (N, h0, hidden)
        h = h.flatten(1)           # (N, h0*hidden)
        h = self.gat2(g, h)       # (N, h1, hidden)
        h = h.flatten(1)           # (N, h1*hidden)
        h = self.gat3(g, h)       # (N, h2, hidden)
        h = h.mean(dim=1)          # (N, hidden)  — average over final heads
        return h


# ---------------------------------------------------------------------------
# Substrate Encoder   →  h_p ∈ R^{substrate_hidden}
# ---------------------------------------------------------------------------

class SubstrateEncoder(nn.Module):
    """
    Encodes the current substrate graph into a single global vector h_p.

    Architecture (idea.md §4.1):
        GATEncoder(3 layers) → mean-pool over all physical nodes → h_p

    Mean-pool captures the average resource availability across all nodes,
    which is the relevant signal for scheduling decisions.
    """

    def __init__(self, node_feat_dim: int, hidden: int, heads: List[int]):
        super().__init__()
        self.gat = GATEncoder(node_feat_dim, hidden, heads)
        self.out_dim = hidden

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        """
        g  : single substrate DGLGraph
             g.ndata['nfeat'] must be set: shape (N_p, node_feat_dim)
        Returns: (1, hidden)  — global substrate embedding (mean-pooled)
        """
        feat = g.ndata['nfeat']            # (N_p, node_feat_dim)
        h    = self.gat(g, feat)            # (N_p, hidden)
        h_p  = h.mean(dim=0, keepdim=True)  # (1, hidden)
        return h_p


# ---------------------------------------------------------------------------
# VNR Encoder   →  h_vi ∈ R^{vnr_hidden}
# ---------------------------------------------------------------------------

class VNREncoder(nn.Module):
    """
    Encodes a single VNR into a fixed-size embedding h_vi.

    Architecture (idea.md §4.2):
        GATEncoder(3 layers) → max-pool over virtual nodes → h_vi

    Max-pool captures the "hardest" (most demanding) virtual node constraint,
    which is the bottleneck for HPSO embedding success. Shared weights across
    all VNRs in the queue — enables variable queue size without retraining.

    Optional extended feature: h_vi can be extended with scalar summary stats
    (idea.md §4.2 "Optional"):
        feat_vi = [h_vi || log(|N_vi|) || max_cpu_demand || max_bw_demand]
    Set `use_summary_feats=True` to enable.
    """

    def __init__(
        self,
        node_feat_dim: int,
        hidden: int,
        heads: List[int],
        use_summary_feats: bool = False,
    ):
        super().__init__()
        self.gat = GATEncoder(node_feat_dim, hidden, heads)
        self.use_summary_feats = use_summary_feats
        # Output dim: hidden (+ 3 summary scalars if enabled)
        self.out_dim = hidden + 3 if use_summary_feats else hidden

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        """
        g  : single VNR DGLGraph
             g.ndata['nfeat'] must be set: shape (N_v, node_feat_dim)
        Returns: (1, out_dim)
        """
        feat = g.ndata['nfeat']              # (N_v, node_feat_dim)
        h    = self.gat(g, feat)              # (N_v, hidden)
        h_vi, _ = h.max(dim=0, keepdim=True)  # (1, hidden)  — max-pool

        if self.use_summary_feats:
            n_nodes   = torch.tensor([[float(g.num_nodes())]],
                                      device=h.device)
            log_n     = torch.log(n_nodes + 1.0)  # (1,1)
            cpu_max   = feat[:, 0:1].max(dim=0, keepdim=True).values  # (1,1) cpu col
            bw_max_val = feat[:, 1:2].max(dim=0, keepdim=True).values if feat.shape[1] > 1 \
                         else torch.zeros(1, 1, device=h.device)         # (1,1) bw col
            h_vi = torch.cat([h_vi, log_n, cpu_max, bw_max_val], dim=-1)  # (1, hidden+3)

        return h_vi


# ---------------------------------------------------------------------------
# Context MLP  →  initial GRU hidden state
# ---------------------------------------------------------------------------

class ContextMLP(nn.Module):
    """
    Combines substrate and VNR-queue information into a context vector used
    to initialise the GRU decoder hidden state (idea.md §4.3).

    context = MLP([h_p || mean({h_vi : i active})]) ∈ R^{context_dim}
    """

    def __init__(self, substrate_dim: int, vnr_dim: int, context_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(substrate_dim + vnr_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        h_p: torch.Tensor,           # (1, substrate_dim)
        h_queue_mean: torch.Tensor,  # (1, vnr_dim)
    ) -> torch.Tensor:
        """Returns (1, context_dim)."""
        x = torch.cat([h_p, h_queue_mean], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Key Projection  →  project VNR embeddings to GRU hidden space
# ---------------------------------------------------------------------------

class KeyProjection(nn.Module):
    """Projects R^{vnr_dim} → R^{key_dim} for pointer attention keys."""

    def __init__(self, vnr_dim: int, key_dim: int):
        super().__init__()
        self.proj = nn.Linear(vnr_dim, key_dim)

    def forward(self, h_vi: torch.Tensor) -> torch.Tensor:
        return self.proj(h_vi)


# ---------------------------------------------------------------------------
# Pointer Decoder  (GRU + attention, idea.md §4.4)
# ---------------------------------------------------------------------------

class PointerDecoder(nn.Module):
    """
    Implements the pointer network decoder step.

    At each scheduling step t:
      1. GRU(input=last_selected_key, hidden=h_{t-1}) → d_t  (decoder state)
      2. u^t_i = v^T tanh(W1·h_vi + W2·d_t)           (GPRL eq. 18-19)
      3. Apply mask (set -inf for processed / padding slots)
      4. Return raw logits (caller applies softmax and samples)
    """

    def __init__(self, key_dim: int, gru_hidden: int):
        super().__init__()
        self.gru = nn.GRUCell(key_dim, gru_hidden)

        # Pointer attention parameters
        self.W1 = nn.Linear(key_dim,    gru_hidden, bias=False)
        self.W2 = nn.Linear(gru_hidden, gru_hidden, bias=False)
        self.v  = nn.Linear(gru_hidden, 1,          bias=False)

    def forward(
        self,
        keys:     torch.Tensor,   # (K_max, key_dim)  — VNR embedding keys
        h_prev:   torch.Tensor,   # (1,     gru_hidden) — previous GRU state
        last_key: torch.Tensor,   # (1,     key_dim)   — last chosen key
        mask:     torch.Tensor,   # (K_max,)  bool      — True = selectable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits  : (K_max,)   raw attention scores (masked)
        h_next  : (1, gru_hidden)
        """
        # --- GRU step ---
        h_next = self.gru(last_key, h_prev)          # (1, gru_hidden)

        # --- Pointer attention (GPRL eq. 18-19) ---
        keys_proj = self.W1(keys)                     # (K_max, gru_hidden)
        dec_proj  = self.W2(h_next)                   # (1,     gru_hidden)
        energy    = self.v(torch.tanh(keys_proj + dec_proj))  # (K_max, 1)
        logits    = energy.squeeze(-1)                # (K_max,)

        # --- Mask unavailable slots ---
        logits = logits.masked_fill(~mask, float('-inf'))

        return logits, h_next


# ---------------------------------------------------------------------------
# Critic Head  (idea.md §5)
# ---------------------------------------------------------------------------

class CriticHead(nn.Module):
    """
    Critic value-function head (shared backbone with actor).

    V(s_t) = MLP([h_p || mean({h_vi : i active})])

    Architecture (idea.md §5.1):
        Linear(192→128) → ReLU → Linear(128→64) → ReLU → Linear(64→1)
    """

    def __init__(self, substrate_dim: int, vnr_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(substrate_dim + vnr_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        h_p: torch.Tensor,           # (1, substrate_dim)
        h_queue_mean: torch.Tensor,  # (1, vnr_dim)
    ) -> torch.Tensor:
        """Returns scalar V(s_t), shape (1,)."""
        x = torch.cat([h_p, h_queue_mean], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Full VNR Scheduler Network  (assembles all components)
# ---------------------------------------------------------------------------

class VNRSchedulerNetwork(nn.Module):
    """
    Complete actor-critic network for VNR scheduling.

    Composed of:
      - SubstrateEncoder  (GAT, shared with critic)
      - VNREncoder        (GAT, shared with critic, weight-shared across VNRs)
      - ContextMLP        (initial GRU hidden state)
      - KeyProjection     (VNR embeddings → pointer key space)
      - PointerDecoder    (GRU + pointer attention)
      - CriticHead        (value function)

    Parameters
    ----------
    cfg : dict
        Configuration dictionary (see DEFAULT_CFG in utils.py).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        c = cfg  # alias

        self.K_max      = c['K_max']
        self.vnr_dim    = c.get('vnr_hidden', 64)
        self.sub_dim    = c.get('substrate_hidden', 128)
        self.gru_hidden = c.get('gru_hidden', 256)

        # --- Shared encoders (GPRL backbone adapted for VNR scheduling) ---
        self.substrate_enc = SubstrateEncoder(
            node_feat_dim = c.get('substrate_node_feat_dim', 4),
            hidden        = self.sub_dim,
            heads         = c.get('substrate_gat_heads', [4, 4, 1]),
        )
        self.vnr_enc = VNREncoder(
            node_feat_dim    = c.get('vnr_node_feat_dim', 3),
            hidden           = self.vnr_dim,
            heads            = c.get('vnr_gat_heads', [4, 4, 1]),
            use_summary_feats= c.get('use_summary_feats', False),
        )
        # Actual VNR embedding dim (may include summary features)
        vnr_out_dim = self.vnr_enc.out_dim

        # --- Context / key projection ---
        context_dim = c.get('context_dim', 256)
        self.context_mlp = ContextMLP(self.sub_dim, vnr_out_dim, context_dim)
        self.key_proj    = KeyProjection(vnr_out_dim, self.gru_hidden)

        # --- Actor: Pointer Decoder ---
        self.pointer = PointerDecoder(self.gru_hidden, self.gru_hidden)

        # Learnable start token: embedding of "no previous selection"
        self.start_token = nn.Parameter(torch.zeros(1, self.gru_hidden))

        # --- Critic head ---
        critic_hidden = c.get('critic_hidden', 128)
        self.critic = CriticHead(self.sub_dim, vnr_out_dim, critic_hidden)

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Do NOT reinitialise start_token (keeps zero init)

    # ------------------------------------------------------------------
    # Encode helpers
    # ------------------------------------------------------------------

    def encode_substrate(self, g_sub: dgl.DGLGraph) -> torch.Tensor:
        """Returns h_p ∈ (1, substrate_hidden)."""
        return self.substrate_enc(g_sub)

    def encode_vnr_queue(
        self,
        vnr_graphs: List[Optional[dgl.DGLGraph]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode K_max VNR slots (None entries are zero-padded).

        Returns
        -------
        keys   : (K_max, gru_hidden)   — pointer attention keys
        h_stack: (K_max, vnr_out_dim)  — raw VNR embeddings (for critic/context)
        """
        embeddings = []
        vnr_out_dim = self.vnr_enc.out_dim
        for g in vnr_graphs:
            if g is not None:
                h = self.vnr_enc(g.to(device))    # (1, vnr_out_dim)
            else:
                h = torch.zeros(1, vnr_out_dim, device=device)
            embeddings.append(h)

        h_stack = torch.cat(embeddings, dim=0)     # (K_max, vnr_out_dim)
        keys    = self.key_proj(h_stack)            # (K_max, gru_hidden)
        return keys, h_stack

    # ------------------------------------------------------------------
    # Single forward step (used during rollout & evaluate_actions)
    # ------------------------------------------------------------------

    def step(
        self,
        h_p:      torch.Tensor,   # (1, sub_dim)
        keys:     torch.Tensor,   # (K_max, gru_hidden)
        h_stack:  torch.Tensor,   # (K_max, vnr_out_dim)
        mask:     torch.Tensor,   # (K_max,) bool
        h_gru:    torch.Tensor,   # (1, gru_hidden)
        last_key: torch.Tensor,   # (1, gru_hidden)
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One scheduling step: choose next VNR from active queue.

        Returns
        -------
        action    : int   — chosen VNR slot index in [0, K_max)
        log_prob  : (1,)  — log π(action|state)
        value     : (1,)  — V(state)
        h_gru_new : (1, gru_hidden)
        """
        # --- Actor (pointer decoder) ---
        logits, h_gru_new = self.pointer(keys, h_gru, last_key, mask)
        probs = F.softmax(logits, dim=0)
        dist  = torch.distributions.Categorical(probs)

        if deterministic:
            action = int(logits.argmax().item())
        else:
            action = int(dist.sample().item())

        log_prob = dist.log_prob(torch.tensor(action, device=device))

        # --- Critic ---
        active_h = h_stack[mask]
        h_queue_mean = active_h.mean(dim=0, keepdim=True) if active_h.numel() > 0 \
                       else torch.zeros(1, h_stack.shape[-1], device=device)
        value = self.critic(h_p, h_queue_mean)

        return action, log_prob, value, h_gru_new
