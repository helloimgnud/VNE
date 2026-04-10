# GCN-RL VNR Ordering Scheduler — Design Document

## 1. Problem Statement

### Current Pipeline
```
batch VNRs → sort by revenue (greedy) → hpso_embed_batch → sequential embedding
```

### Why Ordering Matters
The HPSO embedder is **stateful**: resources consumed by embedding VNR_i reduce substrate capacity for all subsequent VNRs. Revenue-first ordering is myopic — it maximizes short-term gain but can block high-compatibility groups.

**Example:**
```
Substrate: 3 clusters of nodes, each with ~50 CPU / 100 BW

VNR1: 45 CPU, 90 BW  → monopolizes cluster A
VNR2: 20 CPU, 40 BW  → fits cluster B
VNR4: 22 CPU, 45 BW  → fits cluster B alongside VNR2
VNR5: 18 CPU, 38 BW  → fits cluster B alongside VNR2 + VNR4

Revenue sort → embeds VNR1 first → wastes cluster A, blocks nothing useful
                                  → VNR2,4,5 still fit B → OK in this case

But consider:
VNR1: 45 CPU, spans multiple clusters → fragments BOTH A and B
VNR2+4+5: tightly pack into cluster B → all accepted

Revenue sort picks VNR1 → fragments → VNR4 and VNR5 fail
Learned sort picks VNR2 first → compact → all three accepted
```

### Goal
Learn a **scoring function** `Score(VNR_i | Substrate_t, Batch)` that, when used to sort VNRs, maximizes:
- **Acceptance Rate (AR)**: fraction of VNRs successfully embedded
- **Long-term Revenue-to-Cost Ratio (R/C)**: total revenue / total resource cost

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     VNR Ordering Scheduler                      │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────┐  │
│  │ Substrate    │   │  VNR Batch   │   │  Batch Context    │  │
│  │ GCN Encoder  │   │  GCN Encoder │   │  (Optional Attn)  │  │
│  └──────┬───────┘   └──────┬───────┘   └────────┬──────────┘  │
│         │                  │                    │              │
│         └──────────────────┴────────────────────┘              │
│                            │                                   │
│                     ┌──────▼──────┐                           │
│                     │  Scoring    │                           │
│                     │    MLP      │                           │
│                     └──────┬──────┘                           │
│                            │                                   │
│                     score per VNR                             │
└────────────────────────────┼────────────────────────────────────┘
                             │
                    sort descending
                             │
                    ┌────────▼────────┐
                    │ hpso_embed_batch│
                    │  (unchanged)    │
                    └─────────────────┘
```

---

## 3. Technology Stack

| Component | Library | Notes |
|-----------|---------|-------|
| GNN / GCN | `torch_geometric` (PyG) ≥ 2.5 | Modern, actively maintained |
| RL Training | `stable-baselines3` ≥ 2.3 + `gymnasium` ≥ 0.29 | PPO/A2C out-of-the-box |
| Deep Learning | `torch` ≥ 2.2 | CUDA optional |
| Graph ops | `networkx` | Already in your stack |
| Experiment tracking | `tensorboard` or `wandb` | Pick one |

**Why these?**
- PyG 2.x has stable `MessagePassing`, `global_mean_pool`, `GATv2Conv` — no deprecated APIs
- SB3 2.x is fully gymnasium-compatible (gym is deprecated)
- Avoid `torch_geometric` < 2.0, `openai-gym`, `ray[rllib]` (heavy) for initial work

---

## 4. Feature Engineering

### 4.1 Substrate Node Features (per node `s`)
```python
# Shape: [num_substrate_nodes, 5]
features = [
    cpu_available,           # current available CPU (dynamic)
    cpu_ratio,               # cpu_available / cpu_total
    degree,                  # number of substrate edges
    avg_bw_neighbors,        # mean BW of incident edges
    clustering_coefficient,  # local cluster density
]
```

### 4.2 Substrate Edge Features (per edge `(u,v)`)
```python
# Shape: [num_substrate_edges, 2]
features = [
    bw_available,            # current available BW (dynamic)
    bw_ratio,                # bw_available / bw_total
]
```

### 4.3 VNR Node Features (per virtual node `v`)
```python
# Shape: [num_vnr_nodes, 4]
features = [
    cpu_demand,
    degree_in_vnr,
    sum_bw_incident,         # sum of BW on all edges touching this node
    cpu_demand / total_vnr_cpu,  # relative demand
]
```

### 4.4 VNR Edge Features (per virtual edge `(u,v)`)
```python
# Shape: [num_vnr_edges, 1]
features = [
    bw_demand,
]
```

### 4.5 VNR Graph-level Features (after readout)
```python
# Computed from above, used in scoring MLP
[
    total_cpu_demand,
    total_bw_demand,
    revenue,                 # cpu + bw (from eval.py)
    num_nodes,
    num_edges,
    avg_node_cpu,
    max_node_cpu,
]
```

---

## 5. Model Architecture

### 5.1 Substrate GCN Encoder
```
Input: (node_feats [N_s, 5], edge_feats [E_s, 2], edge_index)
   │
   ├─ GATv2Conv(5 → 64, heads=4, edge_dim=2)  + ReLU + LayerNorm
   ├─ GATv2Conv(256 → 64, heads=4, edge_dim=2) + ReLU + LayerNorm
   └─ global_mean_pool + global_max_pool → concat → h_s [128]
```

### 5.2 VNR GCN Encoder
```
Input: (node_feats [N_v, 4], edge_feats [E_v, 1], edge_index)
   │
   ├─ GATv2Conv(4 → 32, heads=4, edge_dim=1) + ReLU + LayerNorm
   ├─ GATv2Conv(128 → 32, heads=4, edge_dim=1) + ReLU + LayerNorm
   └─ global_mean_pool + global_max_pool → concat → h_v [64]
```

### 5.3 Batch Context Encoder (Optional — Phase 2)
Encode the *entire remaining batch* to give each VNR awareness of its peers:
```
For each VNR in batch → h_v_i  (from VNR encoder, shared weights)
Stack → [B, 64]
   │
   └─ Transformer Encoder (1 layer, 4 heads) → mean pool → h_ctx [64]
```
This allows the model to learn: "VNR_i is a good first pick because it doesn't compete with VNR_j,k".

### 5.4 Scoring MLP
```
Input: concat(h_s [128], h_v [64], h_ctx [64])  →  [256]   (or [192] without ctx)
   │
   ├─ Linear(256 → 128) + ReLU + Dropout(0.1)
   ├─ Linear(128 → 64)  + ReLU
   └─ Linear(64 → 1)    → score (scalar, unbounded)
```

### 5.5 Full Model Summary
```python
class VNRScheduler(nn.Module):
    def __init__(self):
        self.substrate_encoder = SubstrateGCN()   # → h_s [128]
        self.vnr_encoder       = VNRGCN()          # → h_v [64]
        self.context_encoder   = BatchTransformer() # → h_ctx [64] (optional)
        self.scorer            = ScoringMLP()       # → score [1]

    def forward(self, substrate_data, vnr_data_list):
        h_s = self.substrate_encoder(substrate_data)           # [128]
        scores = []
        h_vs = [self.vnr_encoder(v) for v in vnr_data_list]    # list of [64]

        h_ctx = self.context_encoder(h_vs)  # [64] (optional)

        for h_v in h_vs:
            x = torch.cat([h_s, h_v, h_ctx], dim=-1)
            scores.append(self.scorer(x))

        return torch.stack(scores).squeeze()  # [B]
```

---

## 6. RL Environment Design

### 6.1 Environment: `VNEOrderingEnv`

Follows `gymnasium.Env` interface.

```
State  s_t  : (substrate_t, remaining_vnr_list)
Action a_t  : index i in [0, |remaining|) → pick VNR_i as next to embed
Transition  : run hpso_embed(substrate_t, VNR_i)
               → update substrate_t (resources consumed)
               → remove VNR_i from remaining list
Reward r_t  : see Section 6.2
Episode     : done when remaining list is empty
```

```python
class VNEOrderingEnv(gymnasium.Env):
    def __init__(self, substrate_fn, batch_fn, hpso_params):
        # substrate_fn(): callable that returns a fresh substrate graph
        # batch_fn():     callable that returns a fresh batch of VNRs
        ...

    def reset(self, seed=None):
        self.substrate = substrate_fn()
        self.vnr_list  = batch_fn()
        self.remaining = list(range(len(self.vnr_list)))
        self.accepted  = []
        self.rejected  = []
        return self._get_obs(), {}

    def step(self, action):
        vnr = self.vnr_list[self.remaining[action]]
        result = hpso_embed(self.substrate, vnr, **self.hpso_params)

        if result is not None:
            mapping, link_paths = result
            self.accepted.append((vnr, mapping, link_paths))
            self.last_success = True
        else:
            self.rejected.append(vnr)
            self.last_success = False

        self.remaining.pop(action)
        done = len(self.remaining) == 0

        reward = self._compute_reward(vnr, done)
        return self._get_obs(), reward, done, False, {}
```

### 6.2 Reward Design (Progressive)

#### Phase 1 — Simple (for architecture validation)
```python
def _compute_reward(self, vnr, done):
    if self.last_success:
        return +1.0
    else:
        return -0.5
```
Validates that the model can learn basic acceptance maximization.

#### Phase 2 — Revenue-aware (after Phase 1 works)
```python
def _compute_reward(self, vnr, done):
    if self.last_success:
        rev = revenue_of_vnr(vnr)
        cost = cost_of_vnr(vnr)   # proxy cost
        return rev / (cost + 1e-6)
    else:
        return -revenue_of_vnr(vnr) * 0.1   # penalize missed revenue
```

#### Phase 3 — Long-term batch reward (advanced)
```python
def _compute_reward(self, vnr, done):
    step_reward = 0.0

    if self.last_success:
        step_reward = revenue_of_vnr(vnr)

    if done:
        # Terminal bonus: how well did the batch do overall?
        total_rev  = sum(revenue_of_vnr(v) for v, _, _ in self.accepted)
        total_cost = sum(cost_of_vnr(v)    for v, _, _ in self.accepted)
        ar = len(self.accepted) / (len(self.accepted) + len(self.rejected) + 1e-9)
        rc = total_rev / (total_cost + 1e-6)

        terminal_bonus = ar * 5.0 + rc * 2.0
        step_reward += terminal_bonus

    return step_reward
```

**Design rationale for Phase 3:**
- `ar` term encourages the policy to not "waste" substrate on large VNRs that block many smaller ones
- `rc` term ensures resource efficiency, not just acceptance count
- Terminal bonus propagates credit assignment backward through the episode naturally with PPO's GAE

---

## 7. Policy Architecture (RL Agent)

Since the action space is **variable-size** (number of remaining VNRs changes each step), we cannot use a fixed-output policy head. Instead:

### Approach: Score-then-sample
```
obs = (substrate_data, [vnr_data for vnr in remaining])
   │
   └─ VNRScheduler.forward(substrate_data, vnr_data_list)
         → scores [|remaining|]
   │
   └─ Categorical(logits=scores) → sample action index
```

This maps naturally to REINFORCE and PPO with a custom policy.

### Custom SB3-compatible Policy
```python
class GNNActorCriticPolicy(BasePolicy):
    def __init__(self, ...):
        self.gnn = VNRScheduler(...)
        self.value_head = nn.Linear(128, 1)   # baseline for variance reduction

    def forward(self, obs):
        scores = self.gnn(obs['substrate'], obs['vnr_list'])
        dist   = Categorical(logits=scores)
        value  = self.value_head(obs['substrate_embedding'])
        return dist, value
```

---

## 8. Training Pipeline

### 8.1 Data Generation
```python
def make_substrate(n=50, seed=None):
    """
    Waxman random topology with CPU ~ Uniform(50,100), BW ~ Uniform(50,200)
    """
    ...

def make_vnr_batch(batch_size=10, seed=None):
    """
    Each VNR: 2–6 nodes, CPU ~ Uniform(5,30), BW ~ Uniform(5,50)
    """
    ...
```

Generate diverse substrates and batches for generalization. For initial tests, fix one substrate topology.

### 8.2 Training Loop (REINFORCE baseline, simplest)
```python
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for episode in range(num_episodes):
    obs, _ = env.reset()
    log_probs, rewards = [], []

    done = False
    while not done:
        scores = model(obs['substrate'], obs['vnr_list'])
        dist   = Categorical(logits=scores)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))
        obs, reward, done, _, _ = env.step(action.item())
        rewards.append(reward)

    # Compute discounted returns
    returns = compute_returns(rewards, gamma=0.99)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = -torch.stack(log_probs) @ returns
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
```

### 8.3 PPO Training (recommended after REINFORCE works)
```python
# Using SB3 with custom policy
from stable_baselines3 import PPO

model_ppo = PPO(
    policy=GNNActorCriticPolicy,
    env=VNEOrderingEnv(...),
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,          # encourages exploration
    verbose=1,
    tensorboard_log="./runs/vne_ppo"
)

model_ppo.learn(total_timesteps=500_000)
```

---

## 9. File & Module Structure

```
src/
├── algorithms/
│   ├── fast_hpso.py          (unchanged)
│   └── hpso_batch.py         (modified — see Section 10)
│
├── scheduler/
│   ├── __init__.py
│   ├── features.py           ← networkx graph → PyG Data conversion
│   ├── encoders.py           ← SubstrateGCN, VNRGCN, BatchTransformer
│   ├── model.py              ← VNRScheduler (full model)
│   ├── policy.py             ← GNNActorCriticPolicy (SB3-compatible)
│   ├── environment.py        ← VNEOrderingEnv (gymnasium.Env)
│   └── rewards.py            ← reward functions (pluggable)
│
├── training/
│   ├── train_reinforce.py    ← Phase 1 training script
│   ├── train_ppo.py          ← Phase 2 PPO training
│   └── generate_data.py      ← substrate / VNR generation utilities
│
└── evaluation/
    └── eval.py               (unchanged)
```

---

## 10. Integration with `hpso_batch.py`

### Modified `hpso_embed_batch`
```python
# hpso_batch.py

from src.scheduler.model import VNRScheduler
from src.scheduler.features import substrate_to_pyg, vnr_to_pyg

def hpso_embed_batch(
    substrate,
    batch,
    scheduler=None,          # ← NEW: optional GNN scheduler
    particles=20,
    ...
):
    accepted = []
    rejected = []

    if len(batch) > 0 and isinstance(batch[0], tuple):
        vnr_list = [vnr for vnr, _ in batch]
    else:
        vnr_list = batch

    # ─── Ordering ───────────────────────────────────────────────
    if scheduler is not None:
        # Learned ordering
        sub_data = substrate_to_pyg(substrate)
        vnr_datas = [vnr_to_pyg(v) for v in vnr_list]
        scores = scheduler.predict(sub_data, vnr_datas)   # [B] tensor
        order = scores.argsort(descending=True).tolist()
        vnr_list = [vnr_list[i] for i in order]
    else:
        # Fallback: revenue sort (original behavior)
        vnr_list.sort(key=lambda x: revenue_of_vnr(x), reverse=True)
    # ────────────────────────────────────────────────────────────

    for i, vnr in enumerate(vnr_list):
        result = hpso_embed(substrate_graph=substrate, vnr_graph=vnr, ...)
        if result is not None:
            mapping, link_paths = result
            accepted.append((vnr, mapping, link_paths))
        else:
            rejected.append(vnr)

    return accepted, rejected
```

### `VNRScheduler.predict` (inference helper)
```python
def predict(self, substrate_data, vnr_data_list):
    self.eval()
    with torch.no_grad():
        return self.forward(substrate_data, vnr_data_list)
```

---

## 11. `features.py` — Graph Conversion

```python
# src/scheduler/features.py
import torch
import networkx as nx
from torch_geometric.data import Data

def substrate_to_pyg(G: nx.Graph) -> Data:
    nodes = sorted(G.nodes())
    idx   = {n: i for i, n in enumerate(nodes)}

    # Node features
    x = torch.tensor([
        [G.nodes[n]['cpu'],
         G.nodes[n]['cpu'] / (G.nodes[n].get('cpu_max', G.nodes[n]['cpu']) + 1e-6),
         G.degree(n),
         sum(G.edges[n, nb].get('bw', 0) for nb in G.neighbors(n)) / (G.degree(n) + 1e-6),
         nx.clustering(G, n) if not G.is_directed() else 0.0]
        for n in nodes
    ], dtype=torch.float)

    # Edge index + features
    edges = list(G.edges(data=True))
    edge_index = torch.tensor(
        [[idx[u], idx[v]] for u, v, _ in edges] +
        [[idx[v], idx[u]] for u, v, _ in edges],
        dtype=torch.long
    ).t().contiguous()

    edge_attr = torch.tensor(
        [[d.get('bw', 0), d.get('bw', 0) / (d.get('bw_max', d.get('bw', 1)) + 1e-6)]
         for _, _, d in edges] * 2,
        dtype=torch.float
    )

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def vnr_to_pyg(G: nx.Graph) -> Data:
    nodes = sorted(G.nodes())
    idx   = {n: i for i, n in enumerate(nodes)}

    total_cpu = sum(G.nodes[n]['cpu'] for n in nodes) + 1e-6

    x = torch.tensor([
        [G.nodes[n]['cpu'],
         G.degree(n),
         sum(G.edges[n, nb].get('bw', 0) for nb in G.neighbors(n)),
         G.nodes[n]['cpu'] / total_cpu]
        for n in nodes
    ], dtype=torch.float)

    edges = list(G.edges(data=True))
    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(
            [[idx[u], idx[v]] for u, v, _ in edges] +
            [[idx[v], idx[u]] for u, v, _ in edges],
            dtype=torch.long
        ).t().contiguous()
        edge_attr = torch.tensor(
            [[d.get('bw', 0)] for _, _, d in edges] * 2,
            dtype=torch.float
        )

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
```

---

## 12. `encoders.py` — GNN Modules

```python
# src/scheduler/encoders.py
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

class SubstrateGCN(nn.Module):
    def __init__(self, in_dim=5, hidden=64, heads=4, edge_dim=2, out_dim=128):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim,  hidden, heads=heads, edge_dim=edge_dim, concat=True)
        self.conv2 = GATv2Conv(hidden*heads, hidden, heads=heads, edge_dim=edge_dim, concat=False)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.norm2 = nn.LayerNorm(hidden)
        self.act   = nn.ReLU()
        self.proj  = nn.Linear(hidden * 2, out_dim)  # mean+max concat

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.act(self.norm1(self.conv1(x, ei, ea)))
        x = self.act(self.norm2(self.conv2(x, ei, ea)))
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        return self.proj(g)   # [num_graphs, 128]


class VNRGCN(nn.Module):
    def __init__(self, in_dim=4, hidden=32, heads=4, edge_dim=1, out_dim=64):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim,  hidden, heads=heads, edge_dim=edge_dim, concat=True)
        self.conv2 = GATv2Conv(hidden*heads, hidden, heads=heads, edge_dim=edge_dim, concat=False)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.norm2 = nn.LayerNorm(hidden)
        self.act   = nn.ReLU()
        self.proj  = nn.Linear(hidden * 2, out_dim)

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # Handle empty edge case (VNRs with no edges)
        if ei.size(1) == 0:
            g = torch.cat([x.mean(0, keepdim=True), x.max(0).values.unsqueeze(0)], dim=-1)
            return self.proj(g)
        x = self.act(self.norm1(self.conv1(x, ei, ea)))
        x = self.act(self.norm2(self.conv2(x, ei, ea)))
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        return self.proj(g)   # [num_graphs, 64]


class BatchContextEncoder(nn.Module):
    """
    Encodes the context of ALL remaining VNRs in a batch.
    Uses a small Transformer to let each VNR attend to its peers.
    """
    def __init__(self, vnr_dim=64, out_dim=64, nhead=4, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vnr_dim, nhead=nhead, dim_feedforward=128,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(vnr_dim, out_dim)

    def forward(self, vnr_embeddings):
        # vnr_embeddings: [B, 64]  (stacked embeddings of remaining VNRs)
        x = vnr_embeddings.unsqueeze(0)              # [1, B, 64]
        x = self.transformer(x).squeeze(0)           # [B, 64]
        ctx = x.mean(0, keepdim=True)                # [1, 64]  global context
        return self.proj(ctx).expand(x.size(0), -1)  # [B, 64]  broadcast to each VNR
```

---

## 13. `model.py` — Full VNRScheduler

```python
# src/scheduler/model.py
import torch
import torch.nn as nn
from src.scheduler.encoders import SubstrateGCN, VNRGCN, BatchContextEncoder

class ScoringMLP(nn.Module):
    def __init__(self, in_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),    nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)   # [B, 1]


class VNRScheduler(nn.Module):
    def __init__(self, use_batch_context=True):
        super().__init__()
        self.use_batch_context = use_batch_context

        self.substrate_encoder = SubstrateGCN()              # → 128
        self.vnr_encoder       = VNRGCN()                    # → 64
        self.context_encoder   = BatchContextEncoder() if use_batch_context else None  # → 64

        in_dim = 128 + 64 + (64 if use_batch_context else 0)
        self.scorer = ScoringMLP(in_dim=in_dim)

    def forward(self, substrate_data, vnr_data_list):
        """
        substrate_data  : PyG Data (single substrate graph)
        vnr_data_list   : list of PyG Data (one per remaining VNR)

        Returns: scores [B]  (higher = higher priority)
        """
        from torch_geometric.data import Batch

        h_s = self.substrate_encoder(substrate_data)       # [1, 128]

        vnr_batch = Batch.from_data_list(vnr_data_list)
        # Encode each VNR individually using batch indices
        h_vs = self.vnr_encoder(vnr_batch)                 # [B, 64]

        if self.use_batch_context and self.context_encoder is not None:
            h_ctx = self.context_encoder(h_vs)             # [B, 64]
        else:
            h_ctx = None

        h_s_exp = h_s.expand(h_vs.size(0), -1)            # [B, 128]

        if h_ctx is not None:
            combined = torch.cat([h_s_exp, h_vs, h_ctx], dim=-1)  # [B, 256]
        else:
            combined = torch.cat([h_s_exp, h_vs], dim=-1)          # [B, 192]

        scores = self.scorer(combined).squeeze(-1)         # [B]
        return scores

    def predict(self, substrate_data, vnr_data_list):
        self.eval()
        with torch.no_grad():
            return self.forward(substrate_data, vnr_data_list)
```

---

## 14. `environment.py` — Gymnasium Environment

```python
# src/scheduler/environment.py
import gymnasium
import numpy as np
from src.algorithms.fast_hpso import hpso_embed
from src.evaluation.eval import revenue_of_vnr, cost_of_vnr
from src.scheduler.features import substrate_to_pyg, vnr_to_pyg
from src.utils.graph_utils import copy_substrate

class VNEOrderingEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self, substrate_fn, batch_fn, hpso_params=None, reward_mode="simple"):
        self.substrate_fn  = substrate_fn
        self.batch_fn      = batch_fn
        self.hpso_params   = hpso_params or {}
        self.reward_mode   = reward_mode  # "simple" | "revenue" | "longterm"

        # These are set in reset()
        self.substrate = None
        self.vnr_list  = None
        self.remaining = None
        self.accepted  = None
        self.rejected  = None
        self.last_result = None

        # Observation/Action spaces are symbolic — actual tensors returned as dicts
        # SB3 requires gym spaces; use a placeholder Dict space
        # For custom training loop, spaces can be left abstract
        self.observation_space = gymnasium.spaces.Dict({})
        self.action_space = gymnasium.spaces.Discrete(1)   # overridden each step

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.substrate = self.substrate_fn()
        self.vnr_list  = self.batch_fn()
        self.remaining = list(range(len(self.vnr_list)))
        self.accepted  = []
        self.rejected  = []
        self.last_result = None
        return self._get_obs(), {}

    def step(self, action):
        vnr_idx = self.remaining[action]
        vnr = self.vnr_list[vnr_idx]

        result = hpso_embed(
            substrate_graph=self.substrate,
            vnr_graph=vnr,
            **self.hpso_params
        )

        if result is not None:
            mapping, link_paths = result
            self.accepted.append((vnr, mapping, link_paths))
            self.last_result = ("accept", vnr)
        else:
            self.rejected.append(vnr)
            self.last_result = ("reject", vnr)

        self.remaining.pop(action)
        done = (len(self.remaining) == 0)

        reward = self._compute_reward(vnr, done)
        obs = self._get_obs()
        return obs, reward, done, False, {"accepted": len(self.accepted), "rejected": len(self.rejected)}

    def _get_obs(self):
        return {
            "substrate":  substrate_to_pyg(self.substrate),
            "vnr_list":   [vnr_to_pyg(self.vnr_list[i]) for i in self.remaining],
            "n_remaining": len(self.remaining),
        }

    def _compute_reward(self, vnr, done):
        status, _ = self.last_result

        if self.reward_mode == "simple":
            return 1.0 if status == "accept" else -0.5

        elif self.reward_mode == "revenue":
            if status == "accept":
                rev  = revenue_of_vnr(vnr)
                cost = cost_of_vnr(vnr)
                return rev / (cost + 1e-6)
            else:
                return -revenue_of_vnr(vnr) * 0.1

        elif self.reward_mode == "longterm":
            step_r = revenue_of_vnr(vnr) if status == "accept" else 0.0
            if done:
                total_rev  = sum(revenue_of_vnr(v) for v, _, _ in self.accepted)
                total_cost = sum(cost_of_vnr(v)    for v, _, _ in self.accepted)
                n_total    = len(self.accepted) + len(self.rejected)
                ar = len(self.accepted) / (n_total + 1e-9)
                rc = total_rev / (total_cost + 1e-6)
                step_r += ar * 5.0 + rc * 2.0
            return step_r

        return 0.0
```

---

## 15. Training Scripts

### Phase 1: REINFORCE (`train_reinforce.py`)
```python
# src/training/train_reinforce.py
import torch
from torch.distributions import Categorical
from src.scheduler.model import VNRScheduler
from src.scheduler.environment import VNEOrderingEnv

def compute_returns(rewards, gamma=0.99):
    R, returns = 0.0, []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

def train(num_episodes=2000, lr=3e-4):
    model = VNRScheduler(use_batch_context=False)  # start simple
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    env = VNEOrderingEnv(substrate_fn=..., batch_fn=..., reward_mode="simple")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        log_probs, rewards = [], []

        done = False
        while not done:
            if not obs["vnr_list"]:
                break

            scores = model(obs["substrate"], obs["vnr_list"])
            dist   = Categorical(logits=scores)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            obs, reward, done, _, _ = env.step(action.item())
            rewards.append(reward)

        if not log_probs:
            continue

        returns = compute_returns(rewards)
        baseline = returns.mean()
        advantages = returns - baseline

        loss = -(torch.stack(log_probs) * advantages).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if ep % 100 == 0:
            print(f"Episode {ep} | Mean reward: {sum(rewards):.3f}")

    torch.save(model.state_dict(), "scheduler_reinforce.pt")
```

---

## 16. Implementation Phases

### Phase 0 — Setup (1–2 days)
- [ ] Install: `pip install torch torch_geometric stable-baselines3[extra] gymnasium`
- [ ] Implement `features.py` (graph → PyG conversion)
- [ ] Write unit tests: substrate round-trip, VNR round-trip
- [ ] Verify `substrate_to_pyg` and `vnr_to_pyg` output correct shapes

### Phase 1 — Baseline GNN + REINFORCE (3–5 days)
- [ ] Implement `encoders.py` (SubstrateGCN + VNRGCN, no context encoder)
- [ ] Implement `model.py` (VNRScheduler without batch context)
- [ ] Implement `environment.py` with `reward_mode="simple"`
- [ ] Train with `train_reinforce.py`
- [ ] Evaluate: compare AR vs revenue-sort baseline on held-out batches
- [ ] **Gate:** AR improvement > 0 on simple cases → proceed

### Phase 2 — PPO + Revenue Reward (3–5 days)
- [ ] Switch to `reward_mode="revenue"` in environment
- [ ] Implement `GNNActorCriticPolicy` for SB3
- [ ] Train with PPO (`train_ppo.py`)
- [ ] Compare: REINFORCE vs PPO on medium substrate (50 nodes, batch=10 VNRs)

### Phase 3 — Batch Context + Long-term Reward (5–7 days)
- [ ] Add `BatchContextEncoder` to `encoders.py`
- [ ] Enable `use_batch_context=True` in `VNRScheduler`
- [ ] Switch to `reward_mode="longterm"`
- [ ] Retrain, evaluate long-term R/C ratio improvement

### Phase 4 — Integration & Evaluation (2–3 days)
- [ ] Integrate into `hpso_batch.py` (see Section 10)
- [ ] End-to-end benchmark: scheduler OFF vs ON across 100 random batch episodes
- [ ] Report: AR, R/C, episode reward, training curves

---

## 17. Evaluation Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Acceptance Rate (AR) | accepted / total | maximize |
| Revenue/Cost (R/C) | Σrevenue / Σcost | maximize |
| Long-term AR | AR averaged over many episodes | maximize |
| vs Baseline Δ | (GNN metric - revenue_sort metric) / revenue_sort metric | > 0% |

---

## 18. Key Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| GATv2Conv over GCN/GraphSAGE | GATv2 attends to edge features (BW), more expressive for resource-aware graphs |
| Variable-action-space via scoring | Natural fit for VNE: batch size changes, scoring + Categorical handles it cleanly |
| `use_batch_context=False` in Phase 1 | Faster iteration; Transformer context adds complexity, validate base first |
| `reward_mode="simple"` first | Isolates learning signal from reward shaping complexity |
| Terminal batch bonus in Phase 3 | Directly optimizes long-term AR + R/C, not just myopic per-step success |
| `copy_substrate` in env reset | Ensures each episode starts from a clean state |
| `LayerNorm` over `BatchNorm` | Graphs have variable sizes; LayerNorm is batch-size independent |
| Gradient clipping (0.5) | REINFORCE has high variance; clipping prevents divergence |
