# GCN-RL VNR Ordering Scheduler — Architecture & Training/Inference Guide

## Table of Contents

1. [Problem Overview](#1-problem-overview)
2. [System Architecture](#2-system-architecture)
3. [Feature Engineering](#3-feature-engineering)
4. [Model Architecture](#4-model-architecture)
5. [Training Pipeline — How It Works](#5-training-pipeline)
6. [Inference Pipeline — How It Works](#6-inference-pipeline)
7. [Reward Modes (Plug-in Design)](#7-reward-modes)
8. [Integration with HPSO](#8-integration-with-hpso)
9. [Module Map](#9-module-map)

---

## 1. Problem Overview

### The Core Problem

The HPSO batch embedder is **stateful**: resources consumed when embedding VNR `i` reduce substrate capacity for all subsequent VNRs. The naive revenue-first ordering is *myopic*—it maximises short-term gain but can fragment the substrate and block high-compatibility groups.

```
Revenue-sort picks VNR1 (large) → fragments clusters A and B
→ VNR2, VNR3, VNR4 (small, tightly-packable) all FAIL

GNN-sort picks small VNRs first → compact packing
→ all three accepted → higher AR and better R/C
```

### Goal

Learn a **scoring function** `Score(VNR_i | Substrate_t, Batch)` such that sorting by descending score maximises:

| Metric | Formula |
|--------|---------|
| Acceptance Rate (AR) | `n_accepted / n_total` |
| Revenue-to-Cost (R/C) | `Σ revenue / Σ cost` |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     VNR Ordering Scheduler                      │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────┐  │
│  │ Substrate    │   │  VNR Batch   │   │  Batch Context    │  │
│  │ GCN Encoder  │   │  GCN Encoder │   │  (Transformer)    │  │
│  │  → h_s[128]  │   │  → h_v[64]   │   │  → h_ctx[64]      │  │
│  └──────┬───────┘   └──────┬───────┘   └────────┬──────────┘  │
│         └──────────────────┴────────────────────┘              │
│                            │                                    │
│                  concat([h_s, h_v, h_ctx])                     │
│                            │                                    │
│                     ┌──────▼──────┐                            │
│                     │  Scoring    │                            │
│                     │    MLP      │                            │
│                     └──────┬──────┘                            │
│                        score per VNR [B]                       │
└────────────────────────────┼───────────────────────────────────┘
                             │
                    argsort(descending)
                             │
                    ┌────────▼────────┐
                    │  hpso_embed×B   │
                    │  (unchanged)    │
                    └─────────────────┘
```

---

## 3. Feature Engineering

Feature extraction lives in `src/scheduler/features.py`. Two conversion functions transform NetworkX graphs into PyTorch Geometric `Data` objects.

### 3.1 Substrate Node Features  `[N_s, 5]`

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `cpu_available` | Current available CPU (dynamic, decreases as VNRs embed) |
| 1 | `cpu_ratio` | `cpu_available / cpu_total` — utilisation signal |
| 2 | `degree` | Number of substrate edges on this node |
| 3 | `avg_bw_neighbors` | Mean BW of incident edges |
| 4 | `clustering_coeff` | Local cluster density (NetworkX `clustering`) |

### 3.2 Substrate Edge Features  `[E_s, 2]`

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `bw_available` | Current available BW (dynamic) |
| 1 | `bw_ratio` | `bw_available / bw_total` |

### 3.3 VNR Node Features  `[N_v, 4]`

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `cpu_demand` | CPU requested by this virtual node |
| 1 | `degree` | Edges in the VNR touching this node |
| 2 | `sum_bw_incident` | Sum of BW demands on all incident edges |
| 3 | `cpu_demand / total_cpu` | Relative demand within the VNR |

### 3.4 VNR Edge Features  `[E_v, 1]`

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `bw_demand` | Bandwidth requested by this virtual link |

> **Design note:** Substrate features are *dynamic* — they reflect current available resources, not original capacities. This means the model can see exactly how congested the substrate is at each step.

---

## 4. Model Architecture

### 4.1 SubstrateGCN  →  `h_s ∈ ℝ^128`

```
Input: x[N_s, 5], edge_index[2, 2E_s], edge_attr[2E_s, 2]
  │
  ├─ GATv2Conv(5 → 64, heads=4, edge_dim=2)  → [N_s, 256]
  │  LayerNorm(256) + ReLU
  │
  ├─ GATv2Conv(256 → 64, heads=4, edge_dim=2) → [N_s, 64]
  │  LayerNorm(64) + ReLU
  │
  ├─ global_mean_pool → [1, 64]
  ├─ global_max_pool  → [1, 64]
  │  cat             → [1, 128]
  │
  └─ Linear(128 → 128)  →  h_s[1, 128]
```

**Why GATv2Conv?** Standard GCN cannot use edge features. GATv2 computes attention weights conditioned on **both node and edge features** — critical for BW-aware graph reasoning.

### 4.2 VNRGCN  →  `h_v ∈ ℝ^64`

```
Input: x[N_v, 4], edge_index[2, 2E_v], edge_attr[2E_v, 1]
  │
  ├─ GATv2Conv(4 → 32, heads=4, edge_dim=1)   → [N_v, 128]
  │  LayerNorm(128) + ReLU
  │
  ├─ GATv2Conv(128 → 32, heads=4, edge_dim=1) → [N_v, 32]
  │  LayerNorm(32) + ReLU
  │
  ├─ global_mean_pool → [1, 32]
  ├─ global_max_pool  → [1, 32]
  │  cat + Linear    →  h_v[1, 64]
```

**Edge-case handling:** VNRs with *no edges* skip GATv2Conv and fall back to direct mean/max pooling over node features.

### 4.3 BatchContextEncoder  →  `h_ctx ∈ ℝ^{B×64}`  *(Phase 2+)*

```
Input: h_vs [B, 64]  (stacked embeddings of all remaining VNRs)
  │
  ├─ TransformerEncoder (1 layer, 4 heads, FFN=128)
  ├─ mean(dim=0)         → global context [1, 64]
  ├─ expand(B, -1)       → [B, 64]
  └─ Linear(64 → 64)    →  h_ctx[B, 64]
```

Allows the model to learn: *"VNR_i is a good pick first because it doesn't compete with VNR_j and VNR_k."*

### 4.4 ScoringMLP  →  `scores ∈ ℝ^B`

```
concat([h_s_expanded, h_vs, h_ctx])  →  [B, 256]
  Linear(256→128) + ReLU + Dropout(0.1)
  Linear(128→64)  + ReLU
  Linear(64→1)   → scores[B]
```

### 4.5 Full Forward Pass (pseudocode)

```python
h_s   = substrate_encoder(substrate_pyg)     # [1, 128]
h_vs  = vnr_encoder(Batch(vnr_pyg_list))     # [B, 64]
h_ctx = context_encoder(h_vs)               # [B, 64]  if enabled
h_s_e = h_s.expand(B, -1)                   # [B, 128]
comb  = cat([h_s_e, h_vs, h_ctx], dim=-1)   # [B, 256]
scores = scorer(comb).squeeze(-1)           # [B]
```

---

## 5. Training Pipeline

### 5.1 MDP Formulation

| MDP Element | Definition |
|------------|-----------|
| **State** `s_t` | `(substrate_t, remaining_vnr_list)` |
| **Action** `a_t` | Index `i ∈ [0, |remaining|)` — pick VNR_i next |
| **Transition** | `hpso_embed(substrate_t, VNR_i)` → updates substrate, removes VNR_i |
| **Reward** `r_t` | Pluggable (see §7) |
| **Terminal** | All VNRs processed |

### 5.2 Phase 1 — REINFORCE  (`src/training/train_reinforce.py`)

```
For each episode:
  obs ← env.reset()
  while not done:
    scores = model(substrate_pyg, vnr_pyg_list)
    dist   = Categorical(logits=scores)
    action = dist.sample()                ← stochastic
    log_π  = dist.log_prob(action)
    obs, reward, done = env.step(action)

  G_t = Σ_{t'≥t} γ^{t'-t} · r_{t'}     ← discounted returns
  A_t = G_t - mean(G)                   ← baseline (variance reduction)
  L   = -(log_π · A_t).mean()           ← REINFORCE loss
  L.backward(); clip_grad(0.5); step()
```

Key design choices:
- **Baseline subtraction** (mean return) reduces variance without introducing bias
- **Gradient clipping (0.5)** prevents parameter explosion from high-variance REINFORCE
- **Stochastic sampling** enables exploration during training

### 5.3 Phase 2 — Custom PPO  (`src/training/train_ppo.py`)

```
Collect N timesteps (rollout phase):
  action, log_π_old, value = actor_critic(obs)
  obs, reward, done = env.step(action)

Compute GAE advantages:
  δ_t    = r_t + γ·V(s_{t+1}) - V(s_t)
  A_t_GAE = Σ_{l≥0} (γλ)^l · δ_{t+l}
  G_t    = A_t_GAE + V(s_t)               ← value targets

Mini-batch gradient update (K epochs):
  ratio   = exp(log_π_new - log_π_old)
  L_actor = -min(ratio·A, clip(ratio,1±ε)·A).mean()
  L_value = MSE(V_new, G).mean()
  L_ent   = -entropy.mean()
  L_total = L_actor + c_v·L_value + c_ent·L_ent
  L_total.backward(); clip_grad; step()
```

PPO default hyper-parameters:

| Param | Value | Why |
|-------|-------|-----|
| `n_steps` | 512 | Rollout buffer size |
| `n_epochs` | 10 | Gradient reuse per rollout |
| `clip_range` | 0.2 | Prevents large destructive updates |
| `gae_lambda` | 0.95 | Bias-variance balance for advantage estimation |
| `ent_coef` | 0.01 | Encourages exploration |
| `gamma` | 0.99 | Long horizon (full batch counts) |

### 5.4 Variable Action Space

```
Standard SB3 PPO:  fixed output dim → cannot handle variable |remaining|
Our solution:      scores[|remaining|] → Categorical(logits) → sample index
```

The Categorical distribution naturally handles any size. Action is always an *index into the remaining list*, not absolute VNR id.

### 5.5 Curriculum via Data Factories

Training difficulty controlled in `src/training/generate_data.py`:

| Stage | Config | Purpose |
|-------|--------|---------|
| Warm-up | `fixed_substrate=True, batch=5, reward=simple` | Validate GNN can learn accept/reject |
| Phase 1 | `fixed_substrate=False, batch=10, reward=simple` | Generalisation |
| Phase 2 | `batch=10, reward=revenue` | Resource-aware ordering |
| Phase 3 | `batch=15, reward=longterm` | Long-term batch optimisation |

---

## 6. Inference Pipeline

### 6.1 Single-Batch Greedy Inference

```
substrate (NetworkX)         vnr_list (NetworkX[])
       │                            │
  substrate_to_pyg()          vnr_to_pyg() × B
       │                            │
       └────────────────────────────┘
             VNRScheduler.predict()        ← no grad, eval mode
                     │
               scores [B]
                     │
          argsort(descending=True)
                     │
             order [B]  (priority index list)
                     │
    hpso_embed_batch_scheduled(substrate, batch, scheduler=model)
         └── calls hpso_embed() in the learned order (unchanged)
```

### 6.2 Plugin Design of `hpso_batch_scheduler.py`

```python
def hpso_embed_batch_scheduled(substrate, batch, scheduler=None, ...):
    if scheduler is not None:
        order = _scheduler_order(...)    # GNN scoring → argsort
    else:
        order = _revenue_sort_order(...) # original behaviour

    for idx in order:
        hpso_embed(substrate, vnr_list[idx], ...)  # unchanged HPSO
```

**Key invariant:** `hpso_embed` is called identically — the scheduler only changes the *sequence*, never the HPSO logic or substrate mutation.

### 6.3 Checkpoint Loading

```python
scheduler = VNRScheduler.load("checkpoints/ppo_phase2_final.pt",
                               device="cuda",
                               use_batch_context=True)
# Automatically in eval() mode
```

---

## 7. Reward Modes (Plug-in Design)

All reward functions are registered in `src/scheduler/rewards.py`.

### `simple`  (Phase 1)
```
r = +1.0  if accepted
r = -0.5  if rejected
```

### `revenue`  (Phase 2)
```
r = revenue(VNR) / cost(VNR)   if accepted
r = -revenue(VNR) × 0.1        if rejected
```

### `longterm`  (Phase 3)
```
r = revenue(VNR)                if accepted, else 0
if done:
    r += AR × 5.0 + RC × 2.0   ← terminal batch bonus
```

The terminal bonus propagates back through GAE, giving credit to early decisions that enabled late VNRs to succeed.

### Adding a Custom Reward

```python
# 1. Add enum value in rewards.py
class RewardMode(str, enum.Enum):
    CUSTOM = "custom"

# 2. Implement function
def _reward_custom(success, vnr, done, accepted, rejected) -> float:
    return ...

# 3. Register
_REWARD_FNS[RewardMode.CUSTOM] = _reward_custom
```

No changes to environment, model, or training loop needed.

---

## 8. Integration with HPSO

### Unchanged files

| File | Description |
|------|-------------|
| `src/algorithms/fast_hpso.py` | HPSO embedding logic |
| `src/algorithms/hpso_batch.py` | Original revenue-sort batch wrapper |
| `src/evaluation/eval.py` | Revenue/cost functions |
| `src/utils/graph_utils.py` | `copy_substrate`, `reserve_node`, etc. |
| All of `src/generators/` | Substrate and VNR generators |

### New files

| File | Purpose |
|------|---------|
| `src/scheduler/` | GCN model, environment, rewards, features |
| `src/training/` | Training scripts, data factories, evaluation |
| `src/algorithms/hpso_batch_scheduler.py` | Batch wrapper with scheduler plugin |

### Dependency graph

```
generate_substrate()    generate_single_vnr()
       └─────────────────────┘
       make_substrate_fn / make_batch_fn
              └─────────────────────────────┐
                   VNEOrderingEnv.reset()   │
                          │                 │
                   VNEOrderingEnv.step()    │
                          │                 │
                   hpso_embed()  ←──────────┘ (unchanged)
                          │
                   copy_substrate()  (unchanged)
```

---

## 9. Module Map

```
src/
├── scheduler/                    NEW: GCN-RL model
│   ├── __init__.py
│   ├── features.py               networkx → PyG Data
│   ├── encoders.py               SubstrateGCN, VNRGCN, BatchContextEncoder
│   ├── model.py                  VNRScheduler (full model + save/load)
│   ├── policy.py                 GNNActorCritic (actor-critic for PPO)
│   ├── environment.py            VNEOrderingEnv (gymnasium.Env)
│   └── rewards.py                pluggable reward functions
│
├── training/                     NEW: training infrastructure
│   ├── __init__.py
│   ├── generate_data.py          substrate_fn/batch_fn factories
│   ├── train_reinforce.py        Phase 1 REINFORCE loop
│   ├── train_ppo.py              Phase 2 custom PPO loop
│   └── evaluate.py               inference + evaluation report
│
├── algorithms/
│   ├── hpso_batch.py             ORIGINAL (unchanged)
│   ├── hpso_batch_scheduler.py   NEW: drop-in extension with scheduler plugin
│   └── fast_hpso.py              ORIGINAL (unchanged)
│
├── generators/                   ORIGINAL (unchanged, reused)
├── utils/                        ORIGINAL (unchanged, reused)
└── evaluation/                   ORIGINAL (unchanged, reused)
```
