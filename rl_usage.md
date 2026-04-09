# RL Scheduler — Usage Guide

Complete guide for training, configuring, loading, and deploying the
RL-based VNR Scheduler (`Graph Pointer Network + PPO`).

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Training Modes](#2-training-modes)
3. [All CLI Parameters](#3-all-cli-parameters)
4. [PSD Curriculum Deep-Dive](#4-psd-curriculum-deep-dive)
5. [Resume & Checkpointing](#5-resume--checkpointing)
6. [Loading a Trained Model for Inference](#6-loading-a-trained-model-for-inference)
7. [Using the Agent inside hpso_batch.py](#7-using-the-agent-inside-hpso_batchpy)
8. [Reading Training Logs](#8-reading-training-logs)
9. [Architecture Configuration](#9-architecture-configuration)
10. [Using Your Own Dataset](#10-using-your-own-dataset)
11. [Common Issues & Fixes](#11-common-issues--fixes)

---

## 1. Quick Start

```bash
# Activate your environment (must have torch + dgl installed)
conda activate rl-vne

# Navigate to the project root
cd d:\HUST_file\@research\@working

# Minimal test run (standard mode, synthetic data, 50 episodes)
python -m src.scripts.train_rl_scheduler --episodes 50

# Output: result/rl_scheduler.pt
```

Expected output after ~60s:
```
[WARNING] PyTorch has CUDA, but DGL is CPU-only. Falling back to CPU.
[Data] Substrate: 30 nodes, 141 edges  |  VNR pool: 200 VNRs
[HPSO] Using fast_hpso.hpso_embed
[INFO] Agent parameters: 1,108,801
======================================================================
  RL VNR Scheduler Training  [Standard]
  episodes=50  window=10  K_max=20  lr=0.0003
----------------------------------------------------------------------
Ep   10/50 | Acc=1.000 | RC=0.843 | R=0.91±0.28 | Loss=2.31 | 23s
Ep   20/50 | Acc=1.000 | RC=0.851 | R=0.93±0.25 | Loss=1.89 | 45s
...
[INFO] Model saved → result/rl_scheduler.pt
```

---

## 2. Training Modes

### Mode A — Standard (default)

Each episode uses a **fresh deepcopy** of the substrate. Good for debugging.
Problem: if substrate has ample capacity, AccRate = 1.0 always → no RL signal.

```bash
python -m src.scripts.train_rl_scheduler \
    --episodes 300 \
    --substrate_nodes 15 \
    --vnr_min_nodes 4 --vnr_max_nodes 10 \
    --save_path result/standard.pt
```

> Use `--substrate_nodes 15` with 4–10 node VNRs to create a tighter problem
> where some VNRs fail and ordering matters.

---

### Mode B — PSD Curriculum (`--psd`)

**Recommended for real training.** The substrate fills monotonically as the
agent masters each load level. Difficulty self-escalates without needing
hand-crafted hard data.

```bash
python -m src.scripts.train_rl_scheduler \
    --psd \
    --patience 10 \
    --ar_thresh 0.95 \
    --rc_thresh 0.90 \
    --max_load  0.85 \
    --episodes 2000 \
    --substrate_nodes 30 \
    --save_path result/psd_trained.pt
```

What to expect:
```
[PSD] Curriculum active. Initial substrate load: 0.0%

Ep   10/2000 | Acc=1.000 | RC=0.910 | R=0.93±0.12 | Loss=2.41 | Load=0.0% | 24s
...
[PSD] Commit #1 at ep=10 | Best ep=7 (RC=0.923, 10 VNRs) | Load: 0.0% → 15.3% | 24s

Ep   20/2000 | Acc=0.980 | RC=0.873 | R=0.85±0.31 | Load=15.3% | ...
...
[PSD] Commit #2 at ep=20 | Best ep=18 (RC=0.891, 9 VNRs) | Load: 15.3% → 28.7%
...
[PSD] Substrate saturated (86.1% >= 85%). Stopping at episode 487.
[PSD] Commits: 24 | VNRs committed: 231 | Final load: 86.1%
```

---

## 3. All CLI Parameters

### Data

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `None` | Path to `(substrate, [vnr])` pickle. If omitted, generates synthetic data. |
| `--substrate_nodes` | `30` | Number of physical nodes in synthetic substrate |
| `--vnr_pool_size` | `200` | Size of the VNR reservoir (randomly sampled each episode) |
| `--vnr_min_nodes` | `3` | Minimum virtual nodes per VNR |
| `--vnr_max_nodes` | `8` | Maximum virtual nodes per VNR |

### Training

| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | `300` | Number of PPO update steps |
| `--window_size` | `10` | VNRs per episode (must be `<= K_max`) |
| `--K_max` | `20` | Max queue length (padding target for the pointer network) |
| `--lr` | `3e-4` | Adam learning rate |
| `--gamma` | `0.98` | PPO discount factor |
| `--gae_lambda` | `0.95` | GAE λ (higher = less bias, more variance) |
| `--clip_eps` | `0.2` | PPO clip epsilon |
| `--entropy_coef` | `0.01` | Entropy bonus weight (exploration) |
| `--value_coef` | `0.5` | Critic loss weight |
| `--k_epochs` | `4` | PPO update epochs per episode |

### PSD Curriculum

| Argument | Default | Description |
|----------|---------|-------------|
| `--psd` | `False` | Enable PSD curriculum (flag, no value needed) |
| `--patience` | `10` | Consecutive mastered episodes before committing |
| `--ar_thresh` | `0.95` | AccRate ≥ threshold to count as "mastered" |
| `--rc_thresh` | `0.90` | RC ratio ≥ threshold to count as "mastered" |
| `--max_load` | `0.85` | Substrate CPU fraction at which training stops |

### Checkpointing

| Argument | Default | Description |
|----------|---------|-------------|
| `--save_path` | `result/rl_scheduler.pt` | Final model output path |
| `--resume` | `None` | Path to a `.pt` checkpoint to continue from |
| `--save_every` | `100` | Save a checkpoint every N episodes |

### Evaluation

| Argument | Default | Description |
|----------|---------|-------------|
| `--eval_every` | `50` | Run evaluation every N episodes |
| `--eval_episodes` | `5` | Episodes to average in each eval run |

### Network Architecture

| Argument | Default | Description |
|----------|---------|-------------|
| `--substrate_hidden` | `128` | Substrate GAT encoder output dimension |
| `--vnr_hidden` | `64` | VNR GAT encoder output dimension |
| `--gru_hidden` | `256` | Pointer decoder (GRU) + context MLP hidden size |
| `--critic_hidden` | `128` | Critic head hidden dimension |

### HPSO Solver

| Argument | Default | Description |
|----------|---------|-------------|
| `--hpso_particles` | `20` | Particles in PSO swarm |
| `--hpso_iterations` | `30` | PSO iterations per VNR |

---

## 4. PSD Curriculum Deep-Dive

### How a commit is chosen

Within each patience window, the episode with the **highest RC ratio** is
selected for commit (not the most recent one). This ensures the physical
substrate gets the most resource-efficient set of paths recorded during that period.

```
Window (patience=10):
  ep 91: RC=0.85   ep 92: RC=0.91   ep 93: RC=0.88
  ep 94: RC=0.95*  ep 95: RC=0.87   ep 96: RC=0.90
  ep 97: RC=0.93   ep 98: RC=0.86   ep 99: RC=0.94
  ep 100: RC=0.89
  ↑ ep 94 (RC=0.95) is committed — best in window
```

### Tuning patience

| `patience` | Effect |
|------------|--------|
| 5 | Faster commits, fills substrate quickly, more aggressive curriculum |
| 10 (default) | Balanced |
| 20 | Slower, agent trains longer at each load level |

### Tuning thresholds

```bash
# Easier trigger (commit more often, faster substrate fill)
--ar_thresh 0.85 --rc_thresh 0.80

# Harder trigger (agent must be near-perfect before committing)
--ar_thresh 0.98 --rc_thresh 0.95

# Stop at lower saturation (more headroom for future VNRs)
--max_load 0.70
```

### Stopping conditions

PSD training stops when **either** condition is first met:
1. `episode >= --episodes`
2. `substrate_load >= --max_load`

---

## 5. Resume & Checkpointing

### Auto-saved checkpoints

Every `--save_every` episodes (default 100), a timestamped checkpoint is written:
```
result/rl_scheduler_ep100.pt
result/rl_scheduler_ep200.pt
result/rl_scheduler_ep300.pt   ← keep these
result/rl_scheduler.pt          ← final
```

### Resume from checkpoint

```bash
# Continue from ep 200 checkpoint for 300 more episodes
python -m src.scripts.train_rl_scheduler \
    --resume result/rl_scheduler_ep200.pt \
    --episodes 300 \
    --save_path result/rl_scheduler_ep500.pt
```

> [!NOTE]
> `--resume` loads both the **model weights** and the **training history** (loss log).
> The optimizer state is NOT saved — learning rate warm-up may be needed for fine-tuning.

### Resume PSD training

```bash
python -m src.scripts.train_rl_scheduler \
    --psd \
    --resume result/psd_ep300.pt \
    --episodes 500 \
    --patience 10 \
    --save_path result/psd_ep800.pt
```

> [!WARNING]
> When resuming PSD, the **substrate state is reset to fresh** (deepcopy of original).
> The curriculum does not persist the committed substrate across sessions.
> This is intentional: the agent resumes training from the original problem difficulty.
> To continue from a specific load level, save your substrate as a pickle alongside the model.

---

## 6. Loading a Trained Model for Inference

```python
import torch
from src.rl import VNRSchedulerAgent, PPOTrainer

# ── Load agent ─────────────────────────────────────────────────────────────
agent   = VNRSchedulerAgent()        # uses DEFAULT_CFG
trainer = PPOTrainer(agent)
history = trainer.load('result/rl_scheduler.pt')

print(f"Loaded. Training history: {len(history)} episodes.")

# ── Produce a VNR ordering (deterministic, no grad) ───────────────────────
order = agent.forward_rl_order(
    substrate = my_substrate_graph,   # NetworkX graph
    vnr_list  = my_vnr_list,          # list of NetworkX VNR graphs
)
print(f"RL ordering: {order}")
# e.g. [4, 1, 7, 0, 2, 6, 3, 8, 5, 9]
# → process VNR[4] first, then VNR[1], etc.
```

### Loading a model trained with non-default architecture

```python
cfg = {
    'substrate_hidden': 64,    # match training config
    'vnr_hidden':       32,
    'gru_hidden':       128,
    'K_max':            15,
}
agent   = VNRSchedulerAgent(cfg)
trainer = PPOTrainer(agent)
trainer.load('result/small_model.pt')
```

> [!IMPORTANT]
> The architecture parameters (`substrate_hidden`, `vnr_hidden`, `gru_hidden`,
> `K_max`, etc.) **must match the checkpoint exactly**. They are saved inside
> the `.pt` file — load and inspect them first if unsure:
> ```python
> ckpt = torch.load('result/rl_scheduler.pt', map_location='cpu')
> print(ckpt['cfg'])
> ```

---

## 7. Using the Agent inside `hpso_batch.py`

```python
from src.rl import VNRSchedulerAgent, PPOTrainer
from src.algorithms.hpso_batch import hpso_embed_batch

# Load trained agent
agent   = VNRSchedulerAgent()
trainer = PPOTrainer(agent)
trainer.load('result/rl_scheduler.pt')

# Run batch with RL ordering (replaces revenue-sort heuristic)
accepted, rejected = hpso_embed_batch(
    substrate  = my_substrate,
    batch      = my_vnr_list,
    rl_agent   = agent,        # ← enables RL ordering
    verbose    = True,
)

print(f"Accepted: {len(accepted)}, Rejected: {len(rejected)}")
```

```python
# Original behaviour is kept when rl_agent=None (default)
accepted, rejected = hpso_embed_batch(substrate, batch)
```

---

## 8. Reading Training Logs

Each line of the training log shows:

```
Ep  130/2000 | Acc=0.800 | RC=0.712 | R=0.65±0.34 | Loss=1.823 | Load=42.1% | 387s
│             │           │           │              │             │             │
│             │           │           │              │             │             └ elapsed seconds
│             │           │           │              │             └ substrate CPU load (PSD only)
│             │           │           │              └ average total PPO loss
│             │           │           └ mean±std of per-VNR rewards (non-trivial = real signal)
│             │           └ Revenue/Cost ratio (real embedding cost proxy)
│             └ acceptance rate over last 10 episodes
└ episode / total episodes
```

### What healthy training looks like

| Phase | Acc | RC | R std | What's happening |
|-------|-----|----|----|------------------|
| Standard, easy data | 1.0 | 1.0 | ≈0.0 | No signal — use harder data or --psd |
| PSD early (low load) | 1.0 | 0.85–0.95 | 0.1–0.2 | agent mastering, commits pending |
| PSD mid (30–60% load) | 0.7–0.9 | 0.6–0.8 | 0.2–0.4 | **real gradient, agent learning** |
| PSD late (60–85% load) | 0.5–0.7 | 0.4–0.6 | 0.3–0.5 | strong ordering pressure |

### PSD commit lines

```
[PSD] Commit #3 at ep=47 | Best ep=41 (RC=0.923, 10 VNRs) | Load: 28.7% → 43.2% | 112s
                  ↑              ↑        ↑         ↑                ↑         ↑
            commit index   episode     RC      VNRs embedded    before    after
```

---

## 9. Architecture Configuration

Default architecture (1.1M parameters):

```
SubstrateEncoder: 3-layer GAT, heads=[4,4,1], hidden=128 → h_p ∈ R^128
VNREncoder:       3-layer GAT, heads=[4,4,1], hidden=64  → h_vi ∈ R^64 (shared)
ContextMLP:       [h_p || mean(h_vi)] → R^256
PointerDecoder:   GRU(256) + attention                   → logits
CriticHead:       MLP → R^1
```

Smaller model (faster, fewer params, ~250K):

```bash
python -m src.scripts.train_rl_scheduler \
    --substrate_hidden 64 \
    --vnr_hidden 32 \
    --gru_hidden 128 \
    --critic_hidden 64 \
    --save_path result/small_model.pt
```

---

## 10. Using Your Own Dataset

Prepare a pickle file containing a tuple `(substrate_nx, [vnr_nx, ...])`:

```python
import pickle, networkx as nx

# Your substrate and VNR pool as NetworkX graphs
substrate = ...   # NetworkX graph with node attr 'cpu', 'cpu_res', edge attr 'bw', 'bw_res'
vnr_pool  = [...]  # list of NetworkX graphs with node attr 'cpu', edge attr 'bw'

with open('dataset/my_data.pkl', 'wb') as f:
    pickle.dump((substrate, vnr_pool), f)
```

Required node/edge attributes:

| Object | Key | Required | Used for |
|--------|-----|----------|----------|
| Substrate node | `cpu`, `cpu_res` | Yes | DGL features, resource deduction |
| Substrate node | `mem`, `mem_res` | Optional | DGL features |
| Substrate edge | `bw`, `bw_res` | Yes | DGL features, resource deduction |
| VNR node | `cpu` | Yes | Revenue, reward, DGL features |
| VNR node | `mem` | Optional | DGL features |
| VNR node | `vnf_type` | Optional | DGL features |
| VNR edge | `bw` | Yes | Revenue, reward, resource deduction |

```bash
python -m src.scripts.train_rl_scheduler \
    --data_path dataset/my_data.pkl \
    --psd \
    --episodes 1000 \
    --save_path result/my_trained.pt
```

---

## 11. Common Issues & Fixes

### `DGLError: Device API cuda is not enabled`

DGL CPU-only wheel installed but PyTorch has CUDA. The fix is automatic (falls
back to CPU with a warning). For GPU training, install the CUDA DGL wheel:
```bash
pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
```

### `AssertionError: K_real=10 exceeds K_max=5`

`window_size` > `K_max`. Fix: always set `K_max >= window_size`.
```bash
--window_size 10 --K_max 20   # ✓
--window_size 10 --K_max 5    # ✗ error
```

### `AccRate=1.0, RC=1.0, R±0.00` — no learning signal

The substrate is too easy.
```bash
# Option 1: tighter substrate
--substrate_nodes 15 --vnr_min_nodes 4 --vnr_max_nodes 10

# Option 2: use PSD curriculum
--psd --patience 10

# Option 3: use denser VNRs
--vnr_min_nodes 6 --vnr_max_nodes 12
```

### `reward_std = 0.00` even with PSD

Increase VNR resource demands or decrease substrate capacity so embeddings vary:
```bash
--substrate_nodes 12 --vnr_min_nodes 5 --vnr_max_nodes 12
```

### Model won't load — size mismatch

Always match architecture args to the training config stored in the checkpoint:
```python
import torch
cfg = torch.load('result/model.pt', map_location='cpu')['cfg']
print(cfg)
# then pass the same cfg to VNRSchedulerAgent(cfg)
```
