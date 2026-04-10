# Network Encoder RL — Usage Guide

## Quick-Start Cheatsheet

```bash
# Phase 1: Train from scratch (REINFORCE, ~10 min on CPU)
python -m src.training.train_reinforce --episodes 2000 --reward simple

# Phase 2: Upgrade to PPO with revenue reward
python -m src.training.train_ppo --total-steps 500000 --reward revenue

# Evaluate a checkpoint vs baseline
python -m src.training.evaluate --checkpoint checkpoints/ppo_phase2_final.pt --episodes 100

# Run inference in your own code (see §4)
```

---

## Table of Contents

1. [Prerequisites & Installation](#1-prerequisites--installation)
2. [Project Structure](#2-project-structure)
3. [Training the Scheduler](#3-training-the-scheduler)
4. [Using the Scheduler for Inference](#4-using-the-scheduler-for-inference)
5. [Plugging Into Your Existing HPSO Pipeline](#5-plugging-into-your-existing-hpso-pipeline)
6. [Evaluating and Comparing to Baseline](#6-evaluating-and-comparing-to-baseline)
7. [Configuration Reference](#7-configuration-reference)
8. [Extending the System](#8-extending-the-system)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites & Installation

### Required packages

```bash
pip install torch>=2.2
pip install torch_geometric>=2.5          # PyG
pip install gymnasium>=0.29
pip install networkx                       # already in your stack
```

> [!NOTE]
> PyG has optional CUDA extensions. For CPU-only training (small substrates),
> the base install is sufficient. For GPU: follow the
> [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
> and install the matching `torch-scatter`, `torch-sparse`, `torch-cluster` wheels.

### Optional (for PPO + tracking)
```bash
pip install tensorboard         # training curves (optional)
```

### Verify installation

```python
import torch, torch_geometric, gymnasium, networkx
print(torch.__version__, torch_geometric.__version__)

# Sanity check: build a scheduler and run a forward pass
from src.scheduler import VNRScheduler, substrate_to_pyg, vnr_to_pyg
from src.generators.substrate_generator import generate_substrate
from src.generators.vnr_generator import generate_single_vnr

sub = generate_substrate(num_nodes_total=10, seed=0)
vnr = generate_single_vnr(num_nodes=3)

model = VNRScheduler(use_batch_context=False)
scores = model(substrate_to_pyg(sub), [vnr_to_pyg(vnr)])
print("scores:", scores)   # should print: tensor([...])
```

---

## 2. Project Structure

```
src/
├── scheduler/            GCN-RL model components
│   ├── features.py       networkx → PyG conversion
│   ├── encoders.py       SubstrateGCN, VNRGCN, BatchContextEncoder
│   ├── model.py          VNRScheduler (main class)
│   ├── policy.py         GNNActorCritic (for PPO)
│   ├── environment.py    VNEOrderingEnv (gymnasium.Env)
│   └── rewards.py        pluggable reward functions
│
├── training/             training scripts
│   ├── generate_data.py  substrate_fn / batch_fn factories
│   ├── train_reinforce.py  Phase 1 REINFORCE
│   ├── train_ppo.py        Phase 2 PPO
│   └── evaluate.py         evaluation & comparison
│
└── algorithms/
    ├── hpso_batch.py           original (unchanged)
    ├── hpso_batch_scheduler.py  NEW: plugin integration point
    └── fast_hpso.py            original (unchanged)
```

---

## 3. Training the Scheduler

### 3.1 Phase 1 — REINFORCE (start here)

Validates that the GNN architecture can learn a useful ordering signal.

```bash
python -m src.training.train_reinforce \
    --episodes 2000 \
    --reward simple \
    --sub-nodes 50 \
    --batch-size 10 \
    --fixed-sub \
    --save-dir checkpoints \
    --run-name phase1
```

**Recommended progression:**

| Step | Command addition | Why |
|------|-----------------|-----|
| Start | `--fixed-sub --sub-nodes 30 --batch-size 5` | Fixed, small substrate = easiest learning signal |
| Step 2 | `--sub-nodes 50 --batch-size 10` (no `--fixed-sub`) | Add substrate diversity |
| Step 3 | `--reward revenue` | Add R/C signal |

**What to watch:**
```
Ep   100/2000 | loss=-0.2341 | reward=3.500 | AR=70.00% | R/C=1.234 | t=45s
Ep   200/2000 | loss=-0.4120 | reward=5.100 | AR=80.00% | R/C=1.456 | t=89s
```
AR should trend upward over training. If AR stays at ~0% for >500 episodes, the substrate is too tight — reduce `--batch-size` or `--sub-nodes`.

### 3.2 Phase 2 — PPO (after Phase 1 AR > baseline)

```bash
python -m src.training.train_ppo \
    --total-steps 500000 \
    --reward revenue \
    --sub-nodes 50 \
    --vnr-batch 10 \
    --save-dir checkpoints \
    --run-name phase2_ppo
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--total-steps` | 500000 | Total environment interaction steps |
| `--n-steps` | 512 | Rollout buffer size before each update |
| `--n-epochs` | 10 | Gradient passes per rollout |
| `--clip` | 0.2 | PPO surrogate clip range |
| `--ent-coef` | 0.01 | Entropy bonus (exploration) |
| `--no-ctx` | off | Disable BatchContextEncoder (Phase 1 model) |
| `--reward` | revenue | `simple` / `revenue` / `longterm` |

### 3.3 Phase 3 — Long-term reward

```bash
python -m src.training.train_ppo \
    --total-steps 1000000 \
    --reward longterm \
    --sub-nodes 50 \
    --vnr-batch 15 \
    --run-name phase3_longterm
```

### 3.4 Python API (programmatic training)

```python
from src.training.train_reinforce import ReinforceTrainer, ReinforceConfig

# Configure
cfg = ReinforceConfig(
    num_episodes    = 3000,
    reward_mode     = "simple",
    use_batch_context = False,     # Phase 1: no context encoder
    substrate_nodes = 50,
    batch_size      = 10,
    fixed_substrate = True,        # easier curriculum start
    save_dir        = "my_runs",
    run_name        = "exp01",
)

# Train
trainer = ReinforceTrainer(cfg)
trainer.train()

# Quick evaluation after training
results = trainer.evaluate(n_episodes=50, verbose=True)
# {"mean_ar": 0.82, "mean_rc": 1.45, "mean_reward": 6.3}
```

```python
from src.training.train_ppo import PPOTrainerScheduler, PPOConfig

cfg = PPOConfig(
    total_timesteps = 500_000,
    reward_mode     = "revenue",
    use_batch_context = True,
    save_dir        = "checkpoints",
    run_name        = "ppo_exp01",
)
trainer = PPOTrainerScheduler(cfg)
trainer.train()
trainer.evaluate(n_episodes=50, verbose=True)
```

### 3.5 Checkpoints

Checkpoints are saved automatically:
- Every `save_every` steps/episodes → `{save_dir}/{run_name}_ep{N}.pt` or `_step{N}.pt`
- At end of training → `{save_dir}/{run_name}_final.pt`

The `.pt` file contains:
```python
{
    "state_dict": { ... },         # model weights
    "use_batch_context": True,     # config flag
    "meta": { "config": { ... } }  # training config
}
```

---

## 4. Using the Scheduler for Inference

### 4.1 Load a Checkpoint

```python
from src.scheduler import VNRScheduler

scheduler = VNRScheduler.load(
    "checkpoints/phase2_ppo_final.pt",
    device="cpu",                  # or "cuda"
    use_batch_context=True,        # must match training config
)
# scheduler is in eval() mode automatically
```

> [!IMPORTANT]
> `use_batch_context` must match the value used during training.
> Phase 1 checkpoints: `use_batch_context=False`
> Phase 2+ checkpoints: `use_batch_context=True`

### 4.2 Score a Batch of VNRs

```python
from src.scheduler.features import substrate_to_pyg, vnr_to_pyg

# Your NetworkX graphs
substrate = ...   # networkx.Graph with cpu/cpu_total, bw/bw_total attrs
vnr_list  = [...]  # list of networkx.Graph with cpu, bw attrs

# Convert to PyG
sub_pyg   = substrate_to_pyg(substrate)
vnr_pygs  = [vnr_to_pyg(v) for v in vnr_list]

# Get scores (higher = embed first)
scores = scheduler.predict(sub_pyg, vnr_pygs)  # Tensor[B], no grad

# Get optimal ordering
order = scores.argsort(descending=True).tolist()
# order[0] = index of the VNR to embed first
```

### 4.3 Get Ordered VNR List Directly

```python
from src.training.evaluate import run_inference

order = run_inference(scheduler, substrate, vnr_list)
ordered_vnrs = [vnr_list[i] for i in order]
```

### 4.4 Full Inference with Embedding

```python
from src.algorithms.hpso_batch_scheduler import hpso_embed_batch_scheduled

accepted, rejected = hpso_embed_batch_scheduled(
    substrate  = substrate,     # mutated in-place on success
    batch      = vnr_list,
    scheduler  = scheduler,     # pass the loaded model
    particles  = 20,
    iterations = 30,
    verbose    = True,
)

print(f"Accepted: {len(accepted)}, Rejected: {len(rejected)}")
for vnr, mapping, link_paths in accepted:
    print(f"  VNR nodes={len(vnr.nodes())}, mapping={mapping}")
```

---

## 5. Plugging Into Your Existing HPSO Pipeline

### 5.1 Zero Change to Original Code

The original `hpso_batch.py` is **not modified**. To upgrade existing call sites:

```python
# BEFORE (original behaviour — revenue sort)
from src.algorithms.hpso_batch import hpso_embed_batch
accepted, rejected = hpso_embed_batch(substrate, batch)

# AFTER (GNN ordering) — just change the import + add scheduler argument
from src.algorithms.hpso_batch_scheduler import hpso_embed_batch_scheduled
accepted, rejected = hpso_embed_batch_scheduled(substrate, batch, scheduler=scheduler)

# AFTER (still revenue-sort, optional migration) — backward-compatible alias
from src.algorithms.hpso_batch_scheduler import hpso_embed_batch
accepted, rejected = hpso_embed_batch(substrate, batch)  # scheduler=None → revenue-sort
```

### 5.2 Conditional Plugin (enable/disable at runtime)

```python
from src.algorithms.hpso_batch_scheduler import hpso_embed_batch_scheduled
from src.scheduler import VNRScheduler

# Load once at program start (or pass None to disable)
scheduler = VNRScheduler.load("checkpoints/final.pt") if USE_GNN else None

# Same call site for both modes
accepted, rejected = hpso_embed_batch_scheduled(
    substrate, batch,
    scheduler=scheduler,   # None → revenue-sort fallback
)
```

### 5.3 Integration with `hpso_batch_rl.py`

The existing `hpso_batch_rl.py` (which routes to the earlier DGL-based agent) is
**separate and unaffected**. If you want to switch to the new PyG-based scheduler:

```python
# Use the new scheduler module instead of src.rl.VNRSchedulerAgent
from src.scheduler import VNRScheduler
from src.algorithms.hpso_batch_scheduler import hpso_embed_batch_scheduled

scheduler = VNRScheduler.load("checkpoints/ppo_phase2_final.pt")
accepted, rejected = hpso_embed_batch_scheduled(substrate, batch, scheduler=scheduler)
```

### 5.4 Using Inside the Existing RL Environment (`src/rl/env.py`)

If you want the new GCN scheduler to order VNRs *before* passing them to the existing `SchedulerEnv`:

```python
from src.training.evaluate import run_inference
from src.scheduler import VNRScheduler

scheduler = VNRScheduler.load("checkpoints/final.pt")

# Pre-order the VNR list
order      = run_inference(scheduler, substrate, vnr_list)
ordered    = [vnr_list[i] for i in order]

# Then pass to existing SchedulerEnv (already ordered)
env = SchedulerEnv(vnr_list=ordered, substrate=substrate,
                   hpso_embed_fn=hpso_embed)
```

---

## 6. Evaluating and Comparing to Baseline

### 6.1 Command-Line Evaluation

```bash
python -m src.training.evaluate \
    --checkpoint checkpoints/ppo_phase2_final.pt \
    --episodes 100 \
    --sub-nodes 50 \
    --vnr-batch 10 \
    --hpso-iter 30
```

Output:
```
=== Evaluation Report (100 episodes) ===
  Acceptance Rate : scheduler=84.20%  baseline=78.50%  Δ=+5.70%
  Revenue/Cost    : scheduler=1.342   baseline=1.198   Δ=+0.144
```

### 6.2 Python API Evaluation

```python
from src.training.evaluate import evaluate_scheduler
from src.scheduler import VNRScheduler
from src.training.generate_data import make_env_fns

scheduler = VNRScheduler.load("checkpoints/ppo_phase2_final.pt")
substrate_fn, batch_fn = make_env_fns(substrate_nodes=50, batch_size=10)

report = evaluate_scheduler(
    scheduler, substrate_fn, batch_fn,
    n_episodes=100, verbose=True
)

print(report.to_dict())
# {'n_episodes': 100, 'scheduler_ar': 0.842, 'baseline_ar': 0.785,
#  'delta_ar': 0.057, 'scheduler_rc': 1.342, 'baseline_rc': 1.198, ...}
```

### 6.3 Evaluate on Your Own Test Set

```python
from src.training.evaluate import run_inference, _embed_with_order, _episode_metrics, _revenue_sort
from src.utils.graph_utils import copy_substrate

# Your test substrates and batches
test_cases = [(substrate_1, vnr_batch_1), (substrate_2, vnr_batch_2), ...]

hpso_params = dict(particles=20, iterations=30)
sched_ars, base_ars = [], []

for substrate, vnr_list in test_cases:
    # GNN order
    order    = run_inference(scheduler, substrate, vnr_list)
    acc, rej = _embed_with_order(substrate, vnr_list, order, hpso_params)
    sched_ars.append(len(acc) / (len(acc) + len(rej) + 1e-9))

    # Baseline order
    order    = _revenue_sort(vnr_list)
    acc, rej = _embed_with_order(substrate, vnr_list, order, hpso_params)
    base_ars.append(len(acc) / (len(acc) + len(rej) + 1e-9))

print(f"GNN AR: {sum(sched_ars)/len(sched_ars):.2%}")
print(f"Base AR: {sum(base_ars)/len(base_ars):.2%}")
```

---

## 7. Configuration Reference

### `ReinforceConfig` (Phase 1)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_episodes` | 2000 | Total training episodes |
| `lr` | 3e-4 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `grad_clip` | 0.5 | Gradient norm clip |
| `reward_mode` | `"simple"` | `simple` / `revenue` / `longterm` |
| `use_batch_context` | `False` | Enable BatchContextEncoder |
| `substrate_nodes` | 50 | Substrate network size |
| `batch_size` | 10 | VNRs per episode |
| `vnr_nodes` | 4 | Average VNR size |
| `fixed_substrate` | `False` | Reuse same substrate each episode |
| `hpso_particles` | 20 | HPSO particle count |
| `hpso_iterations` | 30 | HPSO iteration limit |
| `log_every` | 100 | Print log every N episodes |
| `save_every` | 500 | Save checkpoint every N episodes |
| `save_dir` | `"checkpoints"` | Checkpoint directory |
| `run_name` | `"reinforce_phase1"` | Checkpoint file prefix |
| `device` | `"auto"` | `"auto"` / `"cpu"` / `"cuda"` |

### `PPOConfig` (Phase 2)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_timesteps` | 500000 | Total env steps |
| `n_steps` | 512 | Rollout buffer size |
| `batch_size` | 64 | Mini-batch size for gradient update |
| `n_epochs` | 10 | Gradient reuse per rollout |
| `lr` | 3e-4 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_range` | 0.2 | PPO surrogate clip |
| `ent_coef` | 0.01 | Entropy bonus coefficient |
| `vf_coef` | 0.5 | Value function loss coefficient |
| `grad_clip` | 0.5 | Gradient norm clip |
| `reward_mode` | `"revenue"` | Reward mode |
| `use_batch_context` | `True` | Enable BatchContextEncoder |
| `substrate_nodes` | 50 | Substrate size |
| `batch_size_env` | 10 | VNRs per episode |
| `vnr_nodes` | 4 | Average VNR size |

### `VNRScheduler` model kwargs

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_batch_context` | `True` | Phase 2+: enable Transformer batch context |
| `substrate_kwargs` | `{}` | Override SubstrateGCN defaults (e.g. `{"hidden": 128}`) |
| `vnr_kwargs` | `{}` | Override VNRGCN defaults |
| `context_kwargs` | `{}` | Override BatchContextEncoder defaults |

---

## 8. Extending the System

### 8.1 Custom Reward Function

```python
# src/scheduler/rewards.py

class RewardMode(str, enum.Enum):
    SIMPLE   = "simple"
    REVENUE  = "revenue"
    LONGTERM = "longterm"
    CUSTOM   = "custom_name"    # ← add here

def _reward_custom_name(success, vnr, done, accepted, rejected):
    # Your logic
    base = 2.0 if success else -1.0
    if done:
        base += len(accepted) * 0.5
    return base

_REWARD_FNS[RewardMode.CUSTOM] = _reward_custom_name
```

Usage: `ReinforceConfig(reward_mode="custom_name")`

### 8.2 Custom Substrate Features

Extend `substrate_to_pyg()` in `src/scheduler/features.py`:

```python
def substrate_to_pyg(G: nx.Graph) -> Data:
    ...
    for n in nodes:
        # Add domain-awareness as a 6th feature
        domain = float(G.nodes[n].get("domain", -1))
        x_rows.append([cpu_av, cpu_ratio, deg, avg_bw, clust, domain])
    ...
    x = torch.tensor(x_rows, dtype=torch.float)  # [N_s, 6]
```

Then update model to match:
```python
scheduler = VNRScheduler(substrate_kwargs={"in_dim": 6})
```

### 8.3 Larger / Deeper GNNs

```python
scheduler = VNRScheduler(
    use_batch_context=True,
    substrate_kwargs=dict(in_dim=5, hidden=128, heads=8, out_dim=256),
    vnr_kwargs=dict(in_dim=4, hidden=64, heads=4, out_dim=128),
    context_kwargs=dict(vnr_dim=128, out_dim=128, nhead=8, num_layers=2),
)
```

> [!NOTE]
> If you change `out_dim` in substrate or VNR encoder, the ScoringMLP input
> dimension is computed automatically as `s_dim + v_dim + ctx_dim`.

### 8.4 Integrating with a Real Substrate (Deployment Mode)

```python
from src.algorithms.hpso_batch_scheduler import hpso_embed_batch_scheduled

# Your live substrate (resources are updated in-place on success)
live_substrate = load_current_substrate(...)

scheduler = VNRScheduler.load("checkpoints/final.pt")
accepted, rejected = hpso_embed_batch_scheduled(
    live_substrate, incoming_vnr_batch, scheduler=scheduler
)

# Update accounting, log results, etc.
```

---

## 9. Troubleshooting

### `ImportError: No module named 'torch_geometric'`

```bash
pip install torch_geometric
# For GPU with CUDA 11.8:
pip install torch-scatter torch-sparse torch-cluster --find-links \
    https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

### `use_batch_context mismatch` — wrong output shape

The checkpoint was trained with `use_batch_context=True` but you loaded with `False` (or vice versa). Always pass the matching flag:
```python
VNRScheduler.load("ckpt.pt", use_batch_context=True)  # match training config
```

### `AR = 0%` after many episodes

1. Substrate is too small / resource-constrained → try `--sub-nodes 30 --batch-size 3`
2. VNR demands too high → reduce `vnr_cpu_range` in `make_batch_fn`
3. HPSO particle count too low → add `--hpso-iter 50`

### Very slow training (CPU)

Reduce HPSO iterations which dominate wall-clock time:
```bash
python -m src.training.train_reinforce --hpso-iter 10 --episodes 500
```
Or use `--sub-nodes 20` (fewer substrate nodes = faster HPSO).

### `RuntimeError: Expected all tensors to be on the same device`

The PyG data is on CPU but the model is on CUDA. Move data to device:
```python
sub_data  = obs["substrate"].to(device)
vnr_datas = [v.to(device) for v in obs["vnr_list"]]
```
This is handled automatically inside `ReinforceTrainer` and `PPOTrainerScheduler`.

### Checkpoint loading fails with size mismatch

The model architecture changed between training runs. Retrain from scratch or load with matching constructor kwargs:
```python
# If you trained with larger hidden dim:
VNRScheduler.load("ckpt.pt", substrate_kwargs={"hidden": 128})
```
