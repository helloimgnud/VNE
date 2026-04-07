# HRL-ACRA: Comprehensive Developer Guide

> **Paper Reference**: "Joint Admission Control and Resource Allocation of Virtual Network Embedding via Hierarchical Deep Reinforcement Learning", IEEE TSC 2024.
>
> This guide covers the overall architecture, folder structure, running instructions, configuration, and how to extend the framework with new algorithms.

---

## Table of Contents

1. [Background & Core Concepts](#1-background--core-concepts)
2. [Project Folder Structure](#2-project-folder-structure)
3. [Architecture Overview](#3-architecture-overview)
4. [Installation](#4-installation)
5. [Running the Framework](#5-running-the-framework)
6. [Configuration Reference](#6-configuration-reference)
7. [Available Solvers](#7-available-solvers)
8. [Extending the Framework](#8-extending-the-framework)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Tips & Troubleshooting](#10-tips--troubleshooting)

---

## 1. Background & Core Concepts

### Problem: Virtual Network Embedding (VNE)

Virtual Network Embedding (VNE) is the process of mapping **Virtual Network Requests (VNRs)** — tenant-defined topologies with resource requirements — onto a shared **Physical Network (PN)** owned by an Internet Service Provider (ISP). VNE has two tightly coupled sub-problems:

| Sub-Problem | Description |
|---|---|
| **Admission Control (AC)** | Decide whether to accept or reject an incoming VNR |
| **Resource Allocation (RA)** | If accepted, assign physical nodes/links to virtual nodes/links |

### HRL-ACRA Solution

HRL-ACRA (Hierarchical RL for Admission Control and Resource Allocation) uses a **two-level hierarchical agent**:

```
┌─────────────────────────────────────────────────────────────┐
│          Upper-level Agent: hrl_ac (Admission Control)       │
│  Decides: ACCEPT / REJECT (binary decision per VNR)          │
│  Model: GAT-based policy network                             │
└────────────────────────┬────────────────────────────────────┘
                         │ If ACCEPT
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         Lower-level Agent: hrl_ra (Resource Allocation)      │
│  Decides: node mapping one-by-one (sequential embedding)     │
│  Model: Seq2Seq + Attention policy network                   │
└─────────────────────────────────────────────────────────────┘
```

- The **upper-level** agent sees the global network state and the entire VNR, and decides admission.
- The **lower-level** agent iteratively selects physical nodes to host each virtual node, then uses shortest-path routing for links.
- Both agents are trained with **Proximal Policy Optimization (PPO)**.
- Training is sequential: **pretrain lower-level first**, then **train upper-level with frozen/pretrained lower-level**.

---

## 2. Project Folder Structure

```
hrl-acra-main/
│
├── main.py                  # ← Entry point. Parses config, builds scenario, runs it.
├── config.py                # ← ALL configurable arguments defined here (argparse)
├── utils.py                 # ← General utility functions (read/write YAML, path helpers)
├── install.sh               # ← Bash script for conda environment setup
├── __init__.py
├── .gitignore
│
├── settings/                # ← YAML configuration files for simulation parameters
│   ├── general_setting.yaml     # Global flags (save config, ranking method, etc.)
│   ├── p_net_setting.yaml       # Physical network topology & resource settings
│   └── v_sim_setting.yaml       # VNR simulator settings (arrival rate, lifetime, etc.)
│
├── data/                    # ← Data model layer: defines network graph objects
│   ├── __init__.py
│   ├── network.py               # Base Network class (wraps NetworkX graph)
│   ├── physical_network.py      # PhysicalNetwork class (loads/generates p_net)
│   ├── virtual_network.py       # VirtualNetwork class (single VNR)
│   ├── virtual_network_request_simulator.py  # Stream of VNRs with events
│   ├── attribute.py             # Node/link attribute definitions & resource types
│   ├── generator.py             # Random topology generators (Waxman, random)
│   └── utils.py                 # Data layer helpers
│
├── dataset/                 # ← Pre-generated or auto-saved simulation data
│   ├── p_net/                   # Saved physical network (generated once, reused)
│   ├── v_nets/                  # Saved virtual networks + event schedule
│   └── topology/                # Real-world topologies (Brain.gml, Geant.gml)
│
├── base/                    # ← Core simulation engine (env, controller, recorder...)
│   ├── __init__.py
│   ├── environment.py           # Base Environment class (step/reset loop)
│   ├── scenario.py              # BasicScenario: orchestrates env + solver lifecycle
│   ├── controller.py            # Performs actual resource allocation/release on p_net
│   ├── counter.py               # Tracks statistics (acceptance, revenue, R2C ratio)
│   ├── recorder.py              # Logs per-VNR records, writes CSV summaries
│   ├── solution.py              # Solution object: stores node/link mappings
│   ├── loader.py                # Looks up solver by name from SolverLibrary
│   ├── register.py              # SolverLibrary dict + Register class
│   └── utils.py                 # Base layer helpers
│
└── solver/                  # ← All solver algorithms live here
    ├── solver.py                # Abstract base Solver class
    ├── __init__.py
    │
    ├── heuristic/               # Rule-based solvers (no training required)
    │   ├── __init__.py
    │   └── node_rank.py         # GRC, NRM, PL node ranking strategies
    │
    ├── rank/                    # Ranking utilities for link/node ordering
    │   ├── __init__.py
    │   ├── node_rank.py         # Node ranking algorithms
    │   └── link_rank.py         # Link ranking algorithms
    │
    └── learning/                # RL-based solvers
        ├── __init__.py
        ├── rl_solver.py         # Base RLSolver class (PPO training loop, save/load)
        ├── rl_environment.py    # Upper-level RL environment for admission control
        ├── sub_rl_environment.py# Lower-level RL environment for resource allocation
        ├── obs_handler.py       # Builds observation tensors from network state
        ├── net.py               # Neural network building blocks (GNN, attention, MLP)
        ├── model.py             # Actor-Critic model wrappers
        ├── buffer.py            # Rollout buffer for PPO experience collection
        ├── searcher.py          # Beam/greedy search for decoding
        ├── utils.py             # RL-specific helpers
        │
        ├── hrl_ac/              # Upper-level agent: Admission Control
        │   ├── __init__.py
        │   ├── hrl_ac_solver.py     # HrlAcSolver: PPO training for AC agent
        │   ├── env.py               # HrlAcEnv: wraps rl_environment with AC-specific reward
        │   └── net.py               # AC policy network (GAT encoder + MLP policy)
        │
        ├── hrl_ra/              # Lower-level agent: Resource Allocation
        │   ├── __init__.py
        │   ├── hrl_ra_solver.py     # HrlRaSolver: PPO training for RA agent
        │   ├── sub_env.py           # HrlRaSubEnv: node-by-node embedding env
        │   └── net.py               # RA policy network (Seq2Seq + GAT + Attention)
        │
        ├── a3c_gcn/             # Baseline: A3C-GCN solver
        ├── pg_cnn2/             # Baseline: PG-CNN solver
        ├── gae_vne/             # Baseline: GAE-BFS solver
        └── mcts_vne/            # Baseline: MCTS solver
```

---

## 3. Architecture Overview

### 3.1 Execution Flow

```
main.py
  └── get_config()            # Parse settings from CLI + YAML files
  └── load_simulator(name)    # Look up (Env, Solver) from SolverLibrary
  └── BasicScenario.from_config(Env, Solver, config)
        ├── PhysicalNetwork.load/generate()
        ├── VirtualNetworkRequestSimulator.from_setting()
        ├── Controller, Counter, Recorder instantiation
        └── Env(p_net, v_sim, ...) + Solver(...)
  └── scenario.run()
        ├── solver.learn(env, epochs)   [if num_train_epochs > 0]
        └── for each epoch:
              env.reset() → v_net events
              while not done:
                solution = solver.solve(instance)
                next_instance, _, done, info = env.step(solution)
```

### 3.2 Key Class Relationships

| Class | Location | Role |
|---|---|---|
| `BasicScenario` | `base/scenario.py` | Top-level orchestrator |
| `Controller` | `base/controller.py` | Allocates/releases resources on p_net |
| `Recorder` | `base/recorder.py` | Logs records, writes CSVs |
| `Counter` | `base/counter.py` | Computes metrics (R2C, AC rate) |
| `Solution` | `base/solution.py` | Mapping result container |
| `PhysicalNetwork` | `data/physical_network.py` | The substrate network graph |
| `VirtualNetworkRequestSimulator` | `data/virtual_network_request_simulator.py` | Event stream of VNRs |
| `Solver` | `solver/solver.py` | Abstract base for all solvers |
| `RLSolver` | `solver/learning/rl_solver.py` | PPO training base class |
| `HrlAcSolver` | `solver/learning/hrl_ac/` | Upper-level AC agent |
| `HrlRaSolver` | `solver/learning/hrl_ra/` | Lower-level RA agent |

### 3.3 Solver Registration

Every solver must register itself in `SolverLibrary` via `Register.register()`. This is done in each solver's `__init__.py`:

```python
# Example from solver/learning/hrl_ac/__init__.py
from base.register import Register
from .hrl_ac_solver import HrlAcSolver
from .env import HrlAcEnv

Register.register('hrl_ac', {'solver': HrlAcSolver, 'env': HrlAcEnv})
```

When you run `--solver_name=hrl_ac`, `loader.py` looks up `SolverLibrary['hrl_ac']` and returns the `(Env, Solver)` pair.

---

## 4. Installation

### Prerequisites
- Anaconda / Miniconda
- CUDA 10.2 or 11.3 (for GPU acceleration, recommended)

### With GPU (CUDA 11.3)

```bash
sh install.sh -c=11.3
```

### With CPU only

```bash
sh install.sh
```

The script creates a conda environment and installs all dependencies including PyTorch, PyTorch Geometric, and NetworkX.

---

## 5. Running the Framework

### Step 1 — Pretrain the Lower-level Agent (Resource Allocation)

This trains `hrl_ra` alone, which learns to embed VNRs greedily without an admission controller.

```bash
python main.py \
  --solver_name="hrl_ra" \
  --eval_interval=10 \
  --num_train_epochs=100 \
  --summary_file_name="exp-hrl_ra-training.csv" \
  --seed=0
```

Saved model will be in: `save/hrl_ra/<run_id>/`

### Step 2 — Train the Upper-level Agent (Admission Control)

This trains `hrl_ac` using the pretrained `hrl_ra` as a sub-solver.

```bash
python main.py \
  --solver_name="hrl_ac" \
  --sub_solver_name="hrl_ra" \
  --eval_interval=10 \
  --num_train_epochs=500 \
  --summary_file_name="exp-hrl_ac-training.csv" \
  --pretrained_subsolver_model_path="save/hrl_ra/<run_id>/model.pkl" \
  --seed=0
```

Replace `<run_id>` with the actual run directory (format: `hostname-YYYYMMDDTHHMMSS`).

### Step 3 — Test the Full HRL-ACRA System

```bash
python main.py \
  --solver_name="hrl_ac" \
  --sub_solver_name="hrl_ra" \
  --decode_strategy="beam" \
  --k_searching=3 \
  --num_train_epochs=0 \
  --pretrained_model_path="save/hrl_ac/<run_id>/model.pkl" \
  --pretrained_subsolver_model_path="save/hrl_ra/<run_id>/model.pkl" \
  --summary_file_name="exp-hrl_acra-testing.csv" \
  --seed=0
```

- `--decode_strategy="beam"` with `--k_searching=N` enables beam search (N=1 → greedy)
- `--num_train_epochs=0` → inference only, no training

### Running Heuristic Baselines (no pretraining)

```bash
# GRC baseline
python main.py --solver_name="grc_rank" --summary_file_name="grc.csv" --seed=0

# NRM baseline
python main.py --solver_name="nrm_rank" --summary_file_name="nrm.csv" --seed=0

# PL baseline
python main.py --solver_name="pl_rank" --summary_file_name="pl.csv" --seed=0
```

### Running Learning-based Baselines

```bash
# Train A3C-GCN
python main.py --solver_name="a3c_gcn" --num_train_epochs=100 --eval_interval=10 --seed=0

# Test A3C-GCN
python main.py --solver_name="a3c_gcn" --num_train_epochs=0 \
  --pretrained_model_path="save/a3c_gcn/<run_id>/model.pkl" --seed=0
```

---

## 6. Configuration Reference

Configuration is managed through **three YAML files** (read at startup) and **CLI arguments** (override anything). All parameters are defined in `config.py`.

### `settings/p_net_setting.yaml` — Physical Network

| Key | Default | Description |
|---|---|---|
| `num_nodes` | 100 | Number of physical nodes |
| `topology.type` | `waxman` | Topology type (`waxman` or use `file_path` for real topologies) |
| `topology.wm_alpha` | 0.5 | Waxman α parameter (higher → denser) |
| `topology.wm_beta` | 0.2 | Waxman β parameter |
| `node_attrs_setting[cpu].high` | 100 | Max CPU capacity per node |
| `node_attrs_setting[cpu].low` | 50 | Min CPU capacity per node |
| `link_attrs_setting[bw].high` | 100 | Max bandwidth capacity per link |
| `link_attrs_setting[bw].low` | 50 | Min bandwidth capacity per link |

**Use real-world topologies:**
```yaml
topology:
  file_path: './dataset/topology/Brain.gml'   # or Geant.gml
```

### `settings/v_sim_setting.yaml` — VNR Simulator

| Key | Default | Description |
|---|---|---|
| `num_v_nets` | 1000 | Total VNRs to simulate |
| `v_net_size.high` | 10 | Max VN nodes |
| `v_net_size.low` | 2 | Min VN nodes |
| `arrival_rate.lam` | 0.04 | Poisson arrival rate λ |
| `lifetime.scale` | 1000 | Mean lifetime (exponential distribution) |
| `node_attrs_setting[cpu].high` | 50 | Max CPU demand per VN node |
| `link_attrs_setting[bw].high` | 50 | Max BW demand per VN link |

**Common experiment variations:**

```yaml
# Higher load
arrival_rate:
  lam: 0.08

# Larger VNRs
v_net_size:
  high: 20

# Higher resource demand
node_attrs_setting:
  - high: 90
link_attrs_setting:
  - high: 90
```

### `settings/general_setting.yaml` — Global Options

| Key | Default | Description |
|---|---|---|
| `if_save_config` | `true` | Save the full config to `save/` on each run |
| `node_ranking_method` | `order` | How to rank VN nodes for embedding priority |
| `link_ranking_method` | `order` | How to rank VN links |

### Key CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--solver_name` | `nrm_rank` | Which solver to use |
| `--sub_solver_name` | `nrm_rank` | Sub-solver for hierarchical methods |
| `--num_train_epochs` | 100 | Training epochs (0 = test only) |
| `--eval_interval` | 10 | Evaluate every N epochs |
| `--seed` | None | Random seed for reproducibility |
| `--use_cuda` | True | Use GPU |
| `--embedding_dim` | 128 | GNN embedding dimension |
| `--num_gnn_layers` | 5 | Number of GNN layers |
| `--lr` | 1e-3 | Learning rate |
| `--rl_gamma` | 0.99 | Discount factor |
| `--batch_size` | 256 | PPO batch size |
| `--decode_strategy` | `greedy` | Decoding: `greedy` or `beam` |
| `--k_searching` | 3 | Beam width (if beam search) |
| `--pretrained_model_path` | `''` | Path to upper-level model checkpoint |
| `--pretrained_subsolver_model_path` | `''` | Path to lower-level model checkpoint |
| `--summary_file_name` | `global_summary.csv` | Output CSV filename in `save/` |
| `--save_dir` | `save` | Top-level save directory |

---

## 7. Available Solvers

| Solver Name | Category | Description | Needs Training |
|---|---|---|---|
| `nrm_rank` | Heuristic | Node Resource Management ranking | ❌ |
| `grc_rank` | Heuristic | Global Resource Capacity ranking | ❌ |
| `pl_rank` | Heuristic | Path-Length ranking | ❌ |
| `mcts_vne` | Learning | Monte Carlo Tree Search | ❌ |
| `gae_vne` | Learning | Graph AutoEncoder + BFS | ❌ |
| `a3c_gcn` | Learning | Asynchronous A3C with GCN | ✅ |
| `pg_cnn2` | Learning | Policy Gradient with CNN | ✅ |
| `hrl_ra` | **Ours** | Lower-level RA agent (pretrain first) | ✅ |
| `hrl_ac` | **Ours** | Upper-level AC agent (train second) | ✅ |

---

## 8. Extending the Framework

### 8.1 Adding a New Heuristic Solver

1. **Create the solver file** in `solver/heuristic/` or as a new directory:

```python
# solver/heuristic/my_heuristic.py
from solver.solver import Solver
from base.solution import Solution

class MyHeuristicSolver(Solver):
    def __init__(self, controller, recorder, counter, **kwargs):
        super().__init__(controller, recorder, counter, **kwargs)

    def solve(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        solution = Solution(v_net)
        # --- Your logic here ---
        # e.g., rank nodes, greedily match, find paths
        # solution.node_slots = {vnode: pnode, ...}
        # solution.link_paths = {(u,v): [path], ...}
        return solution
```

2. **Register the solver** in `solver/heuristic/__init__.py`:

```python
from base.register import Register
from base.environment import BasicEnvironment  # or a custom Env
from .my_heuristic import MyHeuristicSolver

Register.register('my_heuristic', {
    'solver': MyHeuristicSolver,
    'env': BasicEnvironment
})
```

3. **Import it** in `solver/__init__.py` or `solver/heuristic/__init__.py` so it is loaded at startup.

4. **Run it**:

```bash
python main.py --solver_name="my_heuristic" --seed=0
```

### 8.2 Adding a New Learning-based Solver

1. **Create a folder** under `solver/learning/my_rl_solver/`:

```
solver/learning/my_rl_solver/
├── __init__.py         # registers the solver
├── my_solver.py        # inherits from RLSolver
├── env.py              # custom RL environment (inherits from RLEnvironment)
└── net.py              # policy network (actor-critic)
```

2. **Define the policy network** in `net.py`:

```python
import torch.nn as nn

class MyPolicyNet(nn.Module):
    def __init__(self, p_net_num_nodes, p_net_num_node_attrs, ...):
        super().__init__()
        # Build your GNN / MLP architecture
        ...

    def forward(self, obs):
        # Return action logits and value estimate
        return action_logits, value
```

3. **Define the RL environment** in `env.py`:

```python
from solver.learning.rl_environment import RLEnvironment

class MyRLEnv(RLEnvironment):
    def compute_reward(self, solution):
        # Define your own reward signal
        ...
```

4. **Define the solver** in `my_solver.py`:

```python
from solver.learning.rl_solver import RLSolver
from .net import MyPolicyNet
from .env import MyRLEnv

class MyRLSolver(RLSolver):
    def __init__(self, controller, recorder, counter, **kwargs):
        super().__init__(controller, recorder, counter, **kwargs)
        self.policy = MyPolicyNet(...)

    def solve(self, instance):
        # Use self.policy to select actions
        ...
```

5. **Register in `__init__.py`**:

```python
from base.register import Register
from .my_solver import MyRLSolver
from .env import MyRLEnv

Register.register('my_rl_solver', {'solver': MyRLSolver, 'env': MyRLEnv})
```

6. **Import** in `solver/learning/__init__.py` and train:

```bash
python main.py --solver_name="my_rl_solver" --num_train_epochs=100 --seed=0
```

### 8.3 Adding New Resource Attributes

The attribute system is defined in `data/attribute.py`. To add a new resource (e.g., `memory`):

1. **Update `p_net_setting.yaml`**:
```yaml
node_attrs_setting:
  - name: cpu
    ...
  - name: memory          # new attribute
    distribution: uniform
    dtype: int
    generative: true
    high: 64
    low: 16
    owner: node
    type: resource
  - name: max_memory
    originator: memory
    owner: node
    type: extrema
```

2. **Update `v_sim_setting.yaml`** similarly with the demand range for `memory`.

3. The framework will automatically include memory in node embedding observations and enforce memory constraints during allocation — no code changes needed for the base engine.

### 8.4 Using a Custom Physical Network Topology

Place a `.gml` file in `dataset/topology/` and update `p_net_setting.yaml`:

```yaml
topology:
  file_path: './dataset/topology/MyTopology.gml'
```

The `gml` file must have integer node IDs. You can export from NetworkX:
```python
import networkx as nx
G = nx.karate_club_graph()
nx.relabel_nodes(G, {n: i for i, n in enumerate(G.nodes())}, copy=False)
nx.write_gml(G, 'dataset/topology/MyTopology.gml')
```

---

## 9. Evaluation Metrics

Results are saved to `save/<summary_file_name>`. Key metrics:

| Metric | Description |
|---|---|
| **Acceptance Rate (AC)** | `accepted / total_arrived` VNRs |
| **Revenue-to-Cost Ratio (R2C)** | Total revenue earned / total resources consumed |
| **Revenue** | Sum of accepted VNR resource demands × service time |
| **Cost** | Sum of physical resources consumed |
| **In-service Count** | Number of VNRs currently hosted |

During a run, the progress bar shows live values:
```
Running with hrl_ac: 100%|███| 1000/1000 [ac: 0.87, r2c: 1.23, inservice: 00312]
```

---

## 10. Tips & Troubleshooting

### Dataset Reuse

Once a physical network is generated, it is saved to `dataset/p_net/` and reused on subsequent runs (same topology, same resources). To force regeneration, delete the directory:

```bash
rm -rf dataset/p_net/
```

VNR event files are similarly cached in `dataset/v_nets/`. Use `--renew_v_net_simulator=True` to regenerate:

```bash
python main.py --solver_name="hrl_ra" --renew_v_net_simulator=True ...
```

### Model Checkpoint Paths

Run directories follow the pattern `save/<solver_name>/<hostname>-<YYYYMMDDTHHMMSS>/`. Look for `model.pkl` or `model_best.pkl` inside. Example:

```
save/
└── hrl_ra/
    └── mypc-20240101T120000/
        ├── model.pkl
        ├── model_best.pkl
        └── config.yaml
```

### CPU-Only Mode

If no GPU is available, add `--use_cuda=False`:

```bash
python main.py --solver_name="hrl_ra" --use_cuda=False --num_train_epochs=100
```

### Reproducibility

Always set `--seed=<int>` for reproducible results:

```bash
python main.py --solver_name="hrl_ac" --seed=42 ...
```

### TensorBoard Logging

Training logs are written to `log/` by default. View with:

```bash
tensorboard --logdir=log/
```

Disable with `--open_tb=False`.

### Common Errors

| Error | Likely Cause | Fix |
|---|---|---|
| `KeyError: 'hrl_ac'` | Solver not imported/registered | Check `solver/learning/__init__.py` imports |
| `FileNotFoundError: model.pkl` | Wrong pretrained path | Use full/correct path to `.pkl` file |
| `CUDA out of memory` | Large batch or model | Reduce `--batch_size` or `--embedding_dim` |
| Dataset not found / mismatch | Setting changed after dataset saved | Delete `dataset/p_net/` and regenerate |

---

## Related Resources

- **[Virne](https://github.com/GeminiLight/virne)** — A full benchmark framework for VNE, integrating HRL-ACRA as `ppo_gat_seq2seq+`.
- **[SDN-NFV-Papers](https://github.com/GeminiLight/sdn-nfv-papers)** — Curated reading list on NFV/SDN resource allocation.
- **Paper DOI**: [10.1109/TSC.2023.3326557](https://ieeexplore.ieee.org/abstract/document/10291038)
