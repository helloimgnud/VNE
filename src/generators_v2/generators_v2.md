# generators_v2 — Comprehensive Reference

This document describes the `src/generators_v2` package: why it was created, how it differs
from the original `src/generators`, what virne brings to the table, and how to use every
public function and class.

---

## Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Architecture Overview](#2-architecture-overview)
3. [Module Map](#3-module-map)
4. [Substrate Generators](#4-substrate-generators)
5. [VNR Generators](#5-vnr-generators)
6. [Topology Types](#6-topology-types)
7. [Dataset Generators](#7-dataset-generators)
8. [Virne Integration Details](#8-virne-integration-details)
9. [Migration from src/generators](#9-migration-from-srcgenerators)
10. [Usage Examples](#10-usage-examples)
11. [JSON File Formats](#11-json-file-formats)
12. [Design Decisions & Limitations](#12-design-decisions--limitations)

---

## 1. Background & Motivation

The original `src/generators` package contains three files:
| File | What it does |
|---|---|
| `generator.py` | Monolithic: substrate + VNR stream v1 + v2 |
| `substrate_generator.py` | Refactored substrate (identical functionality) |
| `vnr_generator.py` | Refactored VNRs (identical functionality) |
| `dataset_generator.py` | Orchestrates fig6 / fig7 / fig8 datasets |

These generators produce lightweight **JSON + `nx.Graph`** output and work well for the existing experiment pipeline.

**virne** (`virne-main/virne/network/`) is a mature VNE simulation framework with:
- A **typed attribute system** (resource, extrema, position, latency, …)
- Multiple **topology models** (Waxman, Barabási–Albert, grid, …)
- **Distribution-driven generation** (uniform, exponential, normal, Poisson, …)
- **GML persistence** with full attribute metadata
- A **full event simulator** (`VirtualNetworkRequestSimulator`) with arrival/departure events

`generators_v2` bridges both worlds: it keeps 100 % backward-compatibility with `src/generators` while optionally unlocking virne's richer machinery.

---

## 2. Architecture Overview

```
src/generators_v2/
├── __init__.py              ← single public API surface
├── topology.py              ← TopologyType constants + build_topology()
├── substrate_generator.py   ← generate_substrate() + generate_substrate_virne()
├── vnr_generator.py         ← all VNR generators + load helpers
└── dataset_generator.py     ← DatasetGeneratorV2 + VirneDatasetGenerator
```

**Dependency flow:**

```
                ┌─────────────────────────────────┐
                │       generators_v2 API          │
                └────────────┬────────────┬────────┘
                             │            │
              ┌──────────────▼──┐    ┌────▼──────────────────────┐
              │  Lightweight     │    │  virne-native              │
              │  (nx.Graph +    │    │  (PhysicalNetwork,          │
              │   JSON output)  │    │   VirtualNetworkRequest     │
              │                 │    │   Simulator, GML, YAML)     │
              └──────────────┬──┘    └────┬──────────────────────┘
                             │            │
              ┌──────────────▼──┐    ┌────▼──────────────────────┐
              │  src/generators  │    │  virne-main/virne/network/ │
              │  (v1 codebase)  │    │                            │
              └─────────────────┘    └────────────────────────────┘
```

**Key design rules:**
1. `generate_substrate`, `generate_vnr*`, `DatasetGeneratorV2` are **always available** — no virne import required.
2. `generate_substrate_virne`, `generate_vnr_stream_virne`, `VirneDatasetGenerator` require virne on `sys.path` and will raise a helpful `ImportError` if absent.
3. All lightweight functions return the **same types** as their v1 equivalents: `nx.Graph` for networks, `list[nx.Graph]` for streams.

---

## 3. Module Map

| Symbol | Module | Returns | virne required? |
|---|---|---|---|
| `generate_substrate` | substrate_generator | `nx.Graph` | No |
| `generate_substrate_virne` | substrate_generator | `PhysicalNetwork` | **Yes** |
| `load_substrate_from_json` | substrate_generator | `nx.Graph` | No |
| `generate_vnr` | vnr_generator | `nx.Graph` | No |
| `generate_single_vnr` | vnr_generator | `nx.Graph` | No |
| `generate_vnr_stream` | vnr_generator | `list[nx.Graph]` | No |
| `generate_vnr_stream_v2` | vnr_generator | `list[nx.Graph]` | No |
| `generate_vnr_stream_virne` | vnr_generator | `VirtualNetworkRequestSimulator` | **Yes** |
| `load_vnr_stream_from_json` | vnr_generator | `list[dict]` | No |
| `sample_lifetime` | vnr_generator | `int` | No |
| `sample_inter_arrival` | vnr_generator | `int` | No |
| `sample_vnr_size` | vnr_generator | `int` | No |
| `DatasetGeneratorV2` | dataset_generator | class | No |
| `VirneDatasetGenerator` | dataset_generator | class | **Yes** |
| `TopologyType` | topology | class (constants) | No |
| `build_topology` | topology | `nx.Graph` | No |

---

## 4. Substrate Generators

### 4.1 `generate_substrate` (lightweight)

Generates a multi-domain substrate network as an `nx.Graph`.

```python
from src.generators_v2 import generate_substrate

G = generate_substrate(
    num_domains=4,
    num_nodes_total=80,
    p_intra=0.5,            # intra-domain edge probability
    p_inter=0.05,           # inter-domain edge probability
    cpu_range=(100, 300),
    bw_range=(1000, 3000),
    node_cost_range=(1, 10),
    inter_domain_bw_cost=(5, 15),
    # New optional attributes
    memory_range=(16, 64),  # adds memory / memory_total to each node
    gpu_range=(0, 4),       # adds gpu / gpu_total to each node
    latency_range=(1, 10),  # adds latency (ms) to each link
    ensure_domain_connectivity=True,  # each domain is internally connected
    seed=42,
    export_path="dataset/substrate.json",  # omit to skip file write
)
```

**Node attributes always present:** `cpu`, `cpu_total`, `domain`, `cost`

**Optional node attributes (when range is provided):** `memory`, `memory_total`, `gpu`, `gpu_total`

**Link attributes always present:** `bw`, `bw_total`, `bw_cost`

**Optional link attributes:** `latency`

> **Note on `ensure_domain_connectivity`:** When `True` (default), a Hamiltonian path is first
> added within each domain before applying random intra-domain edges. This guarantees each domain
> is internally connected even with a low `p_intra`.

### 4.2 `generate_substrate_virne` (full attribute system)

Generates a virne `PhysicalNetwork` using virne's configuration-driven attribute and topology system.

```python
import sys
sys.path.insert(0, 'd:/HUST_file/@research/@working/virne-main')

from src.generators_v2 import generate_substrate_virne

config = {
    "topology": {
        "type": "waxman",
        "num_nodes": 100,
        "wm_alpha": 0.5,
        "wm_beta": 0.2,
    },
    "node_attrs_setting": [
        {"name": "cpu", "owner": "node", "type": "resource",
         "generative": True, "distribution": "uniform",
         "dtype": "int", "low": 100, "high": 300},
    ],
    "link_attrs_setting": [
        {"name": "bw", "owner": "link", "type": "resource",
         "generative": True, "distribution": "uniform",
         "dtype": "int", "low": 1000, "high": 3000},
    ],
}

p_net = generate_substrate_virne(config, seed=42, save_dir="dataset/virne/p_net")
```

Available topology types: `waxman`, `random`, `barabasi_albert`, `path`, `star`, `grid_2d`.

### 4.3 `load_substrate_from_json`

```python
from src.generators_v2 import load_substrate_from_json
G = load_substrate_from_json("dataset/substrate.json")
print(G.nodes[0])  # {'cpu': 150, 'cpu_total': 150, 'domain': 0, 'cost': 4.2}
```

---

## 5. VNR Generators

### 5.1 `generate_vnr` / `generate_single_vnr`

Minimal single-VNR helpers, extended with optional attributes.

```python
from src.generators_v2 import generate_vnr, generate_single_vnr

G = generate_vnr(num_nodes=5, edge_prob=0.4, cpu_range=(1, 10), bw_range=(5, 20))

G = generate_single_vnr(
    num_nodes=5, edge_prob=0.4,
    cpu_range=(1, 10), bw_range=(5, 20),
    memory_range=(1, 8),
    latency_range=(1, 5),
)
```

### 5.2 `generate_vnr_stream` (v1 — fixed size)

Fixed node count, balanced round-robin domain assignment, exponential inter-arrivals.

```python
from src.generators_v2 import generate_vnr_stream

vnr_list = generate_vnr_stream(
    num_vnrs=200,
    num_nodes=6,           # fixed per-VNR size
    num_domains=4,
    edge_prob=0.5,
    cpu_range=(1, 10),
    bw_range=(5, 15),
    max_lifetime=50,
    avg_inter_arrival=5.0,
    memory_range=(1, 4),   # NEW optional
    latency_range=(1, 5),  # NEW optional
    export_mode="single",  # "none" | "single" | "multiple"
    export_path="stream.json",
    seed=42,
)
# Returns list[nx.Graph] with G.graph['id', 'arrival_time', 'lifetime']
```

### 5.3 `generate_vnr_stream_v2` (bursty / realistic)

Variable node count, heavy-tailed lifetimes, bursty arrivals, hot-domain affinity.

```python
from src.generators_v2 import generate_vnr_stream_v2

vnr_list = generate_vnr_stream_v2(
    num_vnrs=500,
    num_domains=4,
    edge_prob=0.5,
    min_vnodes=2, max_vnodes=10,    # bimodal size
    cpu_range=(5, 30),              # bimodal resources
    bw_range=(10, 50),
    max_lifetime=300,               # Pareto(2.5) × 10
    avg_inter_arrival=1.0,          # bursty
    memory_range=(1, 8),            # NEW optional
    gpu_range=(0, 2),               # NEW optional
    latency_range=(1, 5),           # NEW optional
    max_latency_range=(10, 50),     # NEW: per-VNR SLA constraint
    hot_domain_prob=0.6,            # NEW: domain affinity strength
    export_path="stream_v2.json",
    seed=42,
)
```

**Distribution summary:**

| Attribute | Distribution |
|---|---|
| Node count per VNR | Bimodal: 30 % large `[0.7·max, max]`, 70 % small `[min, 0.6·max]` |
| Lifetime | Pareto(`α=2.5`) × 10, clipped to `[5, max_lifetime]` |
| Inter-arrival | 20 % burst `[0,1]`, 80 % `Exp(1/avg)` |
| CPU per node | 20 % heavy `[0.7·max, max]`, 80 % light `[min, 0.6·max]` |
| BW per link | Same bimodal as CPU |
| Node domain | `hot_domain_prob` chance of hot domain, rest uniform |

### 5.4 `generate_vnr_stream_virne`

Returns a virne `VirtualNetworkRequestSimulator` with full event timeline.

```python
import sys
sys.path.insert(0, 'd:/HUST_file/@research/@working/virne-main')
from src.generators_v2 import generate_vnr_stream_virne

config = {
    "v_sim_setting": {
        "num_v_nets": 500,
        "topology": {"type": "random", "random_prob": 0.5},
        "v_net_size": {"distribution": "uniform", "dtype": "int", "low": 2, "high": 10},
        "lifetime": {"distribution": "exponential", "scale": 500, "dtype": "int"},
        "arrival_rate": {"distribution": "exponential", "scale": 100, "dtype": "int"},
        "node_attrs_setting": [
            {"name": "cpu", "owner": "node", "type": "resource",
             "generative": True, "distribution": "uniform",
             "dtype": "int", "low": 1, "high": 30},
        ],
        "link_attrs_setting": [
            {"name": "bw", "owner": "link", "type": "resource",
             "generative": True, "distribution": "uniform",
             "dtype": "int", "low": 1, "high": 30},
        ],
        "output": {
            "events_file_name": "events.yaml",
            "setting_file_name": "v_sim_setting.yaml",
        },
    }
}

v_sim = generate_vnr_stream_virne(config, seed=42, save_dir="dataset/virne/v_nets")

print(len(v_sim.v_nets))   # 500 VirtualNetwork objects
print(len(v_sim.events))   # 1000 events (arrival + departure each)

# Changeable mode (4-stage progressive escalation)
v_sim = generate_vnr_stream_virne(config, changeable=True)
```

**Changeable stages:**
1. CPU × 1.5, BW × 1.5
2. CPU × 2.0, BW × 2.0
3. VNR size × 1.5
4. VNR size × 2.0

### 5.5 `load_vnr_stream_from_json`

```python
from src.generators_v2 import load_vnr_stream_from_json
vnr_dicts = load_vnr_stream_from_json("dataset/vnr_stream.json")
```

---

## 6. Topology Types

```python
from src.generators_v2 import TopologyType
from src.generators_v2.topology import build_topology

G = build_topology(TopologyType.WAXMAN, num_nodes=50, wm_alpha=0.5, wm_beta=0.2)
G = build_topology(TopologyType.BARABASI_ALBERT, num_nodes=50, ba_m=3)
G = build_topology(TopologyType.GRID_2D, num_nodes=25, grid_m=5, grid_n=5)
```

| Type | Description | Parameters |
|---|---|---|
| `random` | Erdős–Rényi | `random_prob=0.5` |
| `multi_domain` | Alias for random (used with domain logic) | `random_prob=0.5` |
| `waxman` | Geographic Waxman model | `wm_alpha=0.5`, `wm_beta=0.2` |
| `path` | Linear chain | — |
| `star` | Hub-and-spoke | — |
| `grid_2d` | 2-D grid | `grid_m`, `grid_n` |
| `barabasi_albert` | Scale-free | `ba_m=2` |

All types guarantee a **connected** output graph.

---

## 7. Dataset Generators

### 7.1 `DatasetGeneratorV2` (extended)

```python
from src.generators_v2 import DatasetGeneratorV2

gen = DatasetGeneratorV2(base_dir="dataset")

# Backward-compatible experiments
gen.generate_fig6_dataset()   # vary virtual node count
gen.generate_fig7_dataset()   # vary domain count
gen.generate_fig8_dataset()   # acceptance rate over time
gen.generate_all()             # all three

# New experiments
gen.generate_custom_dataset(name="my_exp", num_domains=6, num_nodes_total=120, num_vnrs=800)
gen.generate_rl_training_dataset(seed=42)   # tuned for PPO / REINFORCE
gen.generate_stress_dataset(seed=77)        # high-density stress test
```

**Output directory structure:**

```
dataset/
├── fig6/
│   ├── metadata.json
│   ├── replica_0/ … replica_9/
│   │   ├── substrate.json
│   │   ├── vnr_2nodes.json
│   │   └── vnr_4nodes.json, …
├── fig7/
│   ├── metadata.json
│   ├── substrate_4domains.json
│   └── vnr_4domains.json, …
├── fig8/
│   ├── metadata.json
│   ├── substrate.json
│   └── vnr_stream.json
├── rl_training/
│   ├── metadata.json
│   ├── substrate.json
│   └── vnr_stream.json
└── stress_test/ …
```

### 7.2 `VirneDatasetGenerator` (virne-native)

```python
import sys; sys.path.insert(0, 'd:/HUST_file/@research/@working/virne-main')
from src.generators_v2 import VirneDatasetGenerator

gen = VirneDatasetGenerator(base_dir="dataset_virne")
p_net, v_sim = gen.generate(config, name="exp1", save=True)

# Load back
p_net, v_sim = VirneDatasetGenerator.load("dataset_virne/exp1")
```

---

## 8. Virne Integration Details

### 8.1 How virne's attribute system works

virne separates **topology generation** from **attribute generation**:

1. `BaseNetwork.generate_topology(type, num_nodes, ...)` — creates the graph structure.
2. `BaseNetwork.generate_attrs_data()` — iterates over `node_attrs_setting` and `link_attrs_setting`,
   drawing random values from the specified distribution and assigning them to every node/link.

### 8.2 `p_net_setting` config schema

```python
p_net_setting = {
    "topology": {
        "type": "waxman",         # topology model
        "num_nodes": 100,
        "wm_alpha": 0.5,
        "wm_beta": 0.2,
    },
    "node_attrs_setting": [
        {
            "name": "cpu",           # attribute name
            "owner": "node",         # "node" or "link"
            "type": "resource",      # attribute type
            "generative": True,      # draw random values
            "distribution": "uniform",
            "dtype": "int",
            "low": 100, "high": 300,
        },
    ],
    "link_attrs_setting": [
        {
            "name": "bw", "owner": "link", "type": "resource",
            "generative": True, "distribution": "uniform",
            "dtype": "int", "low": 1000, "high": 3000,
        },
    ],
}
```

**Attribute types:**

| `type` | `owner` | Description |
|---|---|---|
| `resource` | node or link | Renewable capacity (cpu, bw, memory, …) |
| `status` | node or link | Boolean on/off indicator |
| `extrema` | node or link | Tracks min/max of another attribute |
| `position` | node | Spatial location (x, y, radius) |
| `latency` | link | Propagation delay |

**Distribution options** (when `generative: True`):

| `distribution` | Required keys | Description |
|---|---|---|
| `uniform` | `low`, `high` | Uniform `[low, high)` |
| `normal` | `loc`, `scale` | Normal (Gaussian) |
| `exponential` | `scale` | Exponential with mean `scale` |
| `poisson` | `lam` | Poisson with rate `lam` |
| `customized` | `min`, `max` | Scaled uniform `[min, max]` |

### 8.3 `v_sim_setting` config schema

```python
v_sim_setting = {
    "num_v_nets": 500,
    "topology": {"type": "random", "random_prob": 0.5},
    "v_net_size": {
        "distribution": "uniform", "dtype": "int", "low": 2, "high": 10
    },
    "lifetime": {
        "distribution": "exponential", "scale": 500, "dtype": "int"
    },
    "arrival_rate": {
        "distribution": "exponential", "scale": 100, "dtype": "int"
    },
    "node_attrs_setting": [ ... ],   # same schema as p_net_setting
    "link_attrs_setting": [ ... ],
    "output": {
        "events_file_name": "events.yaml",
        "setting_file_name": "v_sim_setting.yaml",
    },
}
```

---

## 9. Migration from `src/generators`

| Old import | New import | Changes |
|---|---|---|
| `from src.generators.generator import generate_substrate` | `from src.generators_v2 import generate_substrate` | Added: `memory_range`, `gpu_range`, `latency_range`, `ensure_domain_connectivity` |
| `from src.generators.generator import generate_vnr_stream_v2` | `from src.generators_v2 import generate_vnr_stream_v2` | Added: `memory_range`, `gpu_range`, `latency_range`, `max_latency_range`, `hot_domain_prob` |
| `from src.generators.substrate_generator import generate_substrate` | `from src.generators_v2 import generate_substrate` | Same |
| `from src.generators.vnr_generator import generate_vnr_stream` | `from src.generators_v2 import generate_vnr_stream` | Added: optional attrs |
| `from src.generators.dataset_generator import DatasetGenerator` | `from src.generators_v2 import DatasetGeneratorV2` | Superset; all old methods work |

> **The old `src/generators` package is NOT modified.** Both packages coexist.

---

## 10. Usage Examples

### 10.1 Quick start — drop-in replacement

```python
from src.generators_v2 import generate_substrate, generate_vnr_stream_v2

G = generate_substrate(num_domains=4, num_nodes_total=100, seed=42)
vnrs = generate_vnr_stream_v2(num_vnrs=300, seed=42, export_path="vnrs.json")
```

### 10.2 Add memory & latency attributes

```python
from src.generators_v2 import generate_substrate, generate_vnr_stream_v2

G = generate_substrate(
    num_domains=4, num_nodes_total=100,
    cpu_range=(100, 300),
    memory_range=(32, 128),
    latency_range=(1.0, 10.0),
    seed=42,
    export_path="dataset/substrate_mem.json",
)

vnrs = generate_vnr_stream_v2(
    num_vnrs=300, num_domains=4,
    memory_range=(1, 16),
    max_latency_range=(20.0, 100.0),  # SLA constraint per VNR
    export_path="dataset/vnrs_mem.json",
    seed=42,
)
```

### 10.3 Use virne Waxman topology

```python
import sys
sys.path.insert(0, 'd:/HUST_file/@research/@working/virne-main')
from src.generators_v2 import generate_substrate_virne

p_net = generate_substrate_virne(
    config={
        "topology": {"type": "waxman", "num_nodes": 100,
                     "wm_alpha": 0.5, "wm_beta": 0.2},
        "node_attrs_setting": [
            {"name": "cpu", "owner": "node", "type": "resource",
             "generative": True, "distribution": "uniform",
             "dtype": "int", "low": 100, "high": 300},
        ],
        "link_attrs_setting": [
            {"name": "bw", "owner": "link", "type": "resource",
             "generative": True, "distribution": "uniform",
             "dtype": "int", "low": 1000, "high": 3000},
        ],
    },
    seed=42,
    save_dir="dataset/virne/p_net"
)
```

### 10.4 Train RL agent with pre-tuned dataset

```python
from src.generators_v2 import DatasetGeneratorV2, load_substrate_from_json, load_vnr_stream_from_json

gen = DatasetGeneratorV2(base_dir="dataset")
meta = gen.generate_rl_training_dataset(
    num_domains=4, num_nodes_total=80, num_vnrs=1000,
    vnr_cpu_range=(10, 80), vnr_bw_range=(50, 500),
    seed=42,
)

G_sub = load_substrate_from_json(meta["substrate"]["path"])
vnr_dicts = load_vnr_stream_from_json(meta["vnr_stream"]["path"])
print(f"Substrate: {G_sub.number_of_nodes()} nodes, {len(vnr_dicts)} VNRs")
```

### 10.5 Generate all experiment datasets via CLI

```bash
# Generate all standard experiments
python -m src.generators_v2.dataset_generator --experiments all --output-dir dataset

# Specific experiments
python -m src.generators_v2.dataset_generator --experiments fig6 rl stress

# Custom dataset
python -m src.generators_v2.dataset_generator --experiments custom --num-vnrs 500 --seed 99
```

### 10.6 Virne-native full dataset with save/load

```python
import sys; sys.path.insert(0, 'd:/HUST_file/@research/@working/virne-main')
from src.generators_v2 import VirneDatasetGenerator

config = {
    "p_net_setting": {
        "topology": {"type": "waxman", "num_nodes": 100},
        "node_attrs_setting": [
            {"name": "cpu", "owner": "node", "type": "resource",
             "generative": True, "distribution": "uniform",
             "dtype": "int", "low": 100, "high": 300},
        ],
        "link_attrs_setting": [
            {"name": "bw", "owner": "link", "type": "resource",
             "generative": True, "distribution": "uniform",
             "dtype": "int", "low": 1000, "high": 3000},
        ],
    },
    "v_sim_setting": {
        "num_v_nets": 500,
        "topology": {"type": "random", "random_prob": 0.5},
        "v_net_size": {"distribution": "uniform", "dtype": "int", "low": 2, "high": 10},
        "lifetime": {"distribution": "exponential", "scale": 500, "dtype": "int"},
        "arrival_rate": {"distribution": "exponential", "scale": 100, "dtype": "int"},
        "node_attrs_setting": [
            {"name": "cpu", "owner": "node", "type": "resource",
             "generative": True, "distribution": "uniform",
             "dtype": "int", "low": 1, "high": 30},
        ],
        "link_attrs_setting": [
            {"name": "bw", "owner": "link", "type": "resource",
             "generative": True, "distribution": "uniform",
             "dtype": "int", "low": 1, "high": 30},
        ],
        "output": {
            "events_file_name": "events.yaml",
            "setting_file_name": "v_sim_setting.yaml",
        },
    },
}

gen = VirneDatasetGenerator(base_dir="dataset_virne")
p_net, v_sim = gen.generate(config, name="waxman_exp", save=True)

# Reload later
p_net, v_sim = VirneDatasetGenerator.load("dataset_virne/waxman_exp")
```

### 10.7 Load existing datasets

```python
from src.generators_v2 import load_substrate_from_json, load_vnr_stream_from_json

G = load_substrate_from_json("dataset/fig8/substrate.json")
vnrs = load_vnr_stream_from_json("dataset/fig8/vnr_stream.json")
```

---

## 11. JSON File Formats

### Substrate JSON

```json
{
  "num_domains": 4,
  "nodes": [
    {
      "id": 0,
      "cpu": 150, "cpu_total": 150,
      "domain": 0,
      "cost": 4.2,
      "memory": 32, "memory_total": 32
    }
  ],
  "links": [
    {
      "u": 0, "v": 1,
      "bw": 2000, "bw_total": 2000,
      "bw_cost": 1.0,
      "latency": 3.5
    }
  ]
}
```

### VNR Stream JSON

```json
[
  {
    "id": 0,
    "arrival_time": 3,
    "lifetime": 87,
    "max_latency": 25.0,
    "nodes": [
      {"id": 0, "cpu": 12, "domain": 1, "memory": 4},
      {"id": 1, "cpu": 7,  "domain": 2, "memory": 2}
    ],
    "links": [
      {"u": 0, "v": 1, "bw": 25, "latency": 2.1}
    ]
  }
]
```

---

## 12. Design Decisions & Limitations

| Decision | Rationale |
|---|---|
| **virne not imported at module load time** | Avoids `ImportError` for users who don't have virne on sys.path. Every virne-backed function raises a clear error with instructions. |
| **generators_v2 does not import from generators** | Prevents circular imports; logic is re-implemented (with improvements) rather than delegated. |
| **Lightweight functions keep `list[nx.Graph]` return type** | Existing code iterates over graphs and reads `G.graph['id']` etc. Changing to custom objects would break compatibility. |
| **Optional attributes use `None` default** | Zero-overhead drop-in use; users who don't need memory/gpu/latency pay no cost. |
| **`ensure_domain_connectivity=True` default** | Original v1 had no guarantee of intra-domain connectivity. The new default is safer. |
| **`hot_domain_prob` as configurable knob** | Allows controlling how clustered VNR demand is, useful for locality-aware placement research. |

**Known limitations:**
- virne's `PhysicalNetwork` and `VirtualNetwork` require `omegaconf` and `numpy`; lightweight generators do not.
- GML format (virne) and JSON format (lightweight) are not interchangeable. Use the appropriate load function.
- `generate_substrate_virne` does not support multi-domain intra/inter-domain probability logic — use `generate_substrate` for that.
