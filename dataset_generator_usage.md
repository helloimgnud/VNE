# VNE Dataset Generator Usage Guide

This guide provides a comprehensive overview of how to generate customized datasets for your Virtual Network Embedding (VNE) experiments using the dataset generator script.

## 1. Introduction

The VNE framework uses a two-step process: **generation** and **evaluation**. Before running any algorithms, you must first construct your test environments. This prevents unwanted randomness during evaluations and acts as a cached benchmark so you can replicate results consistently across different models.

**Script entry point:** `src/scripts/generate_datasets.py`

---

## 2. Parameter Configurations

You can precisely control the scale and randomness of the network environment. Below are the arguments available:

### Core Controls
- `--experiments`: Dictates which dataset configurations to generate.
  - Choices: `fig6`, `fig7`, `fig8`, or `all`.
  - Default: `['all']`
- `--output-dir`: Where JSON networks will be stored locally. 
  - Default: `dataset` (generates `dataset/fig6`, etc.)
- `--force`: By default, the script asks for confirmation to prevent overwriting existing dataset configurations. Adding `--force` bypasses this prompt.

### Scale & Topologies
- `--num-replicas`: Creates `N` statistically identical but physically different topologies. Useful for acquiring statistically significant evaluations (default: `10`).
- `--substrate-nodes`: The size of the underlying substrate infrastructure. You can provide a fixed number (`100`) or a uniform-sampled range (`80,120`).
- `--num-vnrs`: The number of incoming Virtual Network Requests per VNR stream replica. Accepts a flat number (`200`) or bounds (`150,250`).
- `--vnode-range`: Discrete amounts of Virtual Nodes to configure your VNR streams. For instance, `--vnode-range "2,4,6,8"` creates 4 distinctive VNR Streams (one where VNRs only have 2 nodes, one with 4 nodes, etc.) for each simulation replica.

### Reproducibility
- `--base-seed`: Central randomness controller. Generating with identical seeds yield identical topologies and VNR metrics (default: `42`).

---

## 3. Practical Usage Examples

### Example A: The Default Generation
Generating the default benchmarks across the board.
```bash
python -m src.scripts.generate_datasets --experiments all
```
*Outcome: 10 fixed replicas of 100-node Substrates with 200-VNR streams for 2, 4, 6, and 8 virtual nodes.*

### Example B: Large Volatile Real-World Simulation
If you want varying topology sizes mimicking realistic network variance:
```bash
python -m src.scripts.generate_datasets \
  --experiments fig6 \
  --num-replicas 5 \
  --substrate-nodes "100,150" \
  --num-vnrs "200,300"
```
*Outcome: 5 Replicas where each replica's Substrate has between 100 and 150 nodes, and its VNR stream contains between 200 and 300 VNRs.*

### Example C: Micro-Benchmarking (Rapid Prototype)
If you just need a small consistent dataset to iteratively debug models.
```bash
python -m src.scripts.generate_datasets \
  --experiments fig6 \
  --num-replicas 1 \
  --substrate-nodes 50 \
  --num-vnrs 50 \
  --force 
```

---

## 4. How the Replicas Engine works

If `--num-replicas` is larger than 1, the script acts as a multi-seed dataset pipeline. For `fig6` with 3 replicas:

1. **Replica 0**: Generated under Seed = `base_seed + 0`. Substrate scale drawn from constraints.
2. **Replica 1**: Generated under Seed = `base_seed + 100`. Substrate scale drawn from constraints.
3. **Replica 2**: Generated under Seed = `base_seed + 200`. Substrate scale drawn from constraints.

This builds robust statistical validation mechanisms.

---

## 5. Next Steps

Once the generator prints `Datasets ready!`, the output JSONs (containing the complete undirected matrices) are locked to disk. You can now pass this exact configuration straight into the Experiment Runner:

```bash
# Evaluate meta-heuristics using all the generated dataset replicas
python -m src.scripts.run_experiments --experiments fig6 --algorithms hpso d_vine_sp --num-runs 3
```
