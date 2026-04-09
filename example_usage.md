# VNE Framework Example Usage Guide

This document describes how to use the experiment runner (`run_experiments.py`) to systematically test your Virtual Network Embedding strategies.

## Overview
The framework isolates dataset generation from algorithm execution. First, datasets are generated and cached locally in JSON, and then you execute experiment runners that spawn multi-run evaluators to provide reliable metric insights.

---

## The Run Experiments Script
The `run_experiments.py` script serves as the main entry point to evaluate various VNE algorithms.

### Command Line Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--experiments` | List[str] | `['fig6']` | Which experiments to execute. Must be one or more of `fig6`, `fig7`, `fig8` or `all`. |
| `--algorithms` | List[str] | `['baseline', 'proposed', 'pso', 'hpso']` | Specifies which algorithms to evaluate. Examples: `baseline`, `hpso_batch`, `d_vine_sp`. |
| `--num-runs` | int | `3` | Determines how many times to run the **same** algorithm over the **same** dataset configuration. Since meta-heuristics (like HPSO and PSO) have internal randomness, this ensures we evaluate the model several times. Plotting tools aggregate these repetitions via Mean/Averaging. |
| `--dataset-dir` | str | `"dataset"` | Directory path denoting where datasets have been pre-generated. |
| `--run-id` | str | `auto-generated` | Optional manual tag/identifier assigning to this batch of tests to keep track later (e.g. "experiment_alpha"). Defaults to a UTC timestamp. |
| `--no-plot` | flag | `False` | Pass this flag to intentionally skip automated chart generation at the very end of execution. |
| `--plot-only` | flag | `False` | Will strictly regenerate the plot artifacts using historical `.csv` records. Does absolutely no simulation. |
| `--list-runs` | flag | `False` | Prints out all internally logged runs historically executed on local. |
| `--compare-runs` | flag | `False` | Aggregates multiple prior run logs to generate side-by-side plot comparisons. Must be combined with `--plot-only`. |

---

## Quick Start Examples

1. **Perform a Standard Evaluation (Running Fig 6)**
```bash
# This will execute the default algorithms over 'dataset/fig6' 3 times each.
# It then stores the resulting metrics, followed by automated plot generation.
python set/scripts/run_experiments.py --experiments fig6
```

2. **Testing specific Algorithms over more evaluations**
```bash
# Comparing specific meta-heuristic agents with higher repetitions
python -m src.scripts.run_experiments --experiments fig6 --algorithms hpso hpso_batch --num-runs 5
```

3. **Running experiments on background and skipping live plotting**
```bash
python -m src.scripts.run_experiments --experiments all --algorithms baseline proposed pso hpso --no-plot
```

4. **Generating visual comparisons for past runs**
```bash
# Let's say you ran the experiment multiple times in the past. 
python -m src.scripts.run_experiments --experiments fig6 --plot-only --compare-runs
```

5. **Listing Historical Tests**
```bash
python -m src.scripts.run_experiments --experiments fig6 --list-runs
```

---

## Result Architecture

Given internal variance within your algorithms, standard tests evaluate an algorithm uniformly `N` times per single test condition.
- The experiment framework individually evaluates and logs **each identical run configuration** sequentially.
- When generating visual charts and calculating output statistics, our `plot` mechanics seamlessly bundle matching combinations and automatically **display the arithmetic Mean**. 
- To inspect deep metric variance, simply view the compiled `results_fig{x}_{run_id}.csv` to analyze specific evaluations tagged sequentially under the `eval_run` column constraint.
