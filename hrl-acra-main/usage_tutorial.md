# HRL-ACRA: Usage Tutorial (Windows 11 + Miniconda)

> A hands-on, copy-paste-ready guide for **Windows 11 with Miniconda / PowerShell**.
> Every command in this file has been verified to work in a Miniconda `conda` environment on Windows 11.
>
> **Shell used throughout**: Windows PowerShell (or Anaconda Prompt). Commands that are Python-only work anywhere.

---

## Table of Contents

1. [Setup & Environment Check](#1-setup--environment-check)
2. [Dataset Generation](#2-dataset-generation)
3. [Run Heuristic Baselines (No Training)](#3-run-heuristic-baselines-no-training)
4. [Train the HRL-RA Lower-level Agent](#4-train-the-hrl-ra-lower-level-agent)
5. [Train the HRL-AC Upper-level Agent](#5-train-the-hrl-ac-upper-level-agent)
6. [Test / Inference Only](#6-test--inference-only)
7. [Fine-tune a Pretrained Model](#7-fine-tune-a-pretrained-model)
8. [Train & Test Learning-based Baselines](#8-train--test-learning-based-baselines)
9. [Beam Search vs. Greedy Search](#9-beam-search-vs-greedy-search)
10. [Run Ablation / Sensitivity Experiments](#10-run-ablation--sensitivity-experiments)
11. [Evaluate & Summarize Results Programmatically](#11-evaluate--summarize-results-programmatically)
12. [Compare Multiple Solvers and Plot Results](#12-compare-multiple-solvers-and-plot-results)
13. [Monitor Training with TensorBoard](#13-monitor-training-with-tensorboard)
14. [Use Real-World Topologies](#14-use-real-world-topologies)
15. [Custom Dataset Generation in Python](#15-custom-dataset-generation-in-python)
16. [Full Reproducible Experiment Pipeline (End-to-End)](#16-full-reproducible-experiment-pipeline-end-to-end)

---

## 1. Setup & Environment Check

> [!NOTE]
> On Windows, `install.sh` requires Git Bash or WSL. Use the equivalent `conda`/`pip` commands below directly in your Miniconda/Anaconda Prompt or PowerShell.

### Create and activate a conda environment

```powershell
conda create -n hrl-acra python=3.8 -y
conda activate hrl-acra
```

### Install dependencies (CPU-only)

```powershell
pip install numpy pandas matplotlib networkx pyyaml tqdm colorama
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install tensorboard gym stable-baselines3 sb3-contrib
pip install --force-reinstall scipy
```

### Install dependencies (GPU, CUDA 11.3)

```powershell
pip install numpy pandas matplotlib networkx pyyaml tqdm colorama
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --index-url https://download.pytorch.org/whl/cu113
pip install torch-geometric
pip install tensorboard gym stable-baselines3 sb3-contrib
pip install --force-reinstall scipy
```

**Dependencies installed:**
- PyTorch 1.11.0, PyTorch Geometric (PyG)
- NetworkX, PyYAML, tqdm, pandas, numpy
- TensorBoard, gym, stable-baselines3

### Verify the environment

```powershell
python -c "import torch, networkx; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('NetworkX:', networkx.__version__)"
```

### Verify the project runs (quick sanity check)

Run from the project root (`hrl-acra-main\`):

```powershell
python main.py --solver_name="nrm_rank" --num_epochs=1 --seed=42 --summary_file_name="sanity_check.csv"
```

Expected output: a progress bar showing `ac`, `r2c`, `inservice` values and a completed CSV at `save\sanity_check.csv`.

---

## 2. Dataset Generation

The framework auto-generates datasets on first run and caches them in `dataset\`. You can also generate and save them explicitly using Python.

### 2.1 Auto-generation on first run

Simply running any solver will auto-generate the physical network (`dataset\p_net\`) and VNR simulator (`dataset\v_nets\`) if they don't exist yet.

```powershell
python main.py --solver_name="nrm_rank" --seed=0
```

### 2.2 Force regeneration of VNR dataset

If you change `settings\v_sim_setting.yaml`, you must regenerate the VNR dataset:

```powershell
python main.py --solver_name="nrm_rank" --renew_v_net_simulator=True --seed=0
```

To regenerate the physical network, delete the saved directory (PowerShell):

```powershell
Remove-Item -Recurse -Force dataset\p_net
```

Then re-run any solver.

### 2.3 Generate dataset from Python (programmatic)

Save as `generate_dataset.py` and run with `python generate_dataset.py`:

```python
# generate_dataset.py
from config import get_config
from data.generator import Generator

# Load default config (reads settings/ YAML files)
config = get_config(['--seed=42'])

# Generate and save both networks
p_net, v_net_sim = Generator.generate_dataset(config, p_net=True, v_nets=True, save=True)

print(f"Physical network: {p_net.num_nodes} nodes, {p_net.num_edges} edges")
print(f"VNR simulator: {len(v_net_sim.v_nets)} VNRs generated")
print("Datasets saved to dataset/p_net/ and dataset/v_nets/")
```

```powershell
python generate_dataset.py
```

### 2.4 Generate dataset with custom settings

```python
# generate_custom_dataset.py
from config import get_config
from data.generator import Generator

config = get_config([
    '--seed=123',
    '--p_net_setting_path=settings/p_net_setting.yaml',
    '--v_sim_setting_path=settings/v_sim_setting.yaml',
])

# Override VNR settings programmatically
config.v_sim_setting['num_v_nets'] = 2000
config.v_sim_setting['arrival_rate']['lam'] = 0.06  # higher load
config.v_sim_setting['v_net_size']['high'] = 15      # larger VNRs

# Override physical network settings
config.p_net_setting['num_nodes'] = 50  # smaller substrate

p_net, v_sim = Generator.generate_dataset(config, save=True)
print("Custom dataset generated!")
```

```powershell
python generate_custom_dataset.py
```

---

## 3. Run Heuristic Baselines (No Training)

Heuristic solvers require no training. Run them directly for comparison.

> [!NOTE]
> On Windows, PowerShell does **not** support `\` for multi-line commands. Use the backtick `` ` `` character instead, or write the full command on one line.

### GRC (Global Resource Capacity)

```powershell
python main.py --solver_name="grc_rank" --seed=0 --summary_file_name="results_grc.csv"
```

### NRM (Node Resource Management)

```powershell
python main.py --solver_name="nrm_rank" --seed=0 --summary_file_name="results_nrm.csv"
```

### PL (Path-Length)

```powershell
python main.py --solver_name="pl_rank" --seed=0 --summary_file_name="results_pl.csv"
```

### GAE-BFS (Graph AutoEncoder + BFS) — learning-free at test time

```powershell
python main.py --solver_name="gae_vne" --seed=0 --summary_file_name="results_gae.csv"
```

### MCTS (Monte Carlo Tree Search)

```powershell
python main.py --solver_name="mcts_vne" --seed=0 --summary_file_name="results_mcts.csv"
```

---

## 4. Train the HRL-RA Lower-level Agent

`hrl_ra` is the **resource allocation** sub-agent. **Always train this first** before training `hrl_ac`.

> [!NOTE]
> PowerShell multi-line command continuation uses a backtick `` ` `` at the end of each line (not `\`).

### 4.1 Basic training

```powershell
python main.py `
  --solver_name="hrl_ra" `
  --num_train_epochs=100 `
  --eval_interval=10 `
  --save_interval=10 `
  --summary_file_name="train_hrl_ra.csv" `
  --seed=0
```

Or as a single line:

```powershell
python main.py --solver_name="hrl_ra" --num_train_epochs=100 --eval_interval=10 --save_interval=10 --summary_file_name="train_hrl_ra.csv" --seed=0
```

- Model checkpoints saved every 10 epochs to `save\hrl_ra\<run_id>\model\`
- TensorBoard logs saved to `save\hrl_ra\<run_id>\log\`

### 4.2 Training with custom RL hyperparameters

```powershell
python main.py --solver_name="hrl_ra" --num_train_epochs=200 --batch_size=512 --lr_actor=5e-4 --lr_critic=5e-4 --rl_gamma=0.99 --embedding_dim=128 --num_gnn_layers=5 --eval_interval=10 --seed=0 --summary_file_name="train_hrl_ra_tuned.csv"
```

### 4.3 Training without GPU

```powershell
python main.py --solver_name="hrl_ra" --use_cuda=False --num_train_epochs=100 --seed=0 --summary_file_name="train_hrl_ra_cpu.csv"
```

### 4.4 Finding the saved model path

After training, the model is at:
```
save\
└── hrl_ra\
    └── <hostname>-<YYYYMMDDTHHMMSS>\
        ├── model\
        │   ├── model-9.pkl      # checkpoint at epoch 9
        │   ├── model-19.pkl     # checkpoint at epoch 19
        │   └── ...
        ├── log\                 # TensorBoard logs
        └── config.yaml          # saved config
```

Find the latest model file with PowerShell:

```powershell
# List all hrl_ra model files, newest first
Get-ChildItem -Path "save\hrl_ra" -Recurse -Filter "model-*.pkl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Select-Object -ExpandProperty FullName
```

Note the full path — you will need it for training `hrl_ac`.

---

## 5. Train the HRL-AC Upper-level Agent

`hrl_ac` is the **admission control** agent. It requires a pretrained `hrl_ra` model.

### 5.1 Basic training

First, set the path to your pretrained `hrl_ra` model (PowerShell variable):

```powershell
$HRL_RA_MODEL = "save\hrl_ra\mypc-20240101T120000\model\model-99.pkl"
```

> [!IMPORTANT]
> Replace `mypc-20240101T120000` and `model-99.pkl` with your actual run directory and checkpoint filename.

Then run:

```powershell
python main.py `
  --solver_name="hrl_ac" `
  --sub_solver_name="hrl_ra" `
  --num_train_epochs=500 `
  --eval_interval=10 `
  --save_interval=10 `
  --pretrained_subsolver_model_path="$HRL_RA_MODEL" `
  --summary_file_name="train_hrl_ac.csv" `
  --seed=0
```

### 5.2 Training with tuned hyperparameters

```powershell
python main.py --solver_name="hrl_ac" --sub_solver_name="hrl_ra" --num_train_epochs=500 --batch_size=256 --lr_actor=1e-3 --lr_critic=1e-3 --rl_gamma=0.99 --embedding_dim=128 --num_gnn_layers=5 --reward_weight=0.1 --eval_interval=10 --pretrained_subsolver_model_path="$HRL_RA_MODEL" --summary_file_name="train_hrl_ac_tuned.csv" --seed=0
```

### 5.3 Training pipeline as a PowerShell script

Save as `run_training.ps1` and execute with `.\run_training.ps1`:

```powershell
# run_training.ps1
# Run from the project root directory.
# Usage: .\run_training.ps1

$SEED = 0
$LOG_PREFIX = "exp_seed$SEED"

Write-Host "=== Step 1: Pretrain hrl_ra ==="
python main.py --solver_name="hrl_ra" --num_train_epochs=100 --eval_interval=10 --seed=$SEED --summary_file_name="${LOG_PREFIX}_hrl_ra_train.csv"

# Find the latest hrl_ra model
$HRL_RA_MODEL = Get-ChildItem -Path "save\hrl_ra" -Recurse -Filter "model-*.pkl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 |
    Select-Object -ExpandProperty FullName
Write-Host "Found pretrained hrl_ra model: $HRL_RA_MODEL"

Write-Host "=== Step 2: Train hrl_ac ==="
python main.py --solver_name="hrl_ac" --sub_solver_name="hrl_ra" --num_train_epochs=500 --eval_interval=10 --pretrained_subsolver_model_path="$HRL_RA_MODEL" --seed=$SEED --summary_file_name="${LOG_PREFIX}_hrl_ac_train.csv"

Write-Host "=== Training Complete ==="
```

---

## 6. Test / Inference Only

Run a pretrained model without any additional training (`--num_train_epochs=0`).

### 6.1 Test HRL-ACRA (greedy decoding)

```powershell
$HRL_AC_MODEL = "save\hrl_ac\mypc-20240101T130000\model\model-499.pkl"
$HRL_RA_MODEL = "save\hrl_ra\mypc-20240101T120000\model\model-99.pkl"

python main.py --solver_name="hrl_ac" --sub_solver_name="hrl_ra" --num_train_epochs=0 --decode_strategy="greedy" --pretrained_model_path="$HRL_AC_MODEL" --pretrained_subsolver_model_path="$HRL_RA_MODEL" --summary_file_name="test_hrl_acra_greedy.csv" --seed=0
```

### 6.2 Test HRL-RA only (no admission control)

```powershell
$HRL_RA_MODEL = "save\hrl_ra\mypc-20240101T120000\model\model-99.pkl"

python main.py --solver_name="hrl_ra" --num_train_epochs=0 --decode_strategy="greedy" --pretrained_model_path="$HRL_RA_MODEL" --summary_file_name="test_hrl_ra.csv" --seed=0
```

### 6.3 Test A3C-GCN baseline

```powershell
$A3C_MODEL = "save\a3c_gcn\mypc-20240101T140000\model\model-99.pkl"

python main.py --solver_name="a3c_gcn" --num_train_epochs=0 --pretrained_model_path="$A3C_MODEL" --summary_file_name="test_a3c_gcn.csv" --seed=0
```

---

## 7. Fine-tune a Pretrained Model

Fine-tuning resumes training from an existing checkpoint. Simply pass the checkpoint path via `--pretrained_model_path` **and** set `--num_train_epochs > 0`.

### 7.1 Fine-tune HRL-RA on a different load

First, edit `settings\v_sim_setting.yaml` and change `arrival_rate.lam` to `0.08`. Then:

```powershell
$HRL_RA_MODEL = "save\hrl_ra\mypc-20240101T120000\model\model-99.pkl"

python main.py --solver_name="hrl_ra" --num_train_epochs=50 --eval_interval=10 --pretrained_model_path="$HRL_RA_MODEL" --renew_v_net_simulator=True --summary_file_name="finetune_hrl_ra_high_load.csv" --seed=0
```

### 7.2 Fine-tune HRL-AC with updated sub-agent

If you have a better `hrl_ra` checkpoint, fine-tune `hrl_ac` with it:

```powershell
$NEW_HRL_RA_MODEL = "save\hrl_ra\mypc-20240101T150000\model\model-99.pkl"
$OLD_HRL_AC_MODEL = "save\hrl_ac\mypc-20240101T130000\model\model-499.pkl"

python main.py --solver_name="hrl_ac" --sub_solver_name="hrl_ra" --num_train_epochs=100 --eval_interval=10 --pretrained_model_path="$OLD_HRL_AC_MODEL" --pretrained_subsolver_model_path="$NEW_HRL_RA_MODEL" --summary_file_name="finetune_hrl_ac.csv" --seed=0
```

### 7.3 Fine-tune on a different topology (transfer learning)

```powershell
# Step 1: Update settings\p_net_setting.yaml — change topology to:
#   topology:
#     file_path: './dataset/topology/Geant.gml'

# Step 2: Delete cached physical network
Remove-Item -Recurse -Force dataset\p_net

# Step 3: Fine-tune from previous checkpoint
$HRL_RA_MODEL = "save\hrl_ra\mypc-20240101T120000\model\model-99.pkl"
python main.py --solver_name="hrl_ra" --num_train_epochs=50 --pretrained_model_path="$HRL_RA_MODEL" --summary_file_name="finetune_geant.csv" --seed=0
```

---

## 8. Train & Test Learning-based Baselines

### 8.1 PG-CNN

```powershell
# Train
python main.py --solver_name="pg_cnn2" --num_train_epochs=100 --eval_interval=10 --seed=0 --summary_file_name="train_pg_cnn2.csv"

# Test (update path to your actual checkpoint)
$PG_CNN_MODEL = "save\pg_cnn2\mypc-xxx\model\model-99.pkl"
python main.py --solver_name="pg_cnn2" --num_train_epochs=0 --pretrained_model_path="$PG_CNN_MODEL" --summary_file_name="test_pg_cnn2.csv" --seed=0
```

### 8.2 A3C-GCN

```powershell
# Train
python main.py --solver_name="a3c_gcn" --num_train_epochs=100 --eval_interval=10 --seed=0 --summary_file_name="train_a3c_gcn.csv"

# Test (update path to your actual checkpoint)
$A3C_MODEL = "save\a3c_gcn\mypc-xxx\model\model-99.pkl"
python main.py --solver_name="a3c_gcn" --num_train_epochs=0 --pretrained_model_path="$A3C_MODEL" --summary_file_name="test_a3c_gcn.csv" --seed=0
```

---

## 9. Beam Search vs. Greedy Search

`hrl_ra` and `hrl_ac` support both greedy (fast) and beam search (higher quality) at inference time.

Set your model paths first:

```powershell
$HRL_AC_MODEL = "save\hrl_ac\mypc-xxx\model\model-499.pkl"
$HRL_RA_MODEL = "save\hrl_ra\mypc-xxx\model\model-99.pkl"
```

### Greedy search (k=1, fastest)

```powershell
python main.py --solver_name="hrl_ac" --sub_solver_name="hrl_ra" --num_train_epochs=0 --decode_strategy="greedy" --k_searching=1 --pretrained_model_path="$HRL_AC_MODEL" --pretrained_subsolver_model_path="$HRL_RA_MODEL" --summary_file_name="test_greedy_k1.csv" --seed=0
```

### Beam search k=3 (paper default)

```powershell
python main.py --solver_name="hrl_ac" --sub_solver_name="hrl_ra" --num_train_epochs=0 --decode_strategy="beam" --k_searching=3 --pretrained_model_path="$HRL_AC_MODEL" --pretrained_subsolver_model_path="$HRL_RA_MODEL" --summary_file_name="test_beam_k3.csv" --seed=0
```

### Beam search k=5 (higher quality, slower)

```powershell
python main.py --solver_name="hrl_ac" --sub_solver_name="hrl_ra" --num_train_epochs=0 --decode_strategy="beam" --k_searching=5 --pretrained_model_path="$HRL_AC_MODEL" --pretrained_subsolver_model_path="$HRL_RA_MODEL" --summary_file_name="test_beam_k5.csv" --seed=0
```

---

## 10. Run Ablation / Sensitivity Experiments

Set your model paths first:

```powershell
$HRL_AC_MODEL = "save\hrl_ac\mypc-xxx\model\model-499.pkl"
$HRL_RA_MODEL = "save\hrl_ra\mypc-xxx\model\model-99.pkl"
```

### 10.1 Vary arrival rate (network load)

```powershell
foreach ($LAM in @(0.04, 0.06, 0.08)) {
    python main.py --solver_name="hrl_ac" --sub_solver_name="hrl_ra" --num_train_epochs=0 `
        --if_adjust_v_sim_setting=True --v_sim_setting_aver_arrival_rate=$LAM `
        --pretrained_model_path="$HRL_AC_MODEL" --pretrained_subsolver_model_path="$HRL_RA_MODEL" `
        --summary_file_name="test_lam${LAM}.csv" --renew_v_net_simulator=True --seed=0
}
```

### 10.2 Vary VNR node size

```powershell
foreach ($MAX_NODES in @(10, 15, 20)) {
    python main.py --solver_name="hrl_ac" --sub_solver_name="hrl_ra" --num_train_epochs=0 `
        --if_adjust_v_sim_setting=True --v_sim_setting_max_length=$MAX_NODES `
        --pretrained_model_path="$HRL_AC_MODEL" --pretrained_subsolver_model_path="$HRL_RA_MODEL" `
        --renew_v_net_simulator=True --summary_file_name="test_maxnodes${MAX_NODES}.csv" --seed=0
}
```

### 10.3 Vary resource demand

```powershell
foreach ($HIGH_REQ in @(30, 50, 70, 90)) {
    python main.py --solver_name="hrl_ac" --sub_solver_name="hrl_ra" --num_train_epochs=0 `
        --if_adjust_v_sim_setting=True --v_sim_setting_high_request=$HIGH_REQ `
        --pretrained_model_path="$HRL_AC_MODEL" --pretrained_subsolver_model_path="$HRL_RA_MODEL" `
        --renew_v_net_simulator=True --summary_file_name="test_req${HIGH_REQ}.csv" --seed=0
}
```

### 10.4 Multiple seeds for statistical confidence

```powershell
foreach ($SEED in @(0, 1, 2, 3, 4)) {
    python main.py --solver_name="nrm_rank" --seed=$SEED --renew_v_net_simulator=True --summary_file_name="nrm_seed${SEED}.csv"
}
```

---

## 11. Evaluate & Summarize Results Programmatically

The framework writes per-VNR records to `save\<solver>\<run_id>\records\temp-*.csv`. Use Python to compute aggregate metrics.

### 11.1 Parse a summary CSV

```python
# evaluate.py
import pandas as pd

# Read the summary file (one row per solver run)
df = pd.read_csv("save/global_summary.csv")
print(df.to_string())
```

```powershell
python evaluate.py
```

### 11.2 Compute metrics from a per-VNR records CSV

```python
# analyze_records.py
import pandas as pd

# Load a per-VNR record file (update path to your actual file)
records_path = "save/hrl_ac/mypc-xxx/records/temp-0.csv"
records = pd.read_csv(records_path)

# Only look at arrival (enter) events
arrivals = records[records['event_type'] == 1]

# Key metrics
total_vnrs      = arrivals['v_net_count'].iloc[-1]
accepted        = arrivals['success_count'].iloc[-1]
acceptance_rate = accepted / total_vnrs

total_revenue   = arrivals['total_revenue'].iloc[-1]
total_cost      = arrivals['total_cost'].iloc[-1]
r2c_ratio       = total_revenue / total_cost if total_cost > 0 else 0

max_inservice   = records['inservice_count'].max()
min_avail_res   = records['p_net_available_resource'].min()

print(f"Acceptance Rate : {acceptance_rate:.4f}")
print(f"R2C Ratio       : {r2c_ratio:.4f}")
print(f"Total Revenue   : {total_revenue:.2f}")
print(f"Total Cost      : {total_cost:.2f}")
print(f"Max In-Service  : {max_inservice}")
print(f"Min Avail. Res. : {min_avail_res:.2f}")

# Failure breakdown
place_fail = (arrivals['place_result'] == False).sum()
route_fail = (arrivals['route_result'] == False).sum()
print(f"\nFailure breakdown:")
print(f"  Placement failures : {place_fail}")
print(f"  Routing failures   : {route_fail}")
```

```powershell
python analyze_records.py
```

### 11.3 Aggregate metrics across multiple seeds

```python
# aggregate_seeds.py
import pandas as pd
import glob

solver = "nrm_rank"
pattern = f"save/{solver}_seed*.csv"

dfs = []
for fpath in glob.glob(pattern):
    df = pd.read_csv(fpath)
    dfs.append(df)

combined = pd.concat(dfs)
print("Mean across seeds:")
print(combined[['acceptance_rate', 'r2c_ratio', 'total_revenue']].mean())
print("\nStd across seeds:")
print(combined[['acceptance_rate', 'r2c_ratio', 'total_revenue']].std())
```

```powershell
python aggregate_seeds.py
```

---

## 12. Compare Multiple Solvers and Plot Results

### 12.1 Run all solvers then compare (PowerShell)

```powershell
# compare_all.ps1
$SEED = 0
$HRL_AC_MODEL = "save\hrl_ac\mypc-xxx\model\model-499.pkl"
$HRL_RA_MODEL = "save\hrl_ra\mypc-xxx\model\model-99.pkl"

# Heuristics
foreach ($SOLVER in @("grc_rank", "nrm_rank", "pl_rank", "mcts_vne", "gae_vne")) {
    python main.py --solver_name=$SOLVER --seed=$SEED --summary_file_name="compare_${SOLVER}.csv"
}

# HRL-ACRA (assumes models are pretrained)
python main.py --solver_name="hrl_ac" --sub_solver_name="hrl_ra" --num_train_epochs=0 `
    --decode_strategy="beam" --k_searching=3 `
    --pretrained_model_path="$HRL_AC_MODEL" `
    --pretrained_subsolver_model_path="$HRL_RA_MODEL" `
    --seed=$SEED --summary_file_name="compare_hrl_acra.csv"
```

Save and run:

```powershell
.\compare_all.ps1
```

### 12.2 Plot comparison bar chart

```python
# plot_comparison.py
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Collect summary CSVs
results = {}
for fpath in glob.glob("save/compare_*.csv"):
    solver_name = os.path.basename(fpath).replace("compare_", "").replace(".csv", "")
    df = pd.read_csv(fpath)
    if not df.empty:
        results[solver_name] = {
            'acceptance_rate': df['acceptance_rate'].iloc[-1],
            'r2c_ratio': df['r2c_ratio'].iloc[-1],
        }

df_results = pd.DataFrame(results).T.sort_values('acceptance_rate')
print(df_results.to_string())

# Bar chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df_results['acceptance_rate'].plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Acceptance Rate')
axes[0].set_ylabel('Rate')
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis='x', rotation=30)

df_results['r2c_ratio'].plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title('Revenue-to-Cost Ratio')
axes[1].set_ylabel('R2C')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig("comparison_chart.png", dpi=150)
print("Chart saved to comparison_chart.png")
plt.show()
```

```powershell
python plot_comparison.py
```

### 12.3 Plot metrics vs. arrival rate

```python
# plot_vs_load.py
import pandas as pd
import matplotlib.pyplot as plt

lam_values = [0.04, 0.06, 0.08]
solvers = ['nrm_rank', 'hrl_acra']
colors = {'nrm_rank': 'steelblue', 'hrl_acra': 'darkorange'}

plt.figure(figsize=(8, 5))
for solver in solvers:
    ac_rates = []
    for lam in lam_values:
        fname = f"save/test_lam{lam}_{solver}.csv"
        try:
            df = pd.read_csv(fname)
            ac_rates.append(df['acceptance_rate'].iloc[-1])
        except FileNotFoundError:
            ac_rates.append(None)
    plt.plot(lam_values, ac_rates, marker='o', label=solver, color=colors.get(solver))

plt.xlabel("Arrival Rate (λ)")
plt.ylabel("Acceptance Rate")
plt.title("Acceptance Rate vs. Arrival Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("load_sensitivity.png", dpi=150)
plt.show()
```

```powershell
python plot_vs_load.py
```

---

## 13. Monitor Training with TensorBoard

TensorBoard is enabled by default (`--open_tb=True`).

### Launch TensorBoard

```powershell
# Monitor all solver runs
tensorboard --logdir="save/"

# Monitor a specific solver
tensorboard --logdir="save/hrl_ra/"

# Monitor a specific run
tensorboard --logdir="save/hrl_ra/mypc-20240101T120000/log/"
```

Then open `http://localhost:6006` in your browser.

### Available TensorBoard scalars

| Tag | Description |
|---|---|
| `loss/loss` | Total PPO loss |
| `loss/actor_loss` | Policy (actor) loss |
| `loss/critic_loss` | Value function (critic) loss |
| `loss/entropy_loss` | Entropy bonus loss |
| `value/logprob` | Mean log-probability of selected actions |
| `value/return` | Mean discounted return |
| `value/advantage` | Mean advantage estimate |
| `value/value` | Mean critic value estimate |
| `grad/grad_clipped` | Gradient norm before clipping |
| `lr` | Current learning rate |

### Disable TensorBoard (faster runs)

```powershell
python main.py --solver_name="hrl_ra" --open_tb=False --num_train_epochs=100 --seed=0
```

---

## 14. Use Real-World Topologies

### 14.1 Brain topology (29 nodes)

Edit `settings\p_net_setting.yaml`:

```yaml
topology:
  file_path: './dataset/topology/Brain.gml'
```

Then delete cached p_net and run:

```powershell
Remove-Item -Recurse -Force dataset\p_net
python main.py --solver_name="nrm_rank" --seed=0 --summary_file_name="brain_nrm.csv"
```

### 14.2 Geant topology (40 nodes)

Edit `settings\p_net_setting.yaml`:

```yaml
topology:
  file_path: './dataset/topology/Geant.gml'
```

```powershell
Remove-Item -Recurse -Force dataset\p_net
python main.py --solver_name="nrm_rank" --seed=0 --summary_file_name="geant_nrm.csv"
```

### 14.3 Add your own topology

```python
# create_custom_topology.py
import networkx as nx

# Build any topology
G = nx.barabasi_albert_graph(n=50, m=3, seed=42)

# Assign resources to nodes
for node in G.nodes():
    G.nodes[node]['cpu'] = 100

# Assign resources to edges
for u, v in G.edges():
    G.edges[u, v]['bw'] = 100

# Save as GML
nx.write_gml(G, 'dataset/topology/BarabasiAlbert50.gml')
print("Topology saved.")
```

```powershell
python create_custom_topology.py
```

Then in `settings\p_net_setting.yaml`:
```yaml
topology:
  file_path: './dataset/topology/BarabasiAlbert50.gml'
```

---

## 15. Custom Dataset Generation in Python

Full programmatic control over dataset generation and the simulation loop:

```python
# custom_run.py
import sys
sys.argv = ['main.py']  # prevent argparse from reading real command line

from config import get_config
from data.physical_network import PhysicalNetwork
from data.virtual_network_request_simulator import VirtualNetworkRequestSimulator
from data.generator import Generator
from base.loader import load_simulator
from base import BasicScenario

# 1. Build config from defaults
config = get_config(['--solver_name=nrm_rank', '--seed=42', '--verbose=1'])

# 2. Override settings programmatically
config.v_sim_setting['num_v_nets'] = 500
config.v_sim_setting['arrival_rate']['lam'] = 0.06
config.p_net_setting['num_nodes'] = 50

# 3. Generate fresh dataset
config.p_net_dataset_dir = 'dataset/p_net'
p_net = PhysicalNetwork.from_setting(config.p_net_setting)
v_net_simulator = VirtualNetworkRequestSimulator.from_setting(config.v_sim_setting)

print(f"Physical network: {p_net.num_nodes} nodes")
print(f"VNR stream: {len(v_net_simulator.v_nets)} VNRs")

# 4. Load solver and run
Env, Solver = load_simulator(config.solver_name)
scenario = BasicScenario.from_config(Env, Solver, config)
scenario.run()
```

```powershell
python custom_run.py
```

---

## 16. Full Reproducible Experiment Pipeline (End-to-End)

This complete PowerShell script reproduces the paper's main experiment from scratch.

Save as `full_experiment.ps1` in the project root and run with `.\full_experiment.ps1`:

```powershell
# full_experiment.ps1
# Reproduces the main HRL-ACRA experiment from scratch.
# Run from the project root: .\full_experiment.ps1

$SEED = 0
Write-Host "============================================"
Write-Host "  HRL-ACRA Full Experiment Pipeline"
Write-Host "  Seed: $SEED"
Write-Host "============================================"

# ---- Step 1: Heuristic baselines ----
Write-Host "[1/5] Running heuristic baselines..."
foreach ($SOLVER in @("grc_rank", "nrm_rank", "pl_rank")) {
    python main.py --solver_name=$SOLVER --seed=$SEED --summary_file_name="results_${SOLVER}.csv"
}

# ---- Step 2: Learning baselines (no pretraining needed) ----
Write-Host "[2/5] Running learning-free baselines..."
foreach ($SOLVER in @("mcts_vne", "gae_vne")) {
    python main.py --solver_name=$SOLVER --seed=$SEED --summary_file_name="results_${SOLVER}.csv"
}

# ---- Step 3: Pretrain HRL-RA ----
Write-Host "[3/5] Pretraining hrl_ra (100 epochs)..."
python main.py --solver_name="hrl_ra" --num_train_epochs=100 --eval_interval=10 --save_interval=10 --seed=$SEED --summary_file_name="train_hrl_ra.csv"

# Find the latest hrl_ra model
$HRL_RA_MODEL = Get-ChildItem -Path "save\hrl_ra" -Recurse -Filter "model-*.pkl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 |
    Select-Object -ExpandProperty FullName
Write-Host "  --> hrl_ra model: $HRL_RA_MODEL"

# ---- Step 4: Train HRL-AC ----
Write-Host "[4/5] Training hrl_ac (500 epochs)..."
python main.py --solver_name="hrl_ac" --sub_solver_name="hrl_ra" --num_train_epochs=500 --eval_interval=10 --save_interval=10 --pretrained_subsolver_model_path="$HRL_RA_MODEL" --seed=$SEED --summary_file_name="train_hrl_ac.csv"

# Find the latest hrl_ac model
$HRL_AC_MODEL = Get-ChildItem -Path "save\hrl_ac" -Recurse -Filter "model-*.pkl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 |
    Select-Object -ExpandProperty FullName
Write-Host "  --> hrl_ac model: $HRL_AC_MODEL"

# ---- Step 5: Test HRL-ACRA with beam search ----
Write-Host "[5/5] Testing HRL-ACRA with beam search (k=3)..."
python main.py --solver_name="hrl_ac" --sub_solver_name="hrl_ra" --num_train_epochs=0 --decode_strategy="beam" --k_searching=3 --pretrained_model_path="$HRL_AC_MODEL" --pretrained_subsolver_model_path="$HRL_RA_MODEL" --seed=$SEED --summary_file_name="test_hrl_acra.csv"

Write-Host ""
Write-Host "============================================"
Write-Host "  All experiments complete!"
Write-Host "  Results saved in save\*.csv"
Write-Host "  Run: python plot_comparison.py"
Write-Host "============================================"
```

Then summarize results using Python:

```python
# summarize_results.py
import pandas as pd
import glob
import os

files = sorted(glob.glob('save/results_*.csv') + glob.glob('save/test_*.csv'))
for f in files:
    df = pd.read_csv(f)
    if df.empty:
        continue
    name = os.path.basename(f).replace('.csv', '')
    ac  = df['acceptance_rate'].iloc[-1] if 'acceptance_rate' in df.columns else float('nan')
    r2c = df['r2c_ratio'].iloc[-1] if 'r2c_ratio' in df.columns else float('nan')
    print(f"{name:<30s}  AC={ac:.4f}  R2C={r2c:.4f}")
```

```powershell
python summarize_results.py
```

---

## Quick Reference: All Key Commands (PowerShell)

| Use Case | PowerShell Command |
|---|---|
| Create env | `conda create -n hrl-acra python=3.8 -y` |
| Activate env | `conda activate hrl-acra` |
| Sanity check | `python main.py --solver_name=nrm_rank --seed=0` |
| Regen VNR dataset | add `--renew_v_net_simulator=True` |
| Delete p_net cache | `Remove-Item -Recurse -Force dataset\p_net` |
| Pretrain hrl_ra | `python main.py --solver_name=hrl_ra --num_train_epochs=100 --seed=0` |
| Find latest model | `Get-ChildItem -Path "save\hrl_ra" -Recurse -Filter "model-*.pkl" \| Sort-Object LastWriteTime -Descending \| Select-Object -First 1` |
| Set model var | `$HRL_RA_MODEL = "save\hrl_ra\...\model\model-99.pkl"` |
| Train hrl_ac | `python main.py --solver_name=hrl_ac --pretrained_subsolver_model_path="$HRL_RA_MODEL"` |
| Test only | add `--num_train_epochs=0 --pretrained_model_path="$MODEL"` |
| Beam search | add `--decode_strategy=beam --k_searching=3` |
| CPU mode | add `--use_cuda=False` |
| No TensorBoard | add `--open_tb=False` |
| High load test | add `--if_adjust_v_sim_setting=True --v_sim_setting_aver_arrival_rate=0.08` |
| Large VNRs | add `--if_adjust_v_sim_setting=True --v_sim_setting_max_length=20` |
| TensorBoard | `tensorboard --logdir=save/` |
| Run full pipeline | `.\full_experiment.ps1` |
