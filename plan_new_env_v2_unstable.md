# PPO v2 Training Pipeline — Revised Plan

## Overview & Motivation

The problem requires a PPO agent to learn how to **prioritize the mapping order of VNRs** for the underlying HPSO solver.
The v2 pipeline ensures that training behavior exactly matches inference behavior.

### Inference Flow (ground truth that training must match)

```
Stream of N VNRs → split into B time-window batches, each with m VNRs
                              ↓
At the start of each time window:
  1. Process leave events: release resources of VNRs whose lifetime has expired
  2. Collect m VNRs in the current window → vnr_queue
  3. PPO agent selects mapping order (action = index of next VNR to map)
  4. HPSO maps each VNR in the order chosen by the agent
  5. Substrate is consumed (if accepted) or unchanged (if failed)
  6. Advance to the next time window
```

### Training Alignment

| Dimension | Training (v2) | Inference |
|---|---|---|
| Substrate source | Fixed dataset | Live substrate |
| VNR source | From pre-generated dataset | Real time-window |
| VNR expiry | Processed at the start of each window (same as inference) | Processed at the start of each window |
| Episode boundary | **1 epoch = 1 episode = all N VNRs** | Continuous stream |
| Substrate reset | Reset to fresh **after each epoch** | No reset (continuous) |
| Action | Index of next VNR to map | Index of next VNR |

---

## Episode and Trajectory Definitions

### Episode = 1 pass through all N VNRs on a single substrate

```
Episode k:
  substrate_k = copy(substrate_original)   ← fresh each epoch
  t=0: Process leave events for window_0
       vnr_queue = [VNR_0, VNR_1, ..., VNR_{m-1}]
       step_0: action=i → HPSO(substrate, VNR_i) → reward_0 → substrate updated
       step_1: action=j → HPSO(substrate, VNR_j) → reward_1 → substrate updated
       ...
       (vnr_queue for window_0 exhausted)
  t=1: Process leave events for window_1
       vnr_queue = [VNR_m, ..., VNR_{2m-1}]
       step_m: action=... → ...
       ...
  ...
  t=B: All N VNRs processed → done=True → Episode ends

Episode k+1:
  substrate_{k+1} = copy(substrate_original)  ← reset to fresh
  (repeats with the same dataset)
```

### Step = 1 mapping decision for 1 VNR

```
State  s_t = (current_substrate, remaining_vnrs_in_window)
Action a_t = index i ∈ {0, ..., |remaining|-1}
Reward r_t = f(HPSO result)
Next   s_{t+1} = (substrate_after_mapping, remaining \ {VNR_i})
```

When `remaining` is exhausted (window ends), **there is no done signal** — the environment automatically
advances to the next window (processes leave events and loads new VNRs). The episode only signals
`done=True` when all N VNRs across all windows have been processed.

### Trajectory

A trajectory = the full sequence of steps within one episode:
- Length: `N steps` (one step per VNR)
- The buffer accumulates **all N transitions** before each update
- GAE is computed over the full trajectory after the episode ends

---

## Part 1 — Data Mode (Selection Flag)

### 1.1 Two Completely Separate Modes

```python
class DataMode(str, enum.Enum):
    PREGENERATED = "pregenerated"   # Use an existing dataset file
```

**Mode A — Pre-generated dataset** (the only supported mode):
```
dataset/rl_training/
├── substrate.json
└── vnr_stream.json      ← all N VNRs with arrival_time and lifetime
```
- **One run = one single dataset** (1 substrate + 1 VNR stream)
- To train on multiple datasets → re-run with a different `--train-dir`
- Loaded once at startup; reused each epoch (substrate resets to fresh, VNR stream stays fixed)
- To increase diversity: run multiple times with different seeds/datasets

### 1.2 VNR Stream Format

Each VNR in the stream must contain:
```json
{
  "id": 42,
  "arrival_time": 150.3,   // Arrival time (float)
  "lifetime": 87.0,         // Duration (float)
  "nodes": [...],
  "links": [...]
}
```

The window at time `t` contains all VNRs with `arrival_time ∈ [t, t + window_size)`.
A VNR expires when `arrival_time + lifetime < current_window_start`.

---

## Part 2 — `environment_v2.py`

### 2.1 Class Structure

```python
class VNEEnvironmentV2(gymnasium.Env):
    """
    PPO environment for VNR ordering.

    Episode = 1 pass through all N VNRs on a single substrate.
    Substrate resets to fresh after each episode (epoch).
    Leave events are processed at the start of each time window (same as inference).

    Data modes:
      - PREGENERATED: load from dataset_paths (list of nx.Graph)
    """
```

### 2.2 `__init__` — Initialization

```python
def __init__(
    self,
    # --- Data mode ---
    data_mode: DataMode = DataMode.PREGENERATED,

    # Pre-generated dataset — one dataset per run
    substrate_path: Optional[str] = None,   # path to substrate.json
    vnr_path: Optional[str] = None,         # path to vnr_stream.json
    # To train on a different dataset → re-run the script with a different path

    # Simulation parameters
    window_size: int = 50,
    max_queue_delay: int = 100,   # VNR is dropped if it waits too long in the queue

    # HPSO parameters
    hpso_params: Optional[dict] = None,
):
```

**Validation inside `__init__`:**
```python
assert data_mode == DataMode.PREGENERATED, "Only PREGENERATED mode is supported"
assert substrate_path is not None and vnr_path is not None, \
    "substrate_path and vnr_path are required"
# Loaded once at startup
self._substrate_original = load_substrate_from_json(substrate_path)
self._vnr_stream_raw = load_vnr_stream_from_json(vnr_path)
# Each epoch's reset() will copy substrate_original and reuse vnr_stream_raw
```

### 2.3 `reset()` — Start a New Episode

```python
def reset(self, seed=None, options=None):
    """
    Starts a new episode = fresh substrate + full VNR stream.

    MUST NOT be called between windows within the same episode.
    Should only be called once the previous episode has reached done=True.

    PREGENERATED mode:
      - substrate: fresh copy from self._substrate_original (loaded once in __init__)
      - vnr_stream: reuses self._vnr_stream_raw (immutable, unchanged each epoch)
      - Each training epoch iterates over the SAME dataset; diversity comes from stochastic HPSO

    Flow:
      1. Load (substrate, vnr_stream) according to data_mode
      2. substrate_working = copy(substrate_original)  ← fresh copy
      3. Partition vnr_stream into windows by arrival_time
      4. active_embeddings = []  ← no VNRs currently active
      5. Initialize window_idx = 0
      6. Process first window: expire + load vnr_queue
      7. Return obs
    """
    super().reset(seed=seed)

    # --- Load data ---
    # Substrate: fresh copy each epoch; VNR stream: immutable
    substrate_raw = self._substrate_original
    vnr_stream = self._vnr_stream_raw

    # --- Setup episode state ---
    self.substrate = copy_substrate(substrate_raw)
    self.substrate_original = substrate_raw  # Kept for utilisation calculations
    self.active_embeddings = []  # [(vnr, mapping, link_paths, expiry_time)]
    self.episode_accepted = []
    self.episode_rejected = []
    self.episode_accepted_costs = []

    # --- Partition into windows ---
    # Window i contains VNRs with arrival_time ∈ [i*W, (i+1)*W)
    self._all_windows = self._partition_into_windows(vnr_stream)
    self.window_idx = 0
    self.total_windows = len(self._all_windows)

    # --- Initialize first window ---
    self._advance_to_window(self.window_idx)
    # _advance_to_window:
    #   1. Expire VNRs whose lifetime has ended (start of window)
    #   2. Load VNRs for this window into self.vnr_queue
    #   3. Drop VNRs exceeding max_queue_delay (not applicable at window 0 — no old queue)

    return self._get_obs(), {}
```

### 2.4 `_partition_into_windows()` — Partition VNRs by Time Window

```python
def _partition_into_windows(self, vnr_stream: list) -> list[list]:
    """
    Partitions the VNR stream into time windows.

    Returns list[list[nx.Graph]], where each element is the list of VNRs in one window.
    Window i: arrival_time ∈ [i * window_size, (i+1) * window_size)

    Note: Empty windows are retained (to handle their corresponding leave events).
    Trailing empty windows are pruned.
    """
    if not vnr_stream:
        return [[]]

    max_time = max(v.graph["arrival_time"] for v in vnr_stream)
    n_windows = int(max_time / self.window_size) + 1

    windows = [[] for _ in range(n_windows)]
    for vnr in vnr_stream:
        w_idx = int(vnr.graph["arrival_time"] / self.window_size)
        windows[w_idx].append(vnr)

    # Prune trailing empty windows (but keep at least 1)
    while len(windows) > 1 and not windows[-1]:
        windows.pop()

    return windows
```

### 2.5 `_advance_to_window()` — Transition to a New Window

```python
def _advance_to_window(self, window_idx: int):
    """
    Handles the transition to a new window.
    Called at the start of each window (same as inference).

    Flow:
      1. Compute window_start_time = window_idx * window_size
      2. Expire VNRs: active_embeddings with expiry < window_start_time
         → release resources back to substrate
      3. Load VNRs for this window into vnr_queue
      4. Drop VNRs exceeding max_queue_delay from vnr_queue (if old queue overflow exists)

    CRITICAL: Step 1 (expire) MUST happen BEFORE step 3 (load new).
    This is the resource release mechanism, matching inference behavior.
    """
    window_start = window_idx * self.window_size
    window_end   = (window_idx + 1) * self.window_size

    # --- Step 1: Expire VNRs whose lifetime has ended ---
    still_active = []
    for emb in self.active_embeddings:
        vnr, mapping, link_paths, expiry = emb
        if expiry < window_start:
            release_vnr_embedding(self.substrate, vnr, mapping, link_paths)
            # Resources returned to substrate safely (handles undirected edge directions correctly)
        else:
            still_active.append(emb)
    self.active_embeddings = still_active

    # --- Step 2: Load VNRs for this window ---
    if window_idx < len(self._all_windows):
        new_vnrs = self._all_windows[window_idx]
    else:
        new_vnrs = []

    # --- Step 3: Build vnr_queue (unprocessed VNRs in this window) ---
    # Drop VNR if it has waited too long (arrival_time + max_queue_delay < window_start)
    self.vnr_queue = []
    for vnr in new_vnrs:
        wait_time = window_start - vnr.graph["arrival_time"]
        if wait_time <= self.max_queue_delay:
            self.vnr_queue.append(vnr)
        # else: drop (expired in queue before being processed)

    self.window_expiry_time = window_end
```

### 2.6 `step()` — Execute an Action

```python
# Required import (top of environment_v2.py)
from src.evaluation.eval import cost_of_embedding
```

```python
def step(self, action: int):
    """
    Action = index into self.vnr_queue (not a global index).
    Selects the VNR at that index for HPSO to map next.

    After the entire vnr_queue has been processed:
      - If windows remain: advance_to_window(next) → automatically loads new VNRs
      - If no windows remain: done=True

    done=True occurs only when ALL N VNRs across ALL windows have been processed.
    There is no done signal between windows.
    """
    assert self.vnr_queue, "step() called with an empty vnr_queue"
    assert 0 <= action < len(self.vnr_queue), f"action {action} out of range"

    vnr = self.vnr_queue.pop(action)

    # --- HPSO embedding ---
    result = hpso_embed(substrate_graph=self.substrate, vnr_graph=vnr, **self.hpso_params)

    if result is not None:
        mapping, link_paths = result
        expiry = vnr.graph["arrival_time"] + vnr.graph["lifetime"]
        self.active_embeddings.append((vnr, mapping, link_paths, expiry))
        self.episode_accepted.append((vnr, mapping, link_paths))
        cost = cost_of_embedding(mapping, link_paths, vnr, self.substrate)
        self.episode_accepted_costs.append(cost)
        success = True
    else:
        self.episode_rejected.append(vnr)
        cost = None
        success = False

    # --- Reward ---
    # done is computed before reward (terminal reward depends on done)
    done = self._check_done_and_advance()
    reward = self._compute_reward(success, vnr, done, cost)

    return self._get_obs(), reward, done, False, self._get_info(success)

def _check_done_and_advance(self) -> bool:
    """
    Checks and advances the window if vnr_queue is empty.
    Returns True if the episode has ended (all windows exhausted).
    """
    if self.vnr_queue:
        return False  # VNRs remain in this window → continue

    # Current window exhausted → advance to the next
    self.window_idx += 1

    if self.window_idx >= self.total_windows:
        return True  # All windows exhausted → done

    # Load the next window (process leave events + new VNRs)
    self._advance_to_window(self.window_idx)

    # New window may be empty (no VNRs, or all dropped)
    # → keep advancing until VNRs are found or all windows are exhausted
    while not self.vnr_queue:
        self.window_idx += 1
        if self.window_idx >= self.total_windows:
            return True
        self._advance_to_window(self.window_idx)

    return False
```

### 2.7 Resource Lifecycle

The manual `_release_embedding` implementation is removed. Iterating over paths manually on
undirected graphs causes `KeyError`s or double-accounting when the edge direction key doesn't
match how NetworkX stored it — this is the root cause of negative-bandwidth resource depletion.

Instead, use the existing robust utility from `src.utils.graph_utils`, which handles edge
direction correctly and mirrors exactly how `fast_hpso.py` deducts resources:

```python
# In environment_v2.py — top-level imports
from src.utils.graph_utils import release_vnr_embedding

# Inside _advance_to_window(self, window_idx: int) — Step 1: Expire VNRs
still_active = []
for emb in self.active_embeddings:
    vnr, mapping, link_paths, expiry = emb
    if expiry < window_start:
        release_vnr_embedding(self.substrate, vnr, mapping, link_paths)
        # Resources returned to substrate safely
    else:
        still_active.append(emb)
self.active_embeddings = still_active
```

`_release_embedding` is removed entirely from the class. All call sites in
`_advance_to_window` are replaced with the direct call to `release_vnr_embedding`.



### 2.9 `episode_summary()` — End-of-Episode Summary

```python
def episode_summary(self) -> dict:
    n_accepted = len(self.episode_accepted)
    n_rejected = len(self.episode_rejected)
    n_total = n_accepted + n_rejected

    total_rev = sum(revenue_of_vnr(v) for v, _, _ in self.episode_accepted)
    total_cost = sum(self.episode_accepted_costs) if self.episode_accepted_costs else 1e-6

    return {
        "acceptance_rate": n_accepted / (n_total + 1e-9),
        "revenue_cost_ratio": min(total_rev / total_cost, 2.0),
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "n_total": n_total,
    }
```

---

## Part 3 — Reward Design

### 3.1 Reward Components

```
r_t = r_step(t) + r_terminal (only when done=True)
```

**Per-step reward** (dense signal):

| Event | Formula | Rationale |
|---|---|---|
| HPSO accept | `α · rc_norm + (1-α) · inline_ar` | R/C + running AR |
| HPSO fail | `-(1-α) · (1 - inline_ar)` | Penalizes choosing an unmappable VNR |

Where:
- `rc_norm = R/C / (1 + R/C)` ∈ [0,1) — squashed, monotone
- `inline_ar = n_accepted / n_total` at the current step ∈ [0,1]
- `α = 0.5` (tunable)
- Reward range: `(-0.5, 1)` — bounded, stable

**Terminal reward** (end of episode, when `done=True`):
```python
ar = n_accepted / (n_total + ε)
rc = total_rev / (total_cost + ε), clamp [0, 1]
r_terminal = λ_ar * ar + λ_rc * rc   # λ_ar=0.4, λ_rc=0.6
```

GAE (γ=0.99, λ=0.95) propagates the terminal reward backward to all prior steps within the episode.
Because 1 episode = all N VNRs, the critic learns long-term value from the full VNR stream encoding.

---

## Part 4 — `train_ppo_v2.py`

### 4.1 PPO Training Loop — Episode-Aligned

```
CRITICAL DIFFERENCE from the old plan:
  - Old plan: collect N_STEPS transitions regardless of episode boundaries
  - New plan: collect ONE FULL EPISODE (done=True) before each update

Reason: 1 episode = N VNR steps. GAE needs the full episode to correctly compute
terminal reward. Truncating mid-episode causes some episodes to receive a bootstrap
value instead of the true terminal bonus.
```

```python
class PPOTrainerV2:

    def __init__(self, cfg: PPOConfigV2):
        import os
        # Train env: completely separate, uses train dataset
        self.train_env = VNEEnvironmentV2(
            data_mode=DataMode.PREGENERATED,
            substrate_path=os.path.join(cfg.train_dir, "substrate.json"),
            vnr_path=os.path.join(cfg.train_dir, "vnr_stream.json"),
            window_size=cfg.window_size,
            # ...
        )
        # Eval env: completely separate, uses eval dataset
        self.eval_env = VNEEnvironmentV2(
            data_mode=DataMode.PREGENERATED,
            substrate_path=os.path.join(cfg.eval_dir, "substrate.json"),
            vnr_path=os.path.join(cfg.eval_dir, "vnr_stream.json"),
            window_size=cfg.window_size,
            # ...
        )
        ...

    def _collect_one_episode(self) -> list:
        """
        Collects transitions for ONE COMPLETE EPISODE.

        Returns: list of (obs, action, log_prob, value, reward, done)

        Does not truncate mid-episode. If an episode exceeds max_steps_per_episode,
        truncate and log a warning (should not occur if N VNRs is controlled).
        """
        transitions = []
        obs, _ = self.train_env.reset()   # Start new episode = fresh substrate
        done = False

        while not done:
            if not obs["vnr_list"]:
                # Should not occur — env handles this in _check_done_and_advance
                # If it does: this is a bug in the env
                raise RuntimeError("vnr_list empty mid-episode — bug in env")

            sub_data  = obs["substrate"].to(self.device)
            vnr_datas = [v.to(self.device) for v in obs["vnr_list"]]
            obs_dev   = {"substrate": sub_data, "vnr_list": vnr_datas}

            with torch.no_grad():
                action, log_prob, _, value = self.ac.get_action_and_value(obs_dev)

            next_obs, reward, done, _, info = self.train_env.step(action.item())

            transitions.append({
                "obs": obs,
                "action": action,
                "log_prob": log_prob,
                "value": value.squeeze(),
                "reward": reward,
                "done": done,
            })

            obs = next_obs

        return transitions

    def train(self):
        """
        Main training loop: 1 iteration = 1 episode = 1 epoch.

        for epoch in range(num_epochs):
            transitions = collect_one_episode()   ← all N VNRs
            advantages, returns = compute_gae(transitions)
            update(transitions, advantages, returns)
            log(...)
            if epoch % eval_every == 0:
                evaluate()
        """
        for epoch in range(self.cfg.num_epochs):
            # --- Collect one complete episode ---
            transitions = self._collect_one_episode()
            ep_summary = self.train_env.episode_summary()

            # --- GAE ---
            advantages, returns = self._compute_gae(transitions)

            # --- PPO update ---
            loss_dict = self._update(transitions, advantages, returns)

            # --- Logging ---
            self._log_train(epoch, ep_summary, loss_dict)

            # --- Periodic evaluation ---
            if (epoch + 1) % self.cfg.eval_every == 0:
                eval_metrics = self.evaluate()
                self._log_eval(epoch, eval_metrics)
                self._maybe_save_best(epoch, eval_metrics)
```

### 4.2 GAE Computation — Full Episode

```python
def _compute_gae(self, transitions: list) -> tuple:
    """
    Computes GAE over the full episode.

    Since the episode always ends with done=True, last_value = 0.
    No bootstrapping from next_obs is needed.

    This is a key distinction:
    - Old plan (collect N_STEPS): must bootstrap last_value from next_obs
    - New plan (collect 1 episode): last done=True → last_value = 0
    """
    T = len(transitions)
    gamma = self.cfg.gamma
    lam   = self.cfg.gae_lambda

    values   = [t["value"].item() for t in transitions]
    rewards  = [t["reward"] for t in transitions]
    dones    = [t["done"] for t in transitions]

    advantages = [0.0] * T
    last_gae   = 0.0

    # Backward pass (last done=True → next_val=0, last_gae=0)
    for t in reversed(range(T)):
        next_val = values[t+1] if t < T-1 else 0.0
        next_nonterminal = 0.0 if dones[t] else 1.0

        delta    = rewards[t] + gamma * next_val * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae

    advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
    returns    = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)
    return advantages, returns
```

### 4.3 Evaluation — Greedy Rollout on Eval Env

```python
def evaluate(self, n_episodes: int = None) -> dict:
    """
    Greedy rollout on self.eval_env.
    self.train_env is not touched at any point during this function.
    """
    n_episodes = n_episodes or self.cfg.eval_episodes
    self.model.eval()

    all_ars, all_rcs, all_rewards = [], [], []

    for ep in range(n_episodes):
        # We evaluate on the EXACT SAME EVAL DATASET n_episodes times
        # to average out HPSO's stochasticity for a smoother/more meaningful plot.
        obs, _ = self.eval_env.reset()
        ep_reward = 0.0
        done = False

        with torch.no_grad():
            while not done:
                if not obs["vnr_list"]:
                    break  # Should not occur
                dist, value = self.model(obs)
                action = dist.logits.argmax()   # Greedy
                obs, reward, done, _, _ = self.eval_env.step(action.item())
                ep_reward += reward

        summary = self.eval_env.episode_summary()
        all_ars.append(summary["acceptance_rate"])
        all_rcs.append(summary["revenue_cost_ratio"])
        all_rewards.append(ep_reward)

    self.model.train()

    return {
        "Eval/AcceptanceRate":   sum(all_ars) / len(all_ars),
        "Eval/RevenueCostRatio": sum(all_rcs) / len(all_rcs),
        "Eval/EpisodeReward":    sum(all_rewards) / len(all_rewards),
        "Eval/AR_std":           float(torch.tensor(all_ars).std()),
    }
```

### 4.4 Config Dataclass

```python
@dataclass
class PPOConfigV2:
    # --- Data paths ---
    train_dir: str = ""
    eval_dir: str = ""
    window_size: int = 50

    # --- Training ---
    num_epochs: int     = 500         # Number of episodes to train
    batch_size: int     = 64          # Mini-batch size in PPO update
    n_ppo_epochs: int   = 8           # Number of passes over the buffer per update
    lr: float           = 3e-4
    gamma: float        = 0.99
    gae_lambda: float   = 0.95
    clip_range: float   = 0.2
    ent_coef: float     = 0.01
    vf_coef: float      = 0.5
    grad_clip: float    = 0.5

    # --- HPSO ---
    hpso_particles: int  = 20
    hpso_iterations: int = 10         # Reduced to speed up training

    # --- Logging / Eval ---
    eval_every: int     = 10          # Evaluate every N epochs
    eval_episodes: int  = 5           # Number of episodes per evaluation
    log_dir: str        = "runs"
    save_dir: str       = "checkpoints"
    run_name: str       = "ppo_v2"
    device: str         = "auto"
    load_checkpoint: Optional[str] = None
```

---

## Part 5 — TensorBoard Metrics

### Group 1 — PPO Loss (per epoch, after update)

| Key | Healthy sign |
|---|---|
| `Train/PolicyLoss` | Decreases then stabilizes |
| `Train/ValueLoss` | Steadily decreases |
| `Train/Entropy` | Starts high (~ln B_avg), decreases slowly |
| `Train/ApproxKL` | < 0.02; spikes indicate LR is too high |
| `Train/ClipFraction` | 0.05–0.20 |
| `Train/ExplainedVariance` | Increases toward 1.0 |

### Group 2 — Training Episode Metrics (per epoch)

| Key | Healthy sign |
|---|---|
| `Train/EpisodeReward` | Upward trend (use smoothing) |
| `Train/AcceptanceRate` | Increases from baseline |
| `Train/RevenueCostRatio` | Increases |
| `Train/NFailed` | Decreases as policy improves |
| `Train/AvgScoreAccepted` | Increases relative to AvgScoreFailed |
| `Train/AvgScoreFailed` | Diverges downward from Accepted |

**Key diagnostic:** `Train/AvgScoreAccepted - Train/AvgScoreFailed` → a growing gap means the GNN is learning embeddability-aware features. A gap of 0 means scores are essentially random.

### Group 3 — Eval Metrics (every `eval_every` epochs)

**This is the definitive signal that learning is effective:**

| Key | Healthy sign |
|---|---|
| `Eval/AcceptanceRate` | Monotonically increasing |
| `Eval/RevenueCostRatio` | Increasing |
| `Eval/EpisodeReward` | Increasing |
| `Eval/AR_std` | Decreasing (policy becomes more consistent) |

**Reading Train vs Eval:**

| Pattern | Diagnosis |
|---|---|
| Both increase together | ✅ Policy generalizes well |
| Train up, Eval flat | ⚠️ Overfitting — add more replicas |
| Both flat | ❌ No learning — check entropy, LR, reward scale |
| Eval up, Train noisy | ✅ Normal — training is noisy due to diverse pool |
| Eval up then collapses | ❌ Catastrophic forgetting — reduce lr or n_ppo_epochs |

### Group 4 — Substrate State

| Key | Description |
|---|---|
| `Substrate/CpuUtilization` | `1 - avail_cpu/total_cpu` at end of episode |
| `Substrate/BwUtilization` | `1 - avail_bw/total_bw` at end of episode |
| `Substrate/ActiveEmbeddingsPeak` | Number of VNRs holding resources at peak |
| `Dataset/EpochWindow` | window_idx / total_windows at end of epoch |

---

## Part 6 — Actor-Critic Architecture

```python
class GNNActorCriticV2(nn.Module):
    def __init__(self, scheduler, substrate_emb_dim=128):
        super().__init__()
        self.scheduler  = scheduler
        self.value_head = nn.Linear(substrate_emb_dim, 1)

    def forward(self, obs) -> tuple[Categorical, torch.Tensor]:
        sub_data = obs["substrate"]
        vnr_list = obs["vnr_list"]

        # Actor: score each remaining VNR → Categorical
        scores = self.scheduler(sub_data, vnr_list)   # [B]
        dist   = Categorical(logits=scores)

        # Critic: value of the current substrate state
        h_s   = self.scheduler.substrate_encoder(sub_data)  # [1, D]
        value = self.value_head(h_s).squeeze(-1)              # scalar

        return dist, value
```

---

## Part 7 — Implementation Checklist

### `environment_v2.py` (~400 lines)

- [ ] `DataMode` enum: `PREGENERATED` only
- [ ] `VNEEnvironmentV2(gymnasium.Env)`:
  - [ ] `__init__`: validates PREGENERATED mode params, loads substrate and VNR stream once
  - [ ] `reset()`: loads data per mode, fresh substrate copy, partitions windows, advances to window 0
  - [ ] `step(action)`: pops VNR from queue, runs HPSO, computes reward, calls `_check_done_and_advance()`
  - [ ] `_partition_into_windows()`: splits VNR stream into list of lists by arrival_time
  - [ ] `_advance_to_window(idx)`: expires lifetime-ended VNRs via `release_vnr_embedding`, loads new VNRs, drops expired-in-queue
  - [ ] `_check_done_and_advance()`: advances window when queue is empty, skips empty windows, returns done
  - [ ] `_get_obs()`: builds PyG Data from substrate + remaining vnr_queue
  - [ ] `_compute_reward()`: pluggable reward computation
  - [ ] `episode_summary()`: AR, R/C, n_accepted, n_rejected

### `train_ppo_v2.py` (~500 lines)

- [ ] `PPOConfigV2` dataclass with `data_mode` flag
- [ ] `PPOTrainerV2`:
  - [ ] `__init__`: constructs `self.train_env` + `self.eval_env` separately
  - [ ] `train()`: loop `num_epochs`, each epoch = one full episode
  - [ ] `_collect_one_episode()`: collects transitions until `done=True`
  - [ ] `_compute_gae()`: GAE over full episode, `last_value=0` because `done=True`
  - [ ] `_update()`: mini-batch PPO with clip + entropy + value loss
  - [ ] `evaluate()`: greedy rollout on `self.eval_env`, `train_env` is untouched
  - [ ] `_log_train()` / `_log_eval()`: TensorBoard logging
  - [ ] `_maybe_save_best()`: saves checkpoint when `Eval/AcceptanceRate` improves

### `generate_datasets.py` — RL branch

- [ ] Update `generate_rl_training_dataset()` call in `src/scripts/generate_datasets.py` to parse and pass `args.num_vnrs`, `args.substrate_nodes`, `args.vnr_min_nodes`, `args.vnr_max_nodes`.
- [ ] Ensure it creates a single `substrate.json` and `vnr_stream.json`.

---

## Part 8 — Data Preparation & Run Commands

```bash
# 1. Generate ONE training dataset
#    → written to dataset/train/rl_training/substrate.json
#                             dataset/train/rl_training/vnr_stream.json
python -m src.scripts.generate_datasets \
    --experiments rl \
    --num-vnrs 1000 \
    --substrate-nodes 80 \
    --vnr-min-nodes 2 \
    --vnr-max-nodes 10 \
    --base-seed 42 \
    --output-dir dataset/train \
    --force

# 2. Generate ONE held-out eval dataset (different seed)
#    → written to dataset/eval/rl_training/substrate.json
#                            dataset/eval/rl_training/vnr_stream.json
python -m src.scripts.generate_datasets \
    --experiments rl \
    --num-vnrs 1000 \
    --substrate-nodes 80 \
    --vnr-min-nodes 2 \
    --vnr-max-nodes 10 \
    --base-seed 9999 \
    --output-dir dataset/eval \
    --force

# 3. Train PPO v2
python -m src.training.train_ppo_v2 \
    --train-dir dataset/train/rl_training \
    --eval-dir  dataset/eval/rl_training \
    --num-epochs 500 \
    --reward simple \
    --eval-every 10 \
    --eval-episodes 5 \
    --run-name ppo_v2_pregenerated \
    --save-dir checkpoints

# 4. Monitor training
tensorboard --logdir runs/
# Key panels:
#   Train/PolicyLoss, Train/ValueLoss, Train/ExplainedVariance  ← optimizer health
#   Train/AcceptanceRate (smoothed ~50 epochs)                  ← noisy training signal
#   Eval/AcceptanceRate, Eval/RevenueCostRatio                  ← true learning signal

# 5. Evaluate final checkpoint
python -m src.training.evaluate \
    --checkpoint checkpoints/ppo_v2_pregenerated_best.pt \
    --eval-dir dataset/eval/rl_training \
    --hpso-iter 30 \
    --episodes 20
```

---

## Part 9 — Key Design Decisions

### Why 1 epoch = 1 episode (all N VNRs)?

**GAE requires a clear episode boundary.** The terminal reward at the end of an episode (AR + R/C across all N VNRs) is the most important signal. If the episode is truncated mid-way (as in the old plan with `n_steps=512`), the terminal reward of long episodes is never fully realized — some episodes are cut before `done`, and those transitions receive a bootstrap value instead of the true terminal bonus.

With 1 epoch = 1 episode: every trajectory ends with `done=True`, `last_value=0`, and the terminal reward is fully computed and propagated backward through GAE to all preceding steps.

**Trade-off:** The per-epoch buffer is larger (N transitions instead of `n_steps`). With N=200 VNRs, this is ~200 transitions/epoch — entirely manageable.

### Why reset the substrate to fresh each epoch?

**To ensure i.i.d. episodes and stable training.** If the substrate carried forward between epochs, each epoch would start from a different state (depending on the previous epoch's policy). This introduces non-stationarity at the epoch level, making it harder for PPO to converge.

However, **within an episode**, the substrate is not reset — it is consumed VNR by VNR as mappings occur, matching inference dynamics. This is the deliberate balance:
- Across epochs: i.i.d. for PPO stability
- Within an episode: non-stationary to learn the correct dynamics

**Contrast with inference:** in inference, the substrate never resets (continuous stream). Training with per-epoch resets is a reasonable approximation because the critic learns the value function from the substrate state encoding.

### Why are leave events processed at the start of each window (not after every step)?

This matches inference. In `hpso_batch_scheduler.py`, resource release happens on window transitions, not after each HPSO call. If training released resources after every step, the dynamics would diverge and the trained policy would suffer a distribution shift at deployment.

### Why two separate env instances (instead of `set_mode()`)?

`set_mode()` would have to reset the substrate to switch context — this breaks substrate state continuity within a running episode. With two separate instances, `self.train_env` is completely frozen throughout `evaluate()`, with no shared state and no side effects.
