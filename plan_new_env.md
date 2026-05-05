# PPO v2 Training Pipeline — Comprehensive Plan

## Overview & Motivation

The current `train_ppo.py` + `VNEOrderingEnv` has a critical mismatch with how the trained
model is actually **used** at inference time (`hpso_batch_scheduler.py`):

| Dimension | Current training | Inference (`hpso_batch_scheduler.py`) |
|---|---|---|
| Substrate source | Fresh random substrate every episode | Live, continuously-depleting substrate |
| VNR source | Random batch each episode | Real time-window from simulator |
| VNR expiry | None — substrate never releases resources | VNRs expire after `lifetime` → resources returned |
| Scheduling loop | Single MDP over one batch | Called repeatedly on the same substrate per window |
| Rejection | No explicit rejection — every VNR is attempted | Any VNR below score threshold can be skipped |

The v2 pipeline closes all these gaps while keeping `VNRScheduler` (model architecture)
completely unchanged.

---

## New Files to Create

```
src/
├── scheduler/
│   ├── environment_v2.py        ← NEW  (replaces environment.py for training)
│   └── model.py                 (unchanged)
└── training/
    ├── train_ppo_v2.py          ← NEW  (replaces train_ppo.py)
    └── generate_data.py         (unchanged — still used for on-the-fly data)
```

---

## Part 1 — `environment_v2.py`

### 1.1 Data Generation Strategy

The environment must expose the agent to **many different (substrate, VNR stream) pairs**
so the policy generalises. Two modes are supported:

**Mode A — Dataset file** (recommended for serious training)
```
dataset/
└── rl_training/
    ├── train/
    │   ├── replica_0/{substrate.json, vnr_stream.json}   ← training pool
    │   ├── replica_1/{substrate.json, vnr_stream.json}
    │   ├── replica_2/{substrate.json, vnr_stream.json}
    │   ├── replica_3/{substrate.json, vnr_stream.json}
    │   └── replica_4/{substrate.json, vnr_stream.json}
    └── eval/
        ├── eval_0/{substrate.json, vnr_stream.json}      ← held-out, NEVER used in training
        └── eval_1/{substrate.json, vnr_stream.json}
```
The environment loads this once at startup, then slices it into time windows during
`reset()`.

**Critical:** eval replicas must be generated separately from train replicas and
never touched during training rollout collection. They exist solely for the periodic
`evaluate()` call in the trainer. This is what makes the TensorBoard eval curve
meaningful — the same fixed inputs every evaluation, so score changes reflect genuine
policy improvement rather than data variance or lucky sampling.

**Mode B — On-the-fly generation** (quick experiments / unit tests)
Uses `make_substrate_fn` + `make_batch_fn` from `generate_data.py` to produce fresh
data for each episode. No file I/O needed.

Both modes produce the same internal episode structure.

### 1.2 Time-Window Simulation

The environment mirrors `BatchedVNRSimulator` from `src/simulation/simulator.py`:

```
Timeline:
  t=0    arrival_time of VNR_0
  ...
  t=W    window boundary  ─────────────────────────────────
         window_0 = {VNR_i : arrival_time ∈ [0, W)}
         window_1 = {VNR_i : arrival_time ∈ [W, 2W)}
         ...

Each window is one "episode" for the agent.
Between windows the substrate is NOT reset — it carries over depletion.
Expired VNRs (arrival_time + lifetime < current_window_start) release resources.
```

### 1.3 Episode (Time-Window) Structure

```
reset(window_idx=None):
    1. Expire all VNRs whose (arrival + lifetime) < window_start_time
    2. Collect VNRs whose arrival_time falls in this window → vnr_queue
    3. Return obs = { substrate_pyg, remaining_vnr_pyg_list, n_remaining }

step():
    # No explicit action argument — the environment itself applies the threshold.
    scores = scheduler.predict(sub_pyg, vnr_pyg_list)   # [B]
    best_score, best_idx = scores.max(dim=0)

    if best_score.item() < REJECT_THRESHOLD:            # threshold-based rejection
        pop vnr_queue[best_idx]                         # remove without embedding
        reward = rejection_penalty(vnr)
    else:
        vnr = vnr_queue[best_idx]
        result = hpso_embed(substrate, vnr)
        if result:
            commit resources to substrate
            push to active_embeddings with expiry = arrival + lifetime
            reward = accept_reward(vnr, result)
        else:
            reward = embed_fail_penalty(vnr)
        pop vnr from queue

    done = (vnr_queue is empty)
    if done:
        terminal_reward = terminal_bonus(episode_stats)
        reward += terminal_reward
    return next_obs, reward, done, truncated, info
```

### 1.4 Threshold-Based Rejection

The environment uses a **score threshold** to decide whether to embed or reject the best-scored VNR. No explicit reject action exists in the action space — the policy expresses rejection implicitly by producing low scores for unembeddable VNRs, and the environment acts on this signal.

The network is trained so that "bad" (unembeddable or inefficient) VNRs naturally receive negative scores. When all remaining VNRs score below the threshold, the highest-scored one is rejected and the loop continues.

```python
REJECT_THRESHOLD = 0.0   # tunable; scores are unbounded logits

scores = scheduler.predict(sub_pyg, vnr_pyg_list)   # [B]
best_score, local_idx = scores.max(dim=0)
if best_score.item() < REJECT_THRESHOLD:
    # reject best candidate — pop from queue, small negative reward
    global_idx = remaining_indices.pop(local_idx)
    rejected.append(vnr_list[global_idx])
    # apply rejection_penalty and continue
else:
    # embed best candidate via HPSO
    ...
```

**Why this is better than a learned reject action:**
- No architecture change to `VNRScheduler` or `GNNActorCritic`
- No extra `reject_head` parameter to tune or maintain
- Inference in `hpso_batch_scheduler.py` already works this way — training and inference are identical
- The threshold (`0.0`) is a transparent, tunable hyperparameter — increase it to be more aggressive at rejecting, decrease for more permissive behaviour

### 1.5 Multi-Substrate Support

The environment maintains a **pool** of (substrate, vnr_stream) pairs:

```python
class VNEEnvironmentV2:
    def __init__(self, train_paths: list[dict], eval_paths: list[dict], ...):
        self.train_pool = [load(d) for d in train_paths]
        self.eval_pool  = [load(d) for d in eval_paths]
        self.mode = "train"              # "train" | "eval"
        self.train_idx = 0
        self.eval_idx  = 0

    def set_mode(self, mode: str):
        """Switch between training pool and held-out eval pool."""
        assert mode in ("train", "eval")
        self.mode = mode
        if mode == "eval":
            self.eval_idx = 0           # always replay eval from the start
            self._reset_substrate()     # clean slate for reproducibility

    def reset(self, new_dataset=True):
        pool     = self.train_pool if self.mode == "train" else self.eval_pool
        idx_attr = "train_idx"     if self.mode == "train" else "eval_idx"
        if new_dataset:
            setattr(self, idx_attr, (getattr(self, idx_attr) + 1) % len(pool))
        # Reset window pointer for chosen dataset
        ...
```

When a stream is exhausted (all windows processed), the next `reset()` picks
the next dataset from the active pool. The eval pool always resets to index 0
at the start of each `evaluate()` call so results are reproducible across runs.

### 1.6 Resource Lifecycle (Exact Physics)

The environment must replicate `BatchedVNRSimulator`'s resource accounting exactly:

```python
# On accept:
for v_node, s_node in mapping.items():
    substrate.nodes[s_node]['cpu'] -= vnr.nodes[v_node]['cpu']
for (u, v), path in link_paths.items():
    bw = vnr.edges[u, v]['bw']
    for i in range(len(path)-1):
        substrate.edges[path[i], path[i+1]]['bw'] -= bw

# Track: active_embeddings.append((vnr, mapping, link_paths, expiry_time))

# On each window transition (before reset):
for emb in expired_embeddings:
    for v_node, s_node in emb.mapping.items():
        substrate.nodes[s_node]['cpu'] += vnr.nodes[v_node]['cpu']
    for (u, v), path in emb.link_paths.items():
        bw = vnr.edges[u, v]['bw']
        for i in range(len(path)-1):
            substrate.edges[path[i], path[i+1]]['bw'] += bw
```

---

## Part 2 — Reward Design

This is the hardest part. The agent sees **one time window** but we care
about **long-term AR and R/C across all windows**.

### 2.1 Reward Components

```
r_t = r_step(t) + r_terminal(episode)
```

**Per-step reward `r_step`** (immediate signal):

| Event | Formula | Rationale |
|---|---|---|
| HPSO accept | `α · rc_norm + (1-α) · 0` | Reward efficiency immediately |
| HPSO fail (attempt but fail) | `-β · revenue(vnr) / max_revenue` | Penalise wasting HPSO budget |
| Explicit reject | `-γ · revenue(vnr) / max_revenue` | Small penalty: rejecting has cost |

Where:
- `rc_norm = clip(revenue/cost, 0, 1)` — R/C normalised to [0,1]
- `α = 0.6`, `β = 0.05`, `γ = 0.02` (tunable)
- All rewards bounded to `(-1, 1)` for stable training

**Terminal reward `r_terminal`** (end-of-window):

```python
n_total    = n_accepted + n_rejected + n_failed
ar         = n_accepted / (n_total + ε)
total_rev  = Σ revenue(accepted_vnrs)
total_cost = Σ real_embedding_cost(accepted_vnrs)
rc         = clip(total_rev / (total_cost + ε), 0, 1)

# Convex blend — tunable weights
r_terminal = λ_ar * ar + λ_rc * rc      # λ_ar=0.4, λ_rc=0.6 initially
```

**Why this design works for long-term:**

GAE (γ=0.99, λ=0.95) propagates the terminal reward backward through all
steps in the window. Since each episode = one window, and the substrate
state carries forward between episodes, decisions in window t affect which
VNRs can be embedded in window t+1 (through resource depletion). The value
function (critic) learns to predict multi-window returns because the
substrate state encodes future opportunities.

### 2.2 Rejection Calibration

The per-step rejection penalty `γ` must be calibrated so that:
- Rejecting a truly unembeddable VNR → reward ≈ 0 (no waste)
- Rejecting an embeddable VNR → reward < 0 (punished)

Start with `γ = 0.02` and increase if the agent learns to reject everything.
Monitor `n_rejected / n_total` in TensorBoard.

### 2.3 Curriculum

Start with `reward_mode = "simple"` (binary accept/reject) to validate
that the policy learns at all. Then switch:

```
Steps 0 → 50k:   simple     (binary: +1 accept, -0.5 reject)
Steps 50k → 150k: r2c_ac    (R/C + AR blend)
Steps 150k+:      longterm   (full terminal bonus)
```

---

## Part 3 — `train_ppo_v2.py`

### 3.1 What the Trainer Does (and Doesn't Do)

**Trainer responsibilities:**
- Collect rollouts from `VNEEnvironmentV2` (training pool only)
- Run GAE to compute advantages
- Run PPO mini-batch updates on `GNNActorCritic`
- Every `eval_every` steps: run `evaluate()` on the held-out eval pool (greedy, no grad)
- Log all metrics to TensorBoard
- Save checkpoints (best checkpoint tracked by `Eval/AcceptanceRate`)

**Trainer does NOT:**
- Generate data (environment does that)
- Manage VNR expiry (environment does that)
- Call HPSO directly (environment does that)

### 3.2 Rollout Collection (Matching Inference)

The rollout loop must exactly mirror `hpso_batch_scheduler.py`:

```python
# Training rollout (1 window = 1 episode):
obs, _ = env.reset()
while not done:
    sub_pyg  = obs["substrate"]          # current (depleted) substrate
    vnr_pygs = obs["vnr_list"]           # remaining VNRs in window

    scores = model(sub_pyg, vnr_pygs)    # [B] — same call as inference

    # Threshold gate — mirrors hpso_batch_scheduler.py exactly
    best_score, best_idx = scores.max(dim=0)
    if best_score.item() < REJECT_THRESHOLD:
        obs, reward, done, _, info = env.step_reject(best_idx.item())
    else:
        dist   = Categorical(logits=scores)
        action = dist.sample()           # ∈ {0, …, B-1}
        obs, reward, done, _, info = env.step(action.item())
    # buffer: store (obs, action, log_prob, value, reward, done)
```

At inference time the same logic runs but with `argmax` instead of `sample`. No separate reject action index — rejection is a deterministic threshold gate, not a sampled action, so training and inference are structurally identical.

### 3.3 Config Dataclass (`PPOConfigV2`)

```python
@dataclass
class PPOConfigV2:
    # Data — training pool
    dataset_paths: list[dict]        # [{"substrate_path":..., "vnr_path":...}]
    # Data — held-out eval pool (never used during rollout collection)
    eval_paths: list[dict] = field(default_factory=list)
    # OR on-the-fly:
    use_generated_data: bool = False
    sub_min_nodes: int = 40
    sub_max_nodes: int = 90

    # Window
    window_size: int = 50            # time units per window
    max_queue_delay: int = 100

    # PPO
    total_timesteps: int = 1_000_000
    n_steps: int = 1024
    batch_size: int = 128
    n_epochs: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    grad_clip: float = 0.5

    # Reward
    reward_mode: str = "simple"      # simple | r2c_ac | longterm
    reject_penalty: float = 0.02
    embed_fail_penalty: float = 0.05
    terminal_ar_weight: float = 0.4
    terminal_rc_weight: float = 0.6
    # Reject
    reject_threshold: float = 0.0       # scores below this → reject without HPSO

    # HPSO
    hpso_particles: int = 20
    hpso_iterations: int = 10        # lower during training for speed

    # Logging
    log_every: int = 1024
    eval_every: int = 5_000          # run held-out evaluation every N timesteps
    eval_episodes: int = 10          # number of eval episodes per eval run
    save_every: int = 5000
    save_dir: str = "checkpoints"
    run_name: str = "ppo_v2"
    device: str = "auto"
```

### 3.4 TensorBoard Metrics to Log

The metrics are split into four groups. Every group has a clear purpose so you can
diagnose whether the policy is learning, overfitting, or broken at a glance.

---

#### Group 1 — PPO Loss (logged every `log_every` timesteps)

These confirm the optimizer is working correctly. Log after every `_update()` call.

| Key | Description | Healthy sign |
|---|---|---|
| `Train/PolicyLoss` | PPO clipped surrogate loss (should decrease then stabilise) | Decreasing, then flat |
| `Train/ValueLoss` | MSE between predicted value and GAE return | Decreasing steadily |
| `Train/EntropyLoss` | Entropy bonus term (−`ent_coef` × entropy) | Slightly negative, stable |
| `Train/TotalLoss` | `PolicyLoss + vf_coef·ValueLoss + EntropyLoss` | Tracks policy + value together |
| `Train/Entropy` | Policy entropy in nats (separate from loss term) | Starts high (~ln B), decays slowly |
| `Train/ApproxKL` | KL divergence between old and new policy | Stays < 0.02; spikes = bad LR |
| `Train/ClipFraction` | Fraction of ratios that were clipped by ε | Should be 0.05–0.20 |
| `Train/ExplainedVariance` | `1 - Var(returns - values) / Var(returns)` | Rises toward 1.0 as critic improves |
| `Train/LearningRate` | Current LR (useful if using LR schedule) | Smooth decay if scheduled |

**How to read them together:**
- `ValueLoss` going down + `ExplainedVariance` going up = critic learning ✓
- `PolicyLoss` going down = actor learning ✓  
- `Entropy` collapsing to 0 early = policy collapsed, increase `ent_coef`
- `ApproxKL` > 0.05 consistently = LR too high or `clip_range` too loose
- `ClipFraction` > 0.3 = updates too large, reduce `lr` or `n_epochs`

---

#### Group 2 — Training Episode Metrics (logged at each window end, x-axis = global_step)

| Key | Description | Healthy sign |
|---|---|---|
| `Train/EpisodeReward` | Total reward for this training window | Rising trend over time |
| `Train/AcceptanceRate` | `n_accepted / n_total` | Rising from baseline |
| `Train/RevenueCostRatio` | `total_rev / total_cost` for accepted VNRs | Rising |
| `Train/NFailed` | HPSO calls that failed (wasted compute) | Should decrease as policy improves |
| `Train/RejectFraction` | `n_rejected / n_total` | Should not be 0% or 100% |
| `Train/AvgScoreAccepted` | Mean logit score of accepted VNRs | Should be > 0 and rising |
| `Train/AvgScoreRejected` | Mean logit score of rejected VNRs | Should be < 0 and diverging from accepted |

**Note:** Training episode metrics are noisy because each window comes from a different
(substrate, VNR stream) pair. Do not expect a smooth curve — look at the rolling mean
trend over ~50 episodes. Use TensorBoard's smoothing slider.

---

#### Group 3 — Held-Out Evaluation Metrics (logged every `eval_every` timesteps)

**This is the definitive measure of whether PPO is actually learning.**

The eval pool is fixed and never seen during training. Running evaluation on the
same 2 substrates every `eval_every` steps produces a stable, interpretable curve.

```python
def evaluate(self, n_episodes: int = 10) -> dict:
    self.model.eval()
    self.env.set_mode("eval")       # switch to held-out pool

    all_ars, all_rcs, all_rewards = [], [], []

    for ep in range(n_episodes):
        obs, _ = self.env.reset()
        ep_reward = 0.0
        done = False

        with torch.no_grad():
            while not done:
                sub_pyg  = obs["substrate"]
                vnr_pygs = obs["vnr_list"]
                scores, value = self.model(obs)

                # Greedy action — no sampling during eval
                best_score, best_idx = scores.logits.max(dim=0)
                if best_score.item() < self.config.reject_threshold:
                    obs, reward, done, _, info = self.env.step_reject(best_idx.item())
                else:
                    obs, reward, done, _, info = self.env.step(best_idx.item())
                ep_reward += reward

        summary = self.env.episode_summary()
        all_ars.append(summary["acceptance_rate"])
        all_rcs.append(summary["revenue_cost_ratio"])
        all_rewards.append(ep_reward)

    self.env.set_mode("train")      # restore training pool
    self.model.train()

    metrics = {
        "Eval/AcceptanceRate":    mean(all_ars),
        "Eval/RevenueCostRatio":  mean(all_rcs),
        "Eval/EpisodeReward":     mean(all_rewards),
        "Eval/AR_std":            std(all_ars),    # stability indicator
    }
    for k, v in metrics.items():
        self.writer.add_scalar(k, v, self.global_step)

    return metrics
```

| Key | Description | Healthy sign |
|---|---|---|
| `Eval/AcceptanceRate` | AR on fixed held-out substrates (greedy policy) | Rising monotonically |
| `Eval/RevenueCostRatio` | R/C on fixed held-out substrates | Rising |
| `Eval/EpisodeReward` | Total reward per eval episode | Rising |
| `Eval/AR_std` | Std-dev of AR across eval episodes | Decreasing (policy more consistent) |

**Diagnosing with Train vs Eval curves:**

| Pattern | Diagnosis |
|---|---|
| Both rising together | ✅ Policy generalising correctly |
| Train rises, Eval flat | ⚠️ Overfitting to training pool — add more replicas |
| Both flat | ❌ Learning stalled — check entropy, LR, reward scale |
| Eval rises, Train noisy | ✅ Normal — train is noisy due to diverse pool, eval is clean |
| Eval collapses after rising | ❌ Catastrophic forgetting — reduce `lr` or `n_epochs` |

---

#### Group 4 — Substrate State & Dataset Cycling

| Key | Description |
|---|---|
| `Substrate/CpuUtilization` | `1 - avail_cpu/total_cpu` after window |
| `Substrate/BwUtilization` | `1 - avail_bw/total_bw` after window |
| `Substrate/ActiveEmbeddings` | VNRs currently occupying substrate |
| `Substrate/ExpiredThisWindow` | VNRs that expired and released resources |
| `Dataset/CurrentIdx` | Which (substrate, stream) pair is active |
| `Dataset/WindowProgress` | `window_idx / total_windows` |

---

## Part 4 — Actor-Critic Architecture (No Reject Head)

`GNNActorCritic` wraps `VNRScheduler` with only a **value head**. There is no `reject_head` — rejection is handled entirely by the `REJECT_THRESHOLD` in the environment and scheduler loop, not by the network.

```python
class GNNActorCriticV2(nn.Module):
    def __init__(self, scheduler, substrate_emb_dim=128):
        super().__init__()
        self.scheduler  = scheduler
        self.value_head = nn.Linear(substrate_emb_dim, 1)
        # NO reject_head — rejection is threshold-based, not learned as a separate action

    def forward(self, obs):
        sub_data = obs["substrate"]
        vnr_list = obs["vnr_list"]

        if not vnr_list:
            # No VNRs — return trivial distribution
            ...

        # VNR scores [B] — the network implicitly signals rejection via score magnitude
        scores = self.scheduler(sub_data, vnr_list)

        # Substrate embedding for value estimate
        h_s   = self.scheduler.substrate_encoder(sub_data)   # [1, 128]
        value = self.value_head(h_s).squeeze(-1)              # [1]

        dist = Categorical(logits=scores)   # action space = {0, …, B-1} only
        return dist, value
```

**Reward signal for rejection:** When the environment rejects via threshold, it applies `rejection_penalty` as the step reward. PPO backpropagates this through the scores — the network learns to produce **negative scores** for VNRs it wants to skip, without any extra parameters.

**Training and inference are identical:**

```python
# Both training (env.step) and hpso_batch_scheduler.py use:
scores = scheduler.predict(sub_data, rem_vnrs)
best_score, local_idx = scores.max(dim=0)
if best_score.item() < 0.0:   # REJECT_THRESHOLD
    global_idx = remaining_indices.pop(local_idx)
    rejected.append(vnr_list[global_idx])
    continue
# else: embed via HPSO
```

This zero-gap between training and inference is the primary motivation for the threshold approach.

---

## Part 5 — Implementation Checklist

### `environment_v2.py`

- [ ] `VNEDataset` class: loads (substrate, vnr_stream) from JSON, splits into windows
- [ ] `VNEDatasetPool` class: manages multiple datasets, round-robin or random
- [ ] `VNEEnvironmentV2(gymnasium.Env)`:
  - [ ] `__init__`: accepts `train_paths` + `eval_paths` (separate pools)
  - [ ] `set_mode(mode)`: switches between `"train"` and `"eval"` pools; resets eval index to 0
  - [ ] `reset()`: expire VNRs, advance window in active pool, build `vnr_queue`
  - [ ] `step(action)`: embed, update substrate, compute reward
  - [ ] `step_reject(idx)`: reject VNR at idx without HPSO, apply rejection penalty
  - [ ] `_expire_vnrs()`: resource release on expiry
  - [ ] `_commit_embedding()`: resource deduction on accept
  - [ ] `_reset_substrate()`: restore substrate to original loaded state (used on eval entry)
  - [ ] `_get_obs()`: build PyG Data objects
  - [ ] `episode_summary()`: returns dict of AR, R/C, utilisation for trainer to log
  - [ ] `_compute_reward()`: pluggable by `reward_mode`
- [ ] On-the-fly mode: fallback when no dataset paths provided
- [ ] Window cycling: when stream exhausted, next `reset()` tries next dataset in active pool

### `train_ppo_v2.py`

- [ ] `PPOConfigV2` dataclass
- [ ] `PPOTrainerV2` class:
  - [ ] `__init__`: build `GNNActorCriticV2`, `VNEEnvironmentV2` (train + eval pools), optimizer
  - [ ] `_collect_rollout(n_steps)`: threshold-based rejection, no reject action index
  - [ ] `_compute_gae()`: unchanged from v1
  - [ ] `_update()`: PPO with KL / clip fraction logging; returns loss dict for TensorBoard
  - [ ] `train()`: main loop — rollout → update → log → eval every `eval_every` steps
  - [ ] `evaluate(n_episodes)`: greedy rollout on held-out eval pool; switches env mode, restores after
  - [ ] best-checkpoint tracking: save whenever `Eval/AcceptanceRate` improves
  - [ ] `save()` / `load()`
- [ ] CLI entry point with all `PPOConfigV2` fields exposed
- [ ] TensorBoard writer with all metrics from §3.4

### Inference compatibility (`hpso_batch_scheduler.py`)

Update `hpso_batch_scheduler.py` to apply the threshold after scoring:

```python
scores = scheduler.predict(sub_data, rem_vnrs)
best_score, local_idx = scores.max(dim=0)
if best_score.item() < 0.0:   # REJECT_THRESHOLD
    global_idx = remaining_indices.pop(local_idx)
    rejected.append(vnr_list[global_idx])
    continue
```

No checkpoint changes needed — the trained `VNRScheduler` weights are loaded identically. The threshold logic is pure Python, not a network parameter.

---

## Part 6 — Data Preparation Workflow

```bash
# 1. Generate training data
python -m src.scripts.generate_datasets \
    --experiments rl \
    --num-vnrs 2000 \
    --substrate-nodes 60,100 \
    --vnr-min-nodes 2 \
    --vnr-max-nodes 10 \
    --num-replicas 5 \
    --output-dir dataset/rl_training/train

# This creates:
# dataset/rl_training/train/replica_0/{substrate.json, vnr_stream.json}
# ...
# dataset/rl_training/train/replica_4/{substrate.json, vnr_stream.json}

# 2. Generate held-out eval data (different random seed — never mixed with training)
python -m src.scripts.generate_datasets \
    --experiments rl \
    --num-vnrs 2000 \
    --substrate-nodes 60,100 \
    --vnr-min-nodes 2 \
    --vnr-max-nodes 10 \
    --num-replicas 2 \
    --seed 9999 \
    --output-dir dataset/rl_training/eval

# This creates:
# dataset/rl_training/eval/eval_0/{substrate.json, vnr_stream.json}
# dataset/rl_training/eval/eval_1/{substrate.json, vnr_stream.json}

# 3. Train PPO v2
python -m src.training.train_ppo_v2 \
    --train-dir dataset/rl_training/train \
    --eval-dir  dataset/rl_training/eval \
    --total-steps 2000000 \
    --window-size 50 \
    --hpso-iter 10 \
    --reward r2c_ac \
    --eval-every 5000 \
    --eval-episodes 10 \
    --run-name ppo_v2_r2c \
    --save-dir checkpoints

# 4. Monitor — open in browser at http://localhost:6006
tensorboard --logdir runs/
# Key panels to watch:
#   Train/PolicyLoss, Train/ValueLoss, Train/ExplainedVariance  ← optimizer health
#   Train/AcceptanceRate (smoothed)                             ← noisy training signal
#   Eval/AcceptanceRate, Eval/RevenueCostRatio                  ← definitive learning signal

# 5. Final evaluation against baseline (uses full HPSO iterations)
python -m src.training.evaluate \
    --checkpoint checkpoints/ppo_v2_r2c_best.pt \
    --eval-dir dataset/rl_training/eval \
    --hpso-iter 30 \
    --episodes 20
```

---

## Part 7 — Key Design Decisions & Rationale

### Why a fixed held-out eval set instead of evaluating on training data?

Evaluating on the same substrates used for training produces a reward curve that
rises because of **memorization**, not generalization. The curve looks identical to
genuine learning but the policy fails on any new substrate at inference time.

The held-out eval set (2 fixed substrates, never seen during rollout collection)
produces a stable, interpretable curve: because the inputs are identical every
eval run, any improvement in `Eval/AcceptanceRate` is genuine policy improvement.
Train reward curves are kept as a noisy secondary signal (use TensorBoard smoothing).

The practical rule: **deploy the checkpoint with the best `Eval/AcceptanceRate`,
not the one with the highest training reward**.

### Why not use a global reward across windows?

PPO requires finite-horizon episodes. A single "window" = one episode is the
natural boundary. The critic's role is to **approximate future returns beyond
the current window** — it looks at the substrate state (current utilisation)
and VNR queue to predict how much reward the policy will accumulate in
future windows. This is why substrate state quality (low utilisation) is
valuable to learn, even without a direct multi-window reward signal.

### Why not reset substrate between episodes?

Resetting would eliminate the covariate shift problem but make training
trivially easy (fresh substrate always has capacity). The whole point of the
scheduler is to make good decisions **on a partially-depleted substrate**. The
environment must carry substrate state forward.

### Why per-step + terminal reward?

Per-step reward (immediate R/C) gives dense gradient signal. Terminal reward
(AR + R/C for the window) gives the sparse but important throughput signal.
GAE propagates the terminal signal backward so early steps in the window
learn that their choices affect later embeddability.

### Why HPSO iterations = 10 during training?

Full HPSO (30 iterations × 20 particles = 600 evals) is expensive. Reducing
to 10 iterations during training speeds up rollout collection by ~3×. The
model learns ordering quality, not HPSO quality. Final evaluation should
use full HPSO.

### Why `use_batch_context = False` initially?

`BatchContextEncoder` (Transformer over all VNR embeddings) adds significant
compute overhead and requires the full VNR list to be encoded jointly. During
Phase 1 (simple reward), this extra cost is not justified. Enable it in Phase
2+ when the agent needs to reason about inter-VNR competition.

---

## Part 8 — File Summaries

### `src/scheduler/environment_v2.py` (~350 lines)

```
Classes:
  VNEDataset          — wraps one (substrate, vnr_stream) pair, manages windows
  VNEDatasetPool      — manages multiple datasets, provides round-robin episodes
  VNEEnvironmentV2    — gymnasium.Env, the main training environment

Key methods:
  VNEEnvironmentV2.reset()      — advance to next window, expire VNRs
  VNEEnvironmentV2.step(action) — embed/reject, compute reward, update substrate
  VNEEnvironmentV2._expire_vnrs()         — resource release
  VNEEnvironmentV2._commit_embedding()    — resource deduction
  VNEEnvironmentV2._compute_step_reward() — per-step reward
  VNEEnvironmentV2._compute_terminal_reward() — end-of-window reward
  VNEEnvironmentV2.episode_summary()      — metrics dict for logging
```

### `src/training/train_ppo_v2.py` (~450 lines)

```
Classes:
  PPOConfigV2         — all hyper-parameters (incl. eval_paths, eval_every, eval_episodes)
  PPOTrainerV2        — main trainer

Key methods:
  PPOTrainerV2.train()            — main loop: rollout → update → log → eval
  PPOTrainerV2._collect_rollout() — step env (train pool), store transitions
  PPOTrainerV2._compute_gae()     — GAE advantages + returns
  PPOTrainerV2._update()          — mini-batch PPO gradient steps; returns loss dict
  PPOTrainerV2.evaluate()         — greedy rollout on held-out eval pool, logs Eval/* metrics
  PPOTrainerV2.save() / load()    — checkpoint I/O; best checkpoint tracked by Eval/AcceptanceRate

CLI flags (all PPOConfigV2 fields):
  --train-dir, --eval-dir, --total-steps, --window-size,
  --hpso-iter, --reward, --reject-penalty,
  --reject-threshold, --eval-every, --eval-episodes,
  --run-name, --save-dir, --device,
  --load-checkpoint, ...
```

---

## Summary

The v2 pipeline resolves all training/inference gaps:

1. **Same substrate depletion dynamics**: substrate persists across windows,
   VNRs expire and return resources — exactly like inference.
2. **Same call pattern**: `model(substrate_pyg, remaining_vnr_pygs)` at every
   step within a window — exactly like `hpso_batch_scheduler.py`.
3. **Threshold-based rejection**: the network learns to produce negative scores for unembeddable VNRs; `hpso_batch_scheduler.py` rejects when `best_score < 0.0` — no architecture change, no extra parameters, training and inference are identical.
4. **Multi-dataset training**: pool of (substrate, stream) pairs prevents
   overfitting to a single topology.
5. **Reward alignment**: dense R/C per step + terminal AR/R/C bonus, all
   propagated via GAE so the critic learns to predict long-term substrate
   quality.
6. **Verified learning via held-out eval**: `Eval/AcceptanceRate` and `Eval/RevenueCostRatio` are logged every `eval_every` steps on a fixed pool never seen during training — the only TensorBoard curve that distinguishes genuine learning from memorization.
7. **Full TensorBoard visibility**: PPO loss terms (`PolicyLoss`, `ValueLoss`, `ExplainedVariance`, `ApproxKL`, `ClipFraction`, `Entropy`), training episode metrics, eval metrics, and substrate state — all on a single dashboard.
