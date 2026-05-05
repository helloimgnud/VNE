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

The v2 pipeline closes all these gaps **except rejection**, which is deferred to v3 after the
core priority-scoring is validated. See §Known Limitations.

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
        ├── replica_0/{substrate.json, vnr_stream.json}      ← held-out, NEVER used in training
        └── replica_1/{substrate.json, vnr_stream.json}
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

step(action):
    # action ∈ {0, …, n_remaining-1} — always an embed attempt, never a reject.
    # The agent's job is purely to rank: pick the VNR most likely to embed successfully.
    vnr = vnr_queue[action]
    result = hpso_embed(substrate, vnr)
    if result:
        commit resources to substrate
        push to active_embeddings with expiry = arrival + lifetime
        reward = accept_reward(vnr, result)
    else:
        reward = embed_fail_penalty(vnr)   # negative: chose a bad VNR to attempt
    pop vnr from queue

    done = (vnr_queue is empty)
    if done:
        terminal_reward = terminal_bonus(episode_stats)
        reward += terminal_reward
    return next_obs, reward, done, truncated, info
```

**No rejection in v2.** Every VNR in the window gets an embed attempt, ordered by
the policy's priority score. The policy learns to put embeddable VNRs first — it
receives a large negative reward when it picks a VNR that HPSO cannot embed.
This is sufficient signal to validate that the GNN is learning meaningful priority scores.

### 1.4 Threshold-Based Rejection

**Rejection is intentionally removed from v2.** The two hidden flaws that make
threshold-based rejection untrainable with plain PPO are documented here so they
are not re-introduced accidentally:

**Flaw 1 — Softmax translation invariance (math):** `Categorical(logits=scores)`
applies softmax internally. Softmax is translation-invariant: scores `[10, 5]`
and `[-100, -105]` produce identical action probabilities. PPO therefore provides
zero gradient signal to push absolute score magnitudes above or below any fixed
threshold like `0.0`. The network has no incentive to move scores into negative
territory. The threshold becomes meaningless.

**Flaw 2 — Missing `log_prob` (engineering):** Hardcoding `if best_score < threshold`
before `dist.sample()` means the rejection branch never produces a `log_prob` to
store in the PPO buffer. Without `log_prob`, no surrogate loss is computed for
that transition. The network cannot learn from the rejection penalty at all.

The correct fix (the "virtual fixed-logit trick" for v3) is to append a **constant**
`0.0` logit to the score vector before `Categorical`, making rejection a proper
sampled action. PPO is then forced to push real VNR scores negative relative to
that anchor to make rejection the argmax — giving the `0.0` threshold absolute
meaning at inference time. This will be implemented in **v3**, after v2 validates
that the GNN learns basic embeddability-aware priority scoring.

**v2 behaviour:** every VNR in the window is attempted via HPSO in the order the
policy ranks them. No VNR is skipped. The policy is rewarded for ordering
embeddable VNRs before unembeddable ones.

### 1.5 Multi-Substrate Support

> **Fix applied:** The original design embedded both train and eval pools inside a
> single `VNEEnvironmentV2` instance with a `set_mode()` toggle. This caused state
> corruption: calling `set_mode("eval")` invoked `_reset_substrate()`, permanently
> overwriting the training substrate's depletion state, active embeddings, and window
> pointer. When `set_mode("train")` was called afterward, the environment resumed
> from a clean slate — directly violating the §1.2 invariant that substrate state
> must carry forward between windows. The fix is to give each concern its own
> instance. `VNEEnvironmentV2` now accepts a single `dataset_paths` list and knows
> nothing about train vs. eval. The trainer owns two separate instances.

The environment maintains a **pool** of (substrate, vnr_stream) pairs. It accepts
a single `dataset_paths` list — the caller decides whether those paths are training
replicas or held-out eval replicas:

```python
class VNEEnvironmentV2(gymnasium.Env):
    def __init__(self, dataset_paths: list[dict], window_size: int = 50, ...):
        """
        Args:
            dataset_paths: list of {"substrate_path": ..., "vnr_path": ...} dicts.
                           Pass train replicas here for the training env;
                           pass eval replicas here for the eval env.
                           The env has no concept of "mode" — that distinction
                           lives entirely in the trainer.
        """
        self.pool = [load(d) for d in dataset_paths]
        self.pool_idx = 0
        self.window_idx = 0
        # ... rest of init ...

    def reset_to_start(self):
        """Reset pool_idx and window_idx to 0.
        Called by the trainer on self.eval_env before each evaluate() run
        so eval results are reproducible across calls.
        Never called on self.train_env — its state must never be disturbed.
        """
        self.pool_idx = 0
        self.window_idx = 0
        self._load_dataset(self.pool_idx)

    def reset(self, new_dataset: bool = False):
        if new_dataset:
            self.pool_idx = (self.pool_idx + 1) % len(self.pool)
            self._load_dataset(self.pool_idx)
        # Advance to next window in the current dataset
        ...
```

`set_mode()`, `train_pool`, `eval_pool`, `train_idx`, and `eval_idx` do not exist.

When a stream is exhausted (all windows processed), the next `reset()` picks
the next dataset from the pool via round-robin. The eval env is always rewound to
`pool_idx=0, window_idx=0` by `reset_to_start()` at the start of each `evaluate()`
call so results are reproducible across runs.

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
| HPSO accept | `α · rc_norm` | Reward embedding efficiency immediately |
| HPSO fail (attempt but fail) | `-β · revenue(vnr) / max_revenue` | Penalise choosing an unembeddable VNR |

Where:
- `rc_norm = clip(revenue/cost, 0, 1)` — R/C normalised to [0,1]
- `α = 0.6`, `β = 0.3` (β should be large enough to make bad ordering painful)
- All rewards bounded to `(-1, 1)` for stable training

**No rejection penalty in v2** — there is no rejection action. The only negative
signal comes from HPSO failures, which directly punishes the policy for poor ordering.

**Terminal reward `r_terminal`** (end-of-window):

```python
n_total    = n_accepted + n_failed          # no rejected bucket in v2
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

### 2.2 Curriculum

Start with `reward_mode = "simple"` (binary: +1 accept, -1 fail) to validate
that the policy learns at all. Then switch:

```
Steps 0 → 50k:   simple     (binary: +1 accept, -1 HPSO fail)
Steps 50k → 150k: r2c        (R/C per accept + fail penalty)
Steps 150k+:      r2c_ac     (R/C + terminal AR/R/C bonus)
```

---

## Part 3 — `train_ppo_v2.py`

### 3.1 What the Trainer Does (and Doesn't Do)

**Trainer responsibilities:**
- Collect rollouts from `self.train_env` (training pool only — never interrupted)
- Run GAE to compute advantages
- Run PPO mini-batch updates on `GNNActorCritic`
- Every `eval_every` steps: run `evaluate()` on `self.eval_env` (greedy, no grad)
- Log all metrics to TensorBoard
- Save checkpoints (best checkpoint tracked by `Eval/AcceptanceRate`)

**Trainer does NOT:**
- Generate data (environment does that)
- Manage VNR expiry (environment does that)
- Call HPSO directly (environment does that)

The trainer instantiates **two separate environment instances** so that
`self.train_env`'s substrate depletion state is physically frozen while
`evaluate()` runs on `self.eval_env`:

```python
class PPOTrainerV2:
    def __init__(self, cfg: PPOConfigV2):
        # Training env — substrate depletion carries forward uninterrupted.
        # This instance is ONLY ever touched inside _collect_rollout().
        self.train_env = VNEEnvironmentV2(
            dataset_paths=cfg.dataset_paths,
            window_size=cfg.window_size,
            ...
        )
        # Eval env — completely separate instance.
        # self.train_env's state is frozen while this runs.
        self.eval_env = VNEEnvironmentV2(
            dataset_paths=cfg.eval_paths,
            window_size=cfg.window_size,
            ...
        )
        self.model = GNNActorCriticV2(...)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        ...
```

### 3.2 Rollout Collection (Matching Inference)

The rollout loop must exactly mirror `hpso_batch_scheduler.py`:

```python
# Training rollout (1 window = 1 episode):
obs, _ = self.train_env.reset()
while not done:
    sub_pyg  = obs["substrate"]          # current (depleted) substrate
    vnr_pygs = obs["vnr_list"]           # remaining VNRs in window

    # Forward pass — scores [B], value [1]
    dist, value = model(obs)             # dist = Categorical(logits=scores)

    action   = dist.sample()             # ∈ {0, …, B-1} — index of VNR to attempt
    log_prob = dist.log_prob(action)

    obs, reward, done, _, info = self.train_env.step(action.item())

    # Every step has a valid log_prob — no missing gradients
    buffer.push(obs, action, log_prob, value, reward, done)
```

At inference time, replace `dist.sample()` with `scores.argmax()` — deterministic
greedy ordering. Embed all VNRs in score order (highest first) until the window
is exhausted. No threshold check needed.

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
    reward_mode: str = "simple"      # simple | r2c | r2c_ac
    embed_fail_penalty: float = 0.3  # penalty weight β for HPSO failures
    terminal_ar_weight: float = 0.4
    terminal_rc_weight: float = 0.6
    # Note: no reject_penalty or reject_threshold — rejection not implemented in v2

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
| `Train/NFailed` | HPSO calls that failed — primary ordering signal | Decreasing as policy improves |
| `Train/AvgScoreAccepted` | Mean logit score of VNRs that HPSO accepted | Rising relative to failed |
| `Train/AvgScoreFailed` | Mean logit score of VNRs where HPSO failed | Diverging downward from accepted |

**Key diagnostic:** `Train/AvgScoreAccepted` − `Train/AvgScoreFailed` is the score
margin. If this gap widens over training, the GNN is learning embeddability-aware
representations. If it stays near zero, scores are random and no learning is happening.

**Note:** Training episode metrics are noisy because each window comes from a different
(substrate, VNR stream) pair. Look at rolling mean trends over ~50 episodes using
TensorBoard's smoothing slider, not individual data points.

---

#### Group 3 — Held-Out Evaluation Metrics (logged every `eval_every` timesteps)

**This is the definitive measure of whether PPO is actually learning.**

The eval env is a separate instance loaded from fixed held-out replicas never seen
during training. Running evaluation on the same replicas every `eval_every` steps
produces a stable, interpretable curve.

> **Fix applied:** The original `evaluate()` called `self.env.set_mode("eval")`
> and `self.env.set_mode("train")` around the eval loop. `set_mode("eval")`
> internally called `_reset_substrate()`, overwriting the training substrate in
> memory and destroying the depletion continuity that §1.2 requires. The fix is
> to run evaluation entirely on `self.eval_env` — a separate instance that is
> physically independent from `self.train_env`. No mode switching, no shared
> state, no discontinuity.

```python
def evaluate(self, n_episodes: int = 10) -> dict:
    """Greedy rollout on the held-out eval env.

    self.train_env is never touched here — its substrate depletion state,
    active embeddings, and window pointer are frozen for the duration of
    this call.
    """
    self.model.eval()
    # Rewind eval env to replica 0 / window 0 for reproducibility.
    # This has zero effect on self.train_env.
    self.eval_env.reset_to_start()

    all_ars, all_rcs, all_rewards = [], [], []

    for ep in range(n_episodes):
        obs, _ = self.eval_env.reset()
        ep_reward = 0.0
        done = False

        with torch.no_grad():
            while not done:
                dist, value = self.model(obs)
                # Greedy: always pick the highest-scored VNR to attempt
                action = dist.logits.argmax()
                obs, reward, done, _, info = self.eval_env.step(action.item())
                ep_reward += reward

        summary = self.eval_env.episode_summary()
        all_ars.append(summary["acceptance_rate"])
        all_rcs.append(summary["revenue_cost_ratio"])
        all_rewards.append(ep_reward)

    # self.train_env state is exactly as it was before this call.
    self.model.train()

    metrics = {
        "Eval/AcceptanceRate":   mean(all_ars),
        "Eval/RevenueCostRatio": mean(all_rcs),
        "Eval/EpisodeReward":    mean(all_rewards),
        "Eval/AR_std":           std(all_ars),    # stability indicator
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

`GNNActorCriticV2` wraps `VNRScheduler` with a **value head only**. The architecture
is minimal — no reject head, no extra parameters beyond what is needed to produce a
score distribution over VNRs and a state value estimate.

```python
class GNNActorCriticV2(nn.Module):
    def __init__(self, scheduler, substrate_emb_dim=128):
        super().__init__()
        self.scheduler  = scheduler
        self.value_head = nn.Linear(substrate_emb_dim, 1)

    def forward(self, obs) -> tuple[Categorical, torch.Tensor]:
        sub_data = obs["substrate"]
        vnr_list = obs["vnr_list"]

        # VNR scores [B] — higher = higher priority to embed
        scores = self.scheduler(sub_data, vnr_list)

        # Substrate embedding for value estimate
        h_s   = self.scheduler.substrate_encoder(sub_data)   # [1, D]
        value = self.value_head(h_s).squeeze(-1)              # scalar

        # Action distribution over B VNRs.
        # Softmax is applied internally — scores are relative rankings.
        # The policy learns that choosing a VNR with poor substrate fit → negative reward.
        dist = Categorical(logits=scores)
        return dist, value
```

**What the network learns:** after training, the highest-scored VNR should correspond
to the one most likely to be successfully embedded by HPSO given the current substrate
state. This is the sole objective of v2 — pure embeddability-aware priority scoring.

---

## Part 5 — Implementation Checklist

### `environment_v2.py`

- [ ] `VNEDataset` class: loads (substrate, vnr_stream) from JSON, splits into windows
- [ ] `VNEDatasetPool` class: manages multiple datasets, round-robin or random
- [ ] `VNEEnvironmentV2(gymnasium.Env)`:
  - [ ] `__init__`: accepts a single `dataset_paths` list (no train/eval duality, no `set_mode`)
  - [ ] `reset_to_start()`: resets `pool_idx=0` and `window_idx=0`; called by trainer on `eval_env` before each `evaluate()` run; never called on `train_env`
  - [ ] `reset()`: expire VNRs, advance window in active pool, build `vnr_queue`
  - [ ] `step(action)`: embed VNR at action index, update substrate, compute reward (accept or fail penalty)
  - [ ] `_expire_vnrs()`: resource release on expiry
  - [ ] `_commit_embedding()`: resource deduction on accept
  - [ ] `_load_dataset(idx)`: load substrate + vnr_stream for pool entry `idx` into working state
  - [ ] `_get_obs()`: build PyG Data objects
  - [ ] `episode_summary()`: returns dict of AR, R/C, utilisation for trainer to log
  - [ ] `_compute_reward()`: pluggable by `reward_mode`
- [ ] On-the-fly mode: fallback when no dataset paths provided
- [ ] Window cycling: when stream exhausted, next `reset()` tries next dataset in pool via round-robin

### `train_ppo_v2.py`

- [ ] `PPOConfigV2` dataclass
- [ ] `PPOTrainerV2` class:
  - [ ] `__init__`: build `GNNActorCriticV2`, two separate `VNEEnvironmentV2` instances (`self.train_env` for rollouts, `self.eval_env` for evaluation), optimizer
  - [ ] `_collect_rollout(n_steps)`: uses `self.train_env` only; sample action → env.step → store (obs, action, log_prob, value, reward, done); every step has a valid log_prob
  - [ ] `_compute_gae()`: unchanged from v1
  - [ ] `_update()`: PPO with KL / clip fraction logging; returns loss dict for TensorBoard
  - [ ] `train()`: main loop — rollout → update → log → eval every `eval_every` steps
  - [ ] `evaluate(n_episodes)`: greedy rollout on `self.eval_env` only; calls `eval_env.reset_to_start()` for reproducibility; `self.train_env` is never touched
  - [ ] best-checkpoint tracking: save whenever `Eval/AcceptanceRate` improves
  - [ ] `save()` / `load()`
- [ ] CLI entry point with all `PPOConfigV2` fields exposed
- [ ] TensorBoard writer with all metrics from §3.4

### `generate_datasets.py` — `rl` experiment branch

> **Fix applied:** The original `elif exp_name == 'rl':` branch called
> `generator.generate_rl_training_dataset(seed=args.base_seed)` exactly once,
> ignoring `--num-replicas`, `--substrate-nodes`, `--num-vnrs`, `--vnr-min-nodes`,
> and `--vnr-max-nodes`. The Part 6 bash commands pass `--num-replicas 5` expecting
> `replica_0/` through `replica_4/` to be created; the old code created none of
> them, causing the trainer to crash on startup. The fix is a loop over
> `args.num_replicas` that forwards all parsed CLI arguments.

The corrected `rl` branch:

```python
elif exp_name == 'rl':
    replicas_meta = []
    for i in range(args.num_replicas):
        replica_seed = args.base_seed + i   # distinct seed per replica
        replica_meta = generator.generate_rl_training_dataset(
            replica_idx=i,
            substrate_nodes_range=substrate_nodes_range,
            num_vnrs_range=num_vnrs_range,
            vnr_min_nodes=vnr_min_nodes,
            vnr_max_nodes=vnr_max_nodes,
            seed=replica_seed,
        )
        replicas_meta.append(replica_meta)
        print(f"   ✓ rl replica_{i} generated (seed={replica_seed})")
    generated[exp_name] = {"replicas": replicas_meta}
```

Required update to `DatasetGenerator.generate_rl_training_dataset()` signature:

```python
# Before:
def generate_rl_training_dataset(self, seed: int = 42) -> dict: ...

# After:
def generate_rl_training_dataset(
    self,
    replica_idx: int = 0,
    substrate_nodes_range: tuple[int, int] = (80, 80),
    num_vnrs_range: tuple[int, int] = (1000, 1000),
    vnr_min_nodes: int = 2,
    vnr_max_nodes: int = 8,
    seed: int = 42,
) -> dict:
    """Generate one (substrate, vnr_stream) replica.
    Output path: {base_dir}/rl/replica_{replica_idx}/
    """
    ...
```

### Inference compatibility (`hpso_batch_scheduler.py`)

For v2, inference is simpler than before — just take the argmax score and embed:

```python
scores = scheduler.predict(sub_data, rem_vnrs)   # [B]
best_idx = scores.argmax().item()
# embed vnr_list[best_idx] via HPSO, pop from queue, repeat
```

No threshold check. All VNRs in the window are attempted in priority order.
Rejection will be added in v3 once priority scoring is validated.

---

## Part 6 — Data Preparation Workflow

> **Fix applied:** The `--seed` flag below is `--base-seed` in the actual CLI
> (see `generate_datasets.py`). The eval command uses `--base-seed 9999` to
> produce a statistically independent held-out pool.

```bash
# 1. Generate training data (5 replicas, seed 42..46)
python -m src.scripts.generate_datasets \
    --experiments rl \
    --num-vnrs 2000 \
    --substrate-nodes 60,100 \
    --vnr-min-nodes 2 \
    --vnr-max-nodes 10 \
    --num-replicas 5 \
    --base-seed 42 \
    --output-dir dataset/rl_training/train \
    --force

# This creates:
# dataset/rl_training/train/rl/replica_0/{substrate.json, vnr_stream.json}
# dataset/rl_training/train/rl/replica_1/{substrate.json, vnr_stream.json}
# dataset/rl_training/train/rl/replica_2/{substrate.json, vnr_stream.json}
# dataset/rl_training/train/rl/replica_3/{substrate.json, vnr_stream.json}
# dataset/rl_training/train/rl/replica_4/{substrate.json, vnr_stream.json}

# 2. Generate held-out eval data (2 replicas, different seed — never mixed with training)
python -m src.scripts.generate_datasets \
    --experiments rl \
    --num-vnrs 2000 \
    --substrate-nodes 60,100 \
    --vnr-min-nodes 2 \
    --vnr-max-nodes 10 \
    --num-replicas 2 \
    --base-seed 9999 \
    --output-dir dataset/rl_training/eval \
    --force

# This creates:
# dataset/rl_training/eval/rl/replica_0/{substrate.json, vnr_stream.json}
# dataset/rl_training/eval/rl/replica_1/{substrate.json, vnr_stream.json}

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

### Why two separate environment instances instead of `set_mode()`?

`set_mode("eval")` had to call `_reset_substrate()` to give the eval pool a clean
slate. That call overwrote the live training substrate in memory, destroying the
depletion continuity that is the whole point of v2. Two separate instances have
physically separate substrate objects — `self.train_env` is provably untouched
while `evaluate()` runs. This is standard RL practice (e.g., Stable-Baselines3
`EvalCallback` creates a separate `eval_env` for the same reason).

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
  VNEEnvironmentV2.__init__(dataset_paths)  — single list, no train/eval duality
  VNEEnvironmentV2.reset_to_start()         — rewind to replica 0 / window 0 (eval use only)
  VNEEnvironmentV2.reset()                  — advance to next window, expire VNRs
  VNEEnvironmentV2.step(action)             — embed VNR, compute reward, update substrate
  VNEEnvironmentV2._expire_vnrs()           — resource release
  VNEEnvironmentV2._commit_embedding()      — resource deduction
  VNEEnvironmentV2._load_dataset(idx)       — load pool entry into working state
  VNEEnvironmentV2._compute_step_reward()   — per-step reward
  VNEEnvironmentV2._compute_terminal_reward() — end-of-window reward
  VNEEnvironmentV2.episode_summary()        — metrics dict for logging
```

### `src/training/train_ppo_v2.py` (~450 lines)

```
Classes:
  PPOConfigV2         — all hyper-parameters (incl. eval_paths, eval_every, eval_episodes)
  PPOTrainerV2        — main trainer

Key methods:
  PPOTrainerV2.__init__()         — builds self.train_env + self.eval_env as separate instances
  PPOTrainerV2.train()            — main loop: rollout → update → log → eval
  PPOTrainerV2._collect_rollout() — step self.train_env only, store transitions
  PPOTrainerV2._compute_gae()     — GAE advantages + returns
  PPOTrainerV2._update()          — mini-batch PPO gradient steps; returns loss dict
  PPOTrainerV2.evaluate()         — greedy rollout on self.eval_env; self.train_env untouched
  PPOTrainerV2.save() / load()    — checkpoint I/O; best checkpoint tracked by Eval/AcceptanceRate

CLI flags (all PPOConfigV2 fields):
  --train-dir, --eval-dir, --total-steps, --window-size,
  --hpso-iter, --reward, --embed-fail-penalty,
  --eval-every, --eval-episodes,
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
3. **Threshold-based rejection deferred to v3**: the "virtual fixed-logit trick"
   will be added once v2 validates that the GNN learns basic priority scoring.
4. **Multi-dataset training**: pool of (substrate, stream) pairs prevents
   overfitting to a single topology.
5. **Reward alignment**: dense R/C per step + terminal AR/R/C bonus, all
   propagated via GAE so the critic learns to predict long-term substrate
   quality.
6. **Verified learning via held-out eval**: `Eval/AcceptanceRate` and
   `Eval/RevenueCostRatio` are logged every `eval_every` steps on a fixed pool
   never seen during training — the only TensorBoard curve that distinguishes
   genuine learning from memorization. Guaranteed by two separate env instances:
   `self.train_env` state is physically frozen while `evaluate()` runs on
   `self.eval_env`.
7. **Full TensorBoard visibility**: PPO loss terms (`PolicyLoss`, `ValueLoss`,
   `ExplainedVariance`, `ApproxKL`, `ClipFraction`, `Entropy`), training episode
   metrics, eval metrics, and substrate state — all on a single dashboard.
