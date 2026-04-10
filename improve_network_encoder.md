# Improve Network Encoder — Comprehensive Plan

## 1. What's Actually Happening Now (Diagnosis)

### 1.1 Why `AR=100%` at Epoch 100 Is Misleading

When you see:
```
Ep 100/2000 | loss=+0.6437 | reward=10.000 | AR=100% | R/C=1.000 | t=151s
```

**This does NOT mean the model learned anything useful yet.** Here is why:

#### The substrate is too easy for the VNRs

The default environment uses:
- Substrate: 50 nodes, CPU ∈ [50, 100], BW ∈ [50, 200]
- VNR batch: 10 VNRs, each with ~4 nodes, CPU demand ∈ [5, 30], BW demand ∈ [5, 50]

Total VNR CPU demand ≈ 10 × 4 × 17 = **680 CPU**  
Total substrate CPU capacity ≈ 50 × 75 = **3750 CPU**

The substrate has **5× more capacity than the batch needs**. HPSO will accept almost everything
regardless of ordering — so the random or learned order makes no difference yet.
`AR=100%` is just "HPSO works on an easy instance," not "the GNN learned a good policy."

#### R/C = 1.000 is a known bug

In `rewards.py`, `_cost()` is defined as:
```python
def _cost(vnr):
    return _revenue(vnr) + 1e-6   # cost == revenue!
```

And `_revenue()` is:
```python
def _revenue(vnr):
    return sum(cpu) + sum(bw)     # demand-based
```

Since `cost = revenue`, the ratio `revenue / cost ≈ 1.000` **always**, no matter what the
model does. This is a tautology — it provides **zero learning signal** for the R/C term.

Similarly in `episode_summary()`:
```python
total_cost = sum(cost_of_vnr(v) for v, _, _ in self.accepted)
# cost_of_vnr() without mapping/link_paths also returns = sum(cpu) + sum(bw)
# → same as revenue → R/C ≈ 1.0 always
```

#### What the logs actually tell you

| Log field | What it measures | Is it meaningful now? |
|-----------|-----------------|----------------------|
| `reward=10.000` | Sum of per-step rewards (each = +1.0 for simple mode, ×10 VNRs) | ✓ correct math, but trivially maximised |
| `AR=100%` | All VNRs accepted | ✗ substrate too easy, ordering irrelevant |
| `R/C=1.000` | Revenue/Cost ratio | ✗ **bug**: cost == revenue always → always 1.0 |
| `loss=+0.6437` | REINFORCE policy gradient loss | ✓ positive = some variance, model is learning |

### 1.2 Will it get better after epoch 100?

On the current environment setup: **barely**. Once all VNRs are accepted at episode 1, the
gradient signal coming from `simple` reward (+1/−0.5) becomes pure noise — every ordering gives
the same total reward (+10), so the advantages after baseline subtraction are all ≈ 0.
The model will keep training but will not converge to anything meaningful.

**The model needs harder problems to learn from.**

---

## 2. Root Cause Summary

| # | Problem | Location | Impact |
|---|---------|----------|--------|
| B1 | `cost == revenue` tautology → R/C always ≈ 1.0 | `rewards.py`, `episode_summary()` | R/C metric meaningless |
| B2 | Substrate too large / VNRs too small → AR=100% trivially | `generate_data.py` defaults | No gradient signal from rejections |
| B3 | No progressive difficulty — same easy env forever | `train_reinforce.py` | Model stops learning early |
| B4 | `reward="simple"` is constant (+10) when AR=100% | `rewards.py` | Advantages → 0, zero learning |
| B5 | No real deployment feedback — simulated fresh substrate each episode | `environment.py` | Model never sees degraded substrate |

---

## 3. Your Idea — "Progressive Deployment Curriculum"

Your intuition is exactly right and maps to a well-known technique called
**curriculum learning with environment progression** combined with
**online/continual RL**.

The core idea:

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Progressive Deployment Loop                       │
│                                                                     │
│  1. Start with fresh substrate                                      │
│  2. Agent learns to order VNRs → HPSO embeds them                  │
│  3. When AR ≥ threshold for N consecutive episodes:                 │
│     a) COMMIT the accepted VNRs to the real substrate               │
│        (resources permanently consumed)                             │
│     b) Generate a NEW HARDER batch of VNRs                         │
│     c) Continue training on the now-degraded substrate              │
│  4. As substrate fills up:                                          │
│     - VNRs that were trivially accepted before now get rejected     │
│     - The model must learn genuine ordering intelligence             │
│  5. When substrate is too full (AR < floor),                        │
│     release some VNRs (lifetime expiry) OR reset to a new substrate │
└─────────────────────────────────────────────────────────────────────┘
```

This creates a **natural difficulty progression**:
- Episode 1–N: substrate nearly full, easy → model learns basics
- Episode N+: substrate partially consumed → ordering decisions matter
- Episode 2N+: substrate congested → rejections happen → hard ordering needed

---

## 4. Fix Plan — What to Change and Why

### Fix B1: Real Cost Calculation (High Priority)

**Problem:** `cost = revenue` always → R/C = 1.0, useless metric.

**Fix:** Use `cost_of_embedding()` (already in `eval.py`) which uses actual substrate path lengths to compute real cost. This requires passing `substrate` into the cost computation.

In `rewards.py`, change `_cost()`:
```python
# CURRENT (wrong):
def _cost(vnr):
    return _revenue(vnr) + 1e-6   # tautology

# FIXED:
def _cost_embedding(vnr, mapping, link_paths, substrate):
    from src.evaluation.eval import cost_of_embedding
    return cost_of_embedding(mapping, link_paths, vnr, substrate)
```

The `accepted` list in `environment.py` already stores `(vnr, mapping, link_paths)`.
We need to also store a reference to the **substrate at time of embedding** or pass it through.

In `episode_summary()`:
```python
# CURRENT (wrong):
total_cost = sum(cost_of_vnr(v) for v, _, _ in self.accepted)

# FIXED: use actual embedding cost
total_cost = sum(
    cost_of_embedding(mapping, link_paths, vnr, self.accepted_substrate_snapshot)
    for vnr, mapping, link_paths in self.accepted
)
```

### Fix B2 + B3 + B5: Progressive Deployment Curriculum

This is the main architectural addition. The design requires a new
`ProgressiveDeploymentCurriculum` class that wraps the environment.

**Trigger condition for promotion (committing to real substrate):**
- AR ≥ `promote_ar_threshold` (e.g. 0.85) for `promote_window` (e.g. 5) consecutive episodes

**On promotion:**
- Apply all accepted VNR mappings permanently to the `live_substrate`
- Generate a new batch of VNRs (optionally larger / higher demand)
- Continue training — model now sees a harder, partially-consumed substrate

**VNR lifetime / reset:**
- Track how many batches each VNR has been committed for
- If substrate AR drops below `floor_ar` (e.g. 0.2), trigger a partial release:
  - Remove VNRs that have exceeded their lifetime
  - This models real-world VNR departures

### Fix B4: Better Reward for Harder Scenarios

Once the substrate is partially filled:
- `simple` mode (+1/−0.5) becomes meaningful — rejections actually happen
- Switch to `revenue` mode to add R/C signal
- Switch to `longterm` when training is stable

---

## 5. Detailed Implementation Plan

### 5.1 Files to Create

```
src/training/
├── progressive_env.py      ← main new module
└── train_progressive.py    ← training script using progressive curriculum
```

### 5.2 `progressive_env.py` — Design

```
ProgressiveDeploymentEnv
│
├── live_substrate : networkx.Graph          (the real network being filled)
├── committed_vnrs : list[(vnr, mapping, link_paths, episode_id)]
├── episode_history : list[float]            (AR per episode, for window tracking)
│
├── reset()
│   └── copies live_substrate as working substrate
│       generates fresh VNR batch
│
├── step(action)
│   └── same as VNEOrderingEnv
│
├── maybe_promote()     ← called after each episode
│   ├── check: mean(last N ARs) >= promote_threshold?
│   ├── if yes: commit accepted VNRs to live_substrate
│   │           expire old VNRs by lifetime
│   │           level_up(): increase batch_size or VNR demand
│   └── if no: continue on same live_substrate
│
├── _commit_to_live(accepted)
│   ├── for each (vnr, mapping, link_paths): reserve resources on live_substrate
│   └── append to committed_vnrs with current episode_id
│
└── _expire_old_vnrs(current_episode)
    ├── find committed_vnrs where episode_id + lifetime <= current_episode
    └── release those resources from live_substrate
```

### 5.3 `ProgressiveDeploymentConfig` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `promote_ar_threshold` | 0.85 | AR (averaged over window) to trigger promotion |
| `promote_window` | 5 | Number of consecutive episodes AR must hold |
| `floor_ar` | 0.20 | If AR drops below this, trigger early release |
| `vnr_lifetime_episodes` | 20 | Episodes before a committed VNR expires |
| `level_up_batch_delta` | +2 | Add N more VNRs per batch on each level-up |
| `level_up_demand_scale` | 1.15 | Scale VNR CPU/BW demands by this on level-up |
| `max_batch_size` | 30 | Cap on batch size |
| `max_demand_scale` | 3.0 | Cap on demand scaling |
| `release_fraction` | 0.3 | Fraction of committed VNRs to release via lifetime |

### 5.4 Training script `train_progressive.py`

Wraps `ProgressiveDeploymentEnv` with the REINFORCE or PPO loops.
Key differences from `train_reinforce.py`:
- Calls `env.maybe_promote()` after each episode
- Logs: `level`, `live_substrate_utilisation`, `committed_vnr_count`
- Saves checkpoint at each level-up

### 5.5 Metric and Logging Fixes

In `train_reinforce.py` and PPO trainer, add:
- **Substrate utilisation**: `used_cpu / total_cpu` and `used_bw / total_bw`
  → tracks how congested the live substrate is
- **Rolling AR (window=10)**: moving average to smooth noise
- **Real R/C**: fix the cost bug (B1) so the ratio is meaningful

---

## 6. Implementation Phases

### Phase A — Bug Fixes (do first, required for all later phases)

| Task | File | Change |
|------|------|--------|
| A1 Fix R/C: pass substrate into cost | `rewards.py` | Add substrate param to `longterm` mode; use `cost_of_embedding()` |
| A2 Fix episode_summary R/C | `environment.py` | Use embedding cost, not demand cost |
| A3 Add substrate utilisation logging | `train_reinforce.py` | Log `used_cpu%` and `used_bw%` per episode |
| A4 Harden default env | `generate_data.py` | Tighter defaults: CPU ∈ [10,40], substrate CPU ∈ [30,80] |

**Gate:** After A1–A4, you should see R/C fluctuating (not locked at 1.0) and AR dropping occasionally below 100%.

### Phase B — Progressive Deployment Environment (core feature)

| Task | File | Change |
|------|------|--------|
| B1 Create `ProgressiveDeploymentEnv` | `src/training/progressive_env.py` | New class |
| B2 Create `train_progressive.py` | `src/training/train_progressive.py` | New script |
| B3 Add VNR lifetime tracking | `progressive_env.py` | `committed_vnrs` list + `_expire_old_vnrs()` |
| B4 Add level-up logic | `progressive_env.py` | `_level_up()`: increase demand or batch size |

**Gate:** Loss should NOT converge to zero after 100 episodes. AR should be ≤ 85% on most episodes as substrate fills.

### Phase C — Improved Reward Signal

| Task | File | Change |
|------|------|--------|
| C1 Add `congestion_aware` reward mode | `rewards.py` | Reward = R/C × (1 + substrate_fill_ratio) |
| C2 Add `rejection_penalty_scaled` mode | `rewards.py` | Penalises rejections proportional to VNR revenue × fill_ratio |
| C3 Auto-switch reward mode by level | `progressive_env.py` | Level 1-3: simple; Level 4-6: revenue; Level 7+: longterm |

### Phase D — Evaluation Improvements

| Task | File | Change |
|------|------|--------|
| D1 Add baseline comparison on depleted substrates | `evaluate.py` | Test both GNN-order and revenue-sort on filled substrate |
| D2 Add level-tracking in reports | `evaluate.py` | Report AR per difficulty level |
| D3 Separate training AR from held-out AR | `train_progressive.py` | Every 100 eps, run evaluation on a fresh held-out substrate |

---

## 7. Expected Training Curve (After Fixes)

```
AR%
100 ┤ ██ (easy - substrate fresh)
 90 ┤ ████
 80 ┤    ████ (first promotion: substrate partially filled)
 70 ┤        ████
 60 ┤            ████ (model learns ordering under pressure)
 50 ┤                ████
 70 ┤                    ████ (model improves on harder substrate)
 80 ┤                        ████ (second promotion: substrate more filled)
 60 ┤                            ████
    └──────────────────────────────────────────────────→ episode

R/C
1.8 ┤                ████████
1.5 ┤        ████████
1.2 ┤ ████████
1.0 ┤ (bug-fixed: no longer locked at 1.0)
    └──────────────────────────────────────────────────→ episode
```

The sawtooth AR pattern is correct and expected: each promotion makes the problem harder → AR drops → model learns → AR recovers → promotion → repeat.

---

## 8. Key Design Decisions

| Decision | Rationale |
|----------|----------|
| Commit entire accepted batch (not partial) | Simpler bookkeeping; consistent with batch-processing model |
| Expire by episode count (not real time) | Reproducible curriculum progression; decoupled from wall clock |
| Level up on batch_size first, then demand | Easier to control; batch_size increase is coarser signal |
| Keep `live_substrate` mutable across episodes | Core of the idea — substrate gets harder naturally |
| Always copy `live_substrate` at episode start | Agent sees fresh copy; gradients don't depend on other episodes |
| Add held-out evaluation substrate | Prevents overfitting to a single filled network state |
| Switch reward mode by level automatically | Avoids manual tuning; simple signal when easy, complex when hard |

---

## 9 Checklist Summary

```
Phase A — Bug Fixes
[ ] A1  Fix R/C: use cost_of_embedding() in rewards.py
[ ] A2  Fix episode_summary to use real embedding cost
[ ] A3  Add substrate utilisation % to training logs
[ ] A4  Tighten default substrate/VNR parameters

Phase B — Progressive Deployment
[ ] B1  Create src/training/progressive_env.py
[ ] B2  Create src/training/train_progressive.py
[ ] B3  VNR lifetime + expiry logic
[ ] B4  Level-up: batch_size growth + demand scaling

Phase C — Reward Signal
[ ] C1  congestion_aware reward mode
[ ] C2  rejection_penalty_scaled reward mode
[ ] C3  Auto-switch reward by level

Phase D — Evaluation
[ ] D1  Held-out AR on filled substrate
[ ] D2  Level-tracking in evaluation report
[ ] D3  Separate train vs eval AR logging
```
