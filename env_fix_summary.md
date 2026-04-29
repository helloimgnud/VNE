# Environment Fix Summary: Stateful Substrate Wrapper

**Date:** 2026-04-29  
**Fix Class:** Covariate Shift / Train-Inference Distribution Mismatch  
**Status:** Implemented

---

## Problem

During inference, `BatchedVNRSimulator` maintains a **single persistent substrate**
that accumulates VNR resource consumption across all time windows. CPU utilisation
floats between 0%-80% throughout a simulation run.

During training, `VNEOrderingEnv.reset()` calls `substrate_fn()`, which always
generates a **brand-new substrate at 0% utilisation**. The agent therefore trains
exclusively on the easiest substrate state and has never seen a 40-70% depleted
substrate -- the typical state it encounters at inference time.

This is **covariate shift**: P_train(s) = delta(empty substrate) while P_inf(s)
is a broad distribution over utilisation levels.

---

## Root Cause (Confirmed in Code)

```
VNEOrderingEnv.reset()          <- environment.py:134
  --- raw_substrate = self.substrate_fn()   <- always generates fresh graph
  --- self.substrate = copy_substrate(raw_substrate)  <- 0% util every episode
```

`substrate_fn` is a closure from `make_substrate_fn()` in `generate_data.py`.
Every call returns a new random substrate with all resources at full capacity.

`BatchedVNRSimulator._embed_batch()` (simulator.py:352) commits accepted VNRs
to `self.substrate` in-place and only releases them via `process_departures()` when
their `lifetime` expires -- creating a persistent load that training never replicates.

---

## Solution

A **`StatefulSubstrateWrapper`** added entirely within `src/training/` intercepts
`env.substrate_fn` and replaces it with a closure returning a copy of a persistent,
depleting `live_substrate`. The wrapper:

1. **Patches** `env.substrate_fn` at construction -> `env.reset()` now receives a
   copy of `live_substrate` instead of a fresh graph.
2. **Commits** accepted VNRs to `live_substrate` after each `env.step()` via `on_step()`.
3. **Expires** committed VNRs after K episodes via `on_episode_start()`,
   releasing their resources -- mirroring `BatchedVNRSimulator.process_departures()`.
4. **Overflow-resets** `live_substrate` if CPU utilisation exceeds `overflow_cpu_threshold`,
   preventing the agent from getting stuck in an unembeddable state.
5. **Warm-up** period: no commits for the first `warmup_episodes` episodes,
   letting the agent develop a basic policy before substrate hardening.

The wrapper is **disabled by default** (`stateful_substrate=False`) -- all existing
training runs are byte-for-byte unchanged.

---

## Files Changed

### New file
```
src/training/stateful_substrate.py
```
Contains:
- `StatefulSubstrateConfig` -- dataclass with all tuning parameters
- `VNRRecord` -- tracks one committed VNR (mapping, paths, commit/expiry episode)
- `_consume_resources(substrate, record)` -- deducts CPU/BW on commit
- `_release_resources(substrate, record)` -- restores CPU/BW on expiry
- `StatefulSubstrateWrapper` -- main class with `on_episode_start()`, `on_step()`, `get_live_util()`

### Modified file
```
src/training/train_ppo.py
```

| Change | Location |
|--------|----------|
| 5 new `PPOConfig` fields (all with safe defaults) | After `load_checkpoint` field |
| Wrapper initialisation block | After `VNEOrderingEnv(...)` construction, before first `env.reset()` |
| `on_episode_start()` Hook 1A (empty vnr_list path) | `_collect_rollout`: before reset when `obs["vnr_list"]` is empty |
| `on_step()` Hook 2 | `_collect_rollout`: immediately after `env.step()` |
| `on_episode_start()` Hook 1B (done=True path) | `_collect_rollout`: before reset when `done=True` |
| TensorBoard `StatefulSubstrate/` metrics | Inside `if done:` block |
| 4 CLI arguments in `_build_parser()` | After `--load-checkpoint` |
| CLI -> `PPOConfig` wiring | `__main__` block |

### Unchanged files (constraint respected)
```
src/scheduler/environment.py
src/algorithms/fast_hpso.py
src/algorithms/hpso_batch_scheduler.py
src/simulation/simulator.py
src/utils/graph_utils.py
```

---

## Key Correction vs. Original Plan

The plan specified `cpu_total` / `bw_total` as capacity attribute names in
`_release_resources`. The **live codebase** uses `max_cpu` / `max_bw` (as used by
`substrate_utilisation()` in `graph_utils.py:150-154`). The implementation uses
`max_cpu` / `max_bw` to match, with a safe fallback:

```python
# Correct (matches graph_utils.py):
cpu_cap = float(substrate.nodes[snode].get('max_cpu', current + cpu_req))
bw_cap  = float(substrate.edges[a, b].get('max_bw',  current_bw + bw_req))
```

---

## Two Reset Sites -- Both Hooked

The original plan noted one `env.reset()` site. The live `_collect_rollout` loop
has **two** distinct reset paths; both now call `on_episode_start()` before reset:

```python
# Site A -- empty vnr_list detected mid-loop
if not obs["vnr_list"]:
    if self.stateful_wrapper is not None:
        self.stateful_wrapper.on_episode_start()   # Hook 1A
    self._obs, _ = self.env.reset()

# Site B -- episode done flag
if done:
    if self.stateful_wrapper is not None:
        self.stateful_wrapper.on_episode_start()   # Hook 1B
    self._obs, _ = self.env.reset()
```

A `_committed_ids_this_episode` set in the wrapper guards against double-commits
if both sites fire within the same iteration.

---

## Parameter Calibration

| Parameter | Default | Calibration formula |
|-----------|---------|---------------------|
| `ss_vnr_lifetime_episodes` (K) | 5 | `K = round(avg_lifetime / (avg_inter_arrival x avg_batch_size))` |
| `ss_overflow_cpu_threshold` | 0.88 | 5-10% above max expected inference util (~80%) |
| `ss_warmup_episodes` | 50 | Enough episodes for agent to learn basic policy |

**K for default config** (lifetime ~25, inter_arrival ~1.0, batch ~10): **K = 3**
**K for fig6 dataset** (Pareto lifetime ~65, batch ~10): **K = 7**

---

## New TensorBoard Metrics

| Metric | Expected range | Failure signal |
|--------|---------------|----------------|
| `StatefulSubstrate/LiveCpuUtil` | 0%-88% | Always 0%: wrapper not committing |
| `StatefulSubstrate/LiveBwUtil` | 0%-85% | Always 0%: wrapper not committing |
| `StatefulSubstrate/CommittedVNRs` | 0-~50 | Monotonically growing: expiry not working |

**Success criterion**: after 5000+ steps, `LiveCpuUtil` must have mean in **[25%, 60%]**
with visible variance (std > 5%) -- confirming the wrapper replicates the inference substrate distribution.

---

## Verification Checklist

- [ ] `python -m src.training.train_ppo --total-steps 100` (no flag) -- baseline unchanged
- [ ] `python -m src.training.train_ppo --stateful-substrate --ss-lifetime-episodes 3 --total-steps 200` -- no error, TensorBoard shows `StatefulSubstrate/` metrics
- [ ] After 1000 episodes with wrapper enabled: `mean(live_cpu_util)` in [30%, 55%]
- [ ] `len(wrapper.committed_vnrs)` decreases after K episodes (expiry working)
- [ ] Overflow reset fires and logs warning when util > 88%

---

## Quick-Start Command

```bash
python -m src.training.train_ppo \
  --total-steps 10000 \
  --stateful-substrate \
  --ss-warmup 20 \
  --ss-lifetime-episodes 5 \
  --ss-overflow-threshold 0.88 \
  --reward longterm \
  --run-name ppo_stateful_test \
  --save-dir checkpoints

tensorboard --logdir runs/ppo_stateful_test
```

Navigate to **StatefulSubstrate/** in TensorBoard and verify `LiveCpuUtil` oscillates
between ~10% and ~85% rather than sitting at 0%.
