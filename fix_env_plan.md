# Implementation Plan: Stateful Substrate Wrapper for PPO Training Distribution Alignment

**Document Version:** 1.0  
**Problem Class:** Covariate Shift / Train-Inference Distribution Mismatch  
**Affected System:** VNE (Virtual Network Embedding) RL Training Pipeline  
**Target Module:** `src/training/` (wrapper only — no existing module is modified)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Root Cause Analysis](#2-root-cause-analysis)
3. [Constraints & Non-Negotiables](#3-constraints--non-negotiables)
4. [Solution Architecture](#4-solution-architecture)
5. [File Layout](#5-file-layout)
6. [Detailed Implementation Spec](#6-detailed-implementation-spec)
   - 6.1 [StatefulSubstrateConfig](#61-statefulsubstrateconfig)
   - 6.2 [VNRRecord](#62-vnrrecord)
   - 6.3 [StatefulSubstrateWrapper](#63-statefulsubstratewrapper)
   - 6.4 [PPOConfig changes](#64-ppoconfig-changes)
   - 6.5 [PPOTrainerScheduler integration](#65-ppotrainerscheduler-integration)
7. [Physics Correctness Requirements](#7-physics-correctness-requirements)
8. [Parameter Calibration Guide](#8-parameter-calibration-guide)
9. [TensorBoard Metrics Specification](#9-tensorboard-metrics-specification)
10. [Verification Checklist](#10-verification-checklist)
11. [Rollback & Safety Switches](#11-rollback--safety-switches)
12. [Known Edge Cases](#12-known-edge-cases)

---

## 1. Problem Statement

### 1.1 Observed Behaviour

During inference, `BatchedVNRSimulator` operates on a **single persistent substrate** that accumulates utilisation over many time-window batches. VNRs are embedded, consume resources, and are later released when their `lifetime` expires. At any point in a real simulation the substrate CPU utilisation can be anywhere in the range **[0 %, ~80 %]**.

During training, `VNEOrderingEnv.reset()` calls `self.substrate_fn()` which always returns a **brand-new substrate** (0 % utilisation). The agent therefore trains exclusively on the easiest possible substrate state.

### 1.2 Formal Description

Let:
- $P_{train}(s)$ = distribution of substrate states seen during training
- $P_{inf}(s)$ = distribution of substrate states seen during `BatchedVNRSimulator` inference

Current situation: $P_{train}(s) = \delta(\text{empty substrate})$, while $P_{inf}(s)$ is a broad distribution over utilisation levels. This is **covariate shift** — the policy $\pi_\theta(a \mid s)$ is optimised on $P_{train}$ but deployed on $P_{inf}$.

### 1.3 Consequence

The trained agent has never learned how to prioritise VNRs when the substrate is already 40 %–70 % full. It therefore makes systematically worse ordering decisions during inference, degrading acceptance ratio and revenue/cost ratio compared to what training metrics would predict.

---

## 2. Root Cause Analysis

```
VNEOrderingEnv.reset()
  └── raw_substrate = self.substrate_fn()   ← always fresh (0% util)
  └── self.substrate = copy_substrate(raw_substrate)
```

The `substrate_fn` is a closure from `make_substrate_fn()` in `generate_data.py`. It either generates a new random substrate each call or returns the same topology with **all resources restored to full**. There is no mechanism to carry depletion state across episodes.

The `BatchedVNRSimulator` on the other hand maintains a single `self.substrate` across all time windows, applying `release_node` / `release_path` when VNRs depart. This cross-window accumulation is the fundamental behaviour the training environment must replicate.

---

## 3. Constraints & Non-Negotiables

| # | Constraint | Rationale |
|---|---|---|
| C1 | Do **not** modify `VNEOrderingEnv`, `hpso_embed`, `BatchedVNRSimulator`, or any module under `src/algorithms/` or `src/simulation/` | These are production inference modules; changes risk breaking experiments |
| C2 | Solution must be a **wrapper or plugin** added entirely within `src/training/` | Clean separation between training infra and core env |
| C3 | When the wrapper is **disabled** (feature flag = False), behaviour must be **byte-for-byte identical** to the current `PPOTrainerScheduler` | Full backward compatibility |
| C4 | Resource depletion must follow **real physics**: consume on accept, release on lifetime expiry — no random noise, no probabilistic shortcuts | Heuristic depletion would just create a different distribution mismatch |
| C5 | The wrapper must be compatible with `PPOTrainerScheduler` as it exists in `train_ppo.py` — constructor and `train()` interface unchanged | Existing training scripts must continue to work |

---

## 4. Solution Architecture

### 4.1 High-Level Design

```
PPOTrainerScheduler
    │
    │ (when stateful_substrate=True)
    ▼
StatefulSubstrateWrapper          ← NEW FILE: src/training/stateful_substrate.py
    │
    │  intercepts substrate_fn calls
    │  maintains live_substrate across episodes
    │  expires committed VNRs by episode counter
    │
    ▼
VNEOrderingEnv                    ← UNCHANGED
    │
    ▼
hpso_embed                        ← UNCHANGED
```

### 4.2 Key Mechanisms

**Mechanism 1 — Persistent substrate across episodes**  
The wrapper owns `live_substrate`, a NetworkX graph that is never reset unless overflow protection fires. Each time `VNEOrderingEnv.reset()` internally calls `substrate_fn()`, the wrapper intercepts this call by replacing `substrate_fn` with a closure that returns `copy_substrate(live_substrate)`. The inner env therefore always "sees" the current depletion state.

**Mechanism 2 — Post-step resource commit**  
After each `env.step()` that results in a successful embedding, the wrapper reads the mapping and link_paths from `env.accepted[-1]` and applies the same resource deductions to `live_substrate`. This mirrors how `BatchedVNRSimulator._embed_batch()` commits resources to its own persistent substrate.

**Mechanism 3 — Lifetime-based expiry**  
Every committed VNR is tagged with an `expiry_episode` = `commit_episode + K` (where K is a config parameter). At the start of each episode, the wrapper scans `committed_vnrs` and calls `release_node` / `release_path` on all expired records, then removes them from the list.

**Mechanism 4 — Overflow protection**  
If `cpu_util(live_substrate) > overflow_threshold` at the start of a new episode, the wrapper resets `live_substrate` back to a fresh substrate. This prevents the training from getting permanently blocked on a nearly-dead substrate.

**Mechanism 5 — Warm-up period**  
For the first `warmup_episodes` episodes, the wrapper returns a fresh substrate exactly as before. This gives the agent time to develop basic ordering intuition before being exposed to pre-depleted states.

---

## 5. File Layout

### New files (implement these):

```
src/training/stateful_substrate.py     ← Main implementation (this document)
```

### Modified files (minimal changes only):

```
src/training/train_ppo.py
  - Add fields to PPOConfig dataclass (all Optional with defaults matching current behaviour)
  - Add 5-line block in PPOTrainerScheduler.__init__() to attach wrapper when enabled
```

### Unchanged files (do not touch):

```
src/scheduler/environment.py
src/algorithms/fast_hpso.py
src/simulation/simulator.py
src/utils/graph_utils.py
src/algorithms/hpso_batch_scheduler.py
```

---

## 6. Detailed Implementation Spec

### 6.1 `StatefulSubstrateConfig`

**Location:** `src/training/stateful_substrate.py`

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class StatefulSubstrateConfig:
    """
    Configuration for StatefulSubstrateWrapper.
    
    All parameters have safe defaults. When enabled=False the wrapper
    is a no-op and training behaviour is identical to the baseline.
    """
    enabled: bool = False
    
    # --- Lifetime simulation ---
    vnr_lifetime_episodes: int = 5
    # How many episodes a committed VNR occupies live_substrate before
    # being expired and its resources returned.
    #
    # Calibration formula (derive from dataset stats):
    #   K = round(avg_lifetime / (avg_inter_arrival * window_size))
    #
    # Example: avg_lifetime=300, avg_inter_arrival=1.0, window_size=10
    #   → K = round(300 / (1.0 * 10)) = 30
    #
    # For the default generate_data.py settings (lifetime drawn up to 50,
    # avg ~25, inter_arrival~1, effective window ~10 VNRs per episode):
    #   → K ≈ 3–5 is a reasonable starting point.
    
    # --- Overflow protection ---
    overflow_cpu_threshold: float = 0.88
    # When live_substrate CPU utilisation exceeds this fraction, the
    # wrapper resets live_substrate to a fresh substrate.
    # 0.88 leaves a small buffer above the expected ~80% inference peak.
    
    # --- Warm-up ---
    warmup_episodes: int = 50
    # For the first N episodes, use fresh substrate (wrapper is passive).
    # Allows the agent to learn a basic policy before substrate hardening.
    
    # --- Substrate regeneration on overflow ---
    regenerate_substrate_on_overflow: bool = True
    # If True (default), overflow resets to a newly generated substrate
    # via the original substrate_fn.
    # If False, overflow resets to the substrate captured at wrapper init
    # (same topology, all resources restored).
```

**Notes:**
- All fields must have defaults that preserve existing behaviour when `enabled=False`.
- `vnr_lifetime_episodes` is the single most important parameter to calibrate correctly — see Section 8.

---

### 6.2 `VNRRecord`

**Location:** `src/training/stateful_substrate.py`

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import networkx as nx

@dataclass
class VNRRecord:
    """
    Tracks a single committed VNR on live_substrate.
    
    Fields
    ------
    vnr          : the original VNR graph (needed for cpu demand per vnode)
    mapping      : {vnode_id: snode_id}  node placement
    link_paths   : {(u, v): [s0, s1, ..., sk]}  path for each virtual link
    commit_episode : episode number when this VNR was committed
    expiry_episode : commit_episode + vnr_lifetime_episodes
    """
    vnr: nx.Graph
    mapping: Dict[int, int]
    link_paths: Dict[Tuple[int, int], List[int]]
    commit_episode: int
    expiry_episode: int
```

---

### 6.3 `StatefulSubstrateWrapper`

**Location:** `src/training/stateful_substrate.py`

This is the core class. Full method-by-method specification follows.

#### Constructor `__init__(self, env, substrate_fn, cfg)`

```
Parameters
----------
env          : VNEOrderingEnv instance (already constructed)
substrate_fn : the original substrate_fn callable from make_env_fns()
               (kept for fresh substrate generation on overflow)
cfg          : StatefulSubstrateConfig
```

**Actions in `__init__`:**
1. Store references: `self.env`, `self.original_substrate_fn`, `self.cfg`
2. Generate initial `self.live_substrate = substrate_fn()`
3. Store `self.initial_substrate = copy_substrate(self.live_substrate)` as fallback for non-regenerative overflow reset
4. Initialise `self.committed_vnrs: List[VNRRecord] = []`
5. Initialise `self.episode_count: int = 0`
6. Initialise `self.live_cpu_util_history: List[float] = []` (for logging)
7. **Patch the inner env**: `env.substrate_fn = self._patched_substrate_fn`
   - This is the key intercept. `VNEOrderingEnv.reset()` calls `self.substrate_fn()` internally. By replacing it here, every subsequent `env.reset()` will receive `copy_substrate(self.live_substrate)` instead of a fresh graph.

#### `_patched_substrate_fn(self) -> nx.Graph`

```
Returns a copy of the current live_substrate.
This is what VNEOrderingEnv.reset() will call when constructing each episode.
```

```python
def _patched_substrate_fn(self) -> nx.Graph:
    return copy_substrate(self.live_substrate)
```

**Important:** `copy_substrate` must perform a deep copy. Verify it copies both node and edge attribute dicts — check `src/utils/graph_utils.py`. The copy is essential so that `hpso_embed` can mutate `env.substrate` during an episode without corrupting `live_substrate`.

#### `on_episode_start(self)`

Called at the beginning of each episode (before `env.reset()`).

**Algorithm:**
```
1. Increment self.episode_count

2. If episode_count <= cfg.warmup_episodes:
       # Passive: the patched substrate_fn already returns a copy of live_substrate,
       # but live_substrate hasn't been committed to yet, so it's fresh anyway.
       # No action needed.
       return

3. Expire committed VNRs:
   remaining = []
   for record in self.committed_vnrs:
       if self.episode_count >= record.expiry_episode:
           _release_resources(self.live_substrate, record)
           # log: "Expired VNR committed at ep X"
       else:
           remaining.append(record)
   self.committed_vnrs = remaining

4. Check overflow:
   util = substrate_utilisation(self.live_substrate)
   self.live_cpu_util_history.append(util['cpu_util'])
   
   if util['cpu_util'] > cfg.overflow_cpu_threshold:
       if cfg.regenerate_substrate_on_overflow:
           self.live_substrate = self.original_substrate_fn()
       else:
           self.live_substrate = copy_substrate(self.initial_substrate)
       self.committed_vnrs = []
       # log: "OVERFLOW RESET at episode N, util was X%"
```

#### `on_step(self, action, result_obs, reward, done, info)`

Called after each `env.step()` returns. Checks if a new VNR was accepted and commits it to `live_substrate`.

**Algorithm:**
```
1. If episode_count <= cfg.warmup_episodes:
       return   # passive during warm-up

2. Check whether the last step accepted a VNR:
   # The env tracks accepted VNRs in env.accepted (list of (vnr, mapping, link_paths))
   # Compare current len(env.accepted) to the count before this step.
   # A simpler approach: check env.last_success (bool set by env.step)

3. If env.last_success == True:
   vnr, mapping, link_paths = env.accepted[-1]   # most recently accepted
   
   record = VNRRecord(
       vnr            = vnr,
       mapping        = mapping,
       link_paths     = link_paths,
       commit_episode = self.episode_count,
       expiry_episode = self.episode_count + cfg.vnr_lifetime_episodes,
   )
   
   _consume_resources(self.live_substrate, record)
   self.committed_vnrs.append(record)
```

#### `_consume_resources(substrate, record)` (module-level helper)

Applies the embedding to the substrate graph in-place. Must exactly mirror the resource accounting used by `hpso_embed`.

```python
def _consume_resources(substrate: nx.Graph, record: VNRRecord) -> None:
    """
    Deduct CPU and BW from substrate nodes/edges per the VNR mapping.
    
    Uses the same attribute names as hpso_embed and graph_utils:
      node attribute: 'cpu'
      edge attribute: 'bw'
    
    IMPORTANT: Does NOT deduct below zero. If a deduction would produce
    a negative value, clamp to 0 and log a warning. This handles any
    slight floating-point drift between env.substrate and live_substrate.
    """
    # Node resources
    for vnode, snode in record.mapping.items():
        if snode not in substrate.nodes:
            continue  # defensive: substrate topology may differ slightly
        cpu_req = float(record.vnr.nodes[vnode].get('cpu', 0.0))
        current = float(substrate.nodes[snode].get('cpu', 0.0))
        substrate.nodes[snode]['cpu'] = max(0.0, current - cpu_req)
    
    # Link resources
    for (u, v), path in record.link_paths.items():
        bw_req = float(record.vnr.edges[u, v].get('bw', 0.0))
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            if substrate.has_edge(a, b):
                current_bw = float(substrate.edges[a, b].get('bw', 0.0))
                substrate.edges[a, b]['bw'] = max(0.0, current_bw - bw_req)
            elif substrate.has_edge(b, a):  # undirected graph: try reverse
                current_bw = float(substrate.edges[b, a].get('bw', 0.0))
                substrate.edges[b, a]['bw'] = max(0.0, current_bw - bw_req)
```

#### `_release_resources(substrate, record)` (module-level helper)

Returns resources to the substrate. Must NOT exceed `cpu_total` / `bw_total`.

```python
def _release_resources(substrate: nx.Graph, record: VNRRecord) -> None:
    """
    Restore CPU and BW to substrate nodes/edges when a VNR expires.
    
    Clamps restored values to their original totals to prevent
    floating-point accumulation from exceeding physical capacity.
    """
    # Node resources
    for vnode, snode in record.mapping.items():
        if snode not in substrate.nodes:
            continue
        cpu_req   = float(record.vnr.nodes[vnode].get('cpu', 0.0))
        current   = float(substrate.nodes[snode].get('cpu', 0.0))
        cpu_total = float(substrate.nodes[snode].get('cpu_total', current + cpu_req))
        substrate.nodes[snode]['cpu'] = min(cpu_total, current + cpu_req)
    
    # Link resources
    for (u, v), path in record.link_paths.items():
        bw_req = float(record.vnr.edges[u, v].get('bw', 0.0))
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            if substrate.has_edge(a, b):
                current_bw = float(substrate.edges[a, b].get('bw', 0.0))
                bw_total   = float(substrate.edges[a, b].get('bw_total', current_bw + bw_req))
                substrate.edges[a, b]['bw'] = min(bw_total, current_bw + bw_req)
            elif substrate.has_edge(b, a):
                current_bw = float(substrate.edges[b, a].get('bw', 0.0))
                bw_total   = float(substrate.edges[b, a].get('bw_total', current_bw + bw_req))
                substrate.edges[b, a]['bw'] = min(bw_total, current_bw + bw_req)
```

#### `get_live_util(self) -> dict`

Returns current utilisation stats for TensorBoard logging.

```python
def get_live_util(self) -> dict:
    from src.utils.graph_utils import substrate_utilisation
    util = substrate_utilisation(self.live_substrate)
    return {
        'live_cpu_util': util['cpu_util'],
        'live_bw_util':  util.get('bw_util', 0.0),
        'committed_vnrs': len(self.committed_vnrs),
        'episode': self.episode_count,
    }
```

---

### 6.4 `PPOConfig` changes

**Location:** `src/training/train_ppo.py`

Add the following fields to the `PPOConfig` dataclass. All have defaults that preserve existing behaviour:

```python
@dataclass
class PPOConfig:
    # ... existing fields unchanged ...

    # ── Stateful substrate (distribution alignment) ──────────────────────────
    stateful_substrate:          bool  = False
    # Master switch. When False (default), all fields below are ignored
    # and training is identical to the original.
    
    ss_warmup_episodes:          int   = 50
    # Episodes using fresh substrate before depletion kicks in.
    
    ss_vnr_lifetime_episodes:    int   = 5
    # How many episodes a committed VNR occupies live_substrate.
    # Calibrate using: K = round(avg_lifetime / (avg_inter_arrival * avg_batch_size))
    
    ss_overflow_cpu_threshold:   float = 0.88
    # CPU utilisation fraction above which live_substrate is force-reset.
    
    ss_regenerate_on_overflow:   bool  = True
    # True = generate a new random substrate on overflow.
    # False = restore the initial substrate's resources.
```

---

### 6.5 `PPOTrainerScheduler` integration

**Location:** `src/training/train_ppo.py`, inside `PPOTrainerScheduler.__init__()`

Add the following block **after** `self.env = VNEOrderingEnv(...)` is constructed and **before** `self._obs, _ = self.env.reset()`:

```python
# ── Stateful substrate wrapper (optional) ────────────────────────────────────
self.stateful_wrapper = None
if cfg.stateful_substrate:
    from src.training.stateful_substrate import (
        StatefulSubstrateWrapper, StatefulSubstrateConfig
    )
    ss_cfg = StatefulSubstrateConfig(
        enabled                        = True,
        warmup_episodes                = cfg.ss_warmup_episodes,
        vnr_lifetime_episodes          = cfg.ss_vnr_lifetime_episodes,
        overflow_cpu_threshold         = cfg.ss_overflow_cpu_threshold,
        regenerate_substrate_on_overflow = cfg.ss_regenerate_on_overflow,
    )
    self.stateful_wrapper = StatefulSubstrateWrapper(
        env          = self.env,
        substrate_fn = substrate_fn,   # original fn from make_env_fns()
        cfg          = ss_cfg,
    )
    print(f"[PPO] StatefulSubstrateWrapper ENABLED "
          f"(warmup={ss_cfg.warmup_episodes}, K={ss_cfg.vnr_lifetime_episodes}, "
          f"overflow={ss_cfg.overflow_cpu_threshold:.0%})")
```

**Modification to `_collect_rollout()`:**

In the rollout loop, the wrapper's hooks must be called at the right moments. Add two hook calls:

```python
# --- Hook 1: episode start ---
# Place this BEFORE the call to env.reset() inside the "if done:" block
# and also at the very start of the first rollout call.
if self.stateful_wrapper is not None:
    self.stateful_wrapper.on_episode_start()

# --- Hook 2: after each step ---
# Place this IMMEDIATELY AFTER the call to env.step()
next_obs, reward, done, _, info = self.env.step(action.item())

if self.stateful_wrapper is not None:
    self.stateful_wrapper.on_step(action.item(), next_obs, reward, done, info)
```

**Modification to TensorBoard logging (inside the `if done:` block):**

```python
if done:
    ep_info = self.env.episode_summary()
    # ... existing logging ...
    
    # Stateful substrate metrics
    if self.stateful_wrapper is not None:
        live = self.stateful_wrapper.get_live_util()
        self.writer.add_scalar("StatefulSubstrate/LiveCpuUtil",
                               live['live_cpu_util'], global_step)
        self.writer.add_scalar("StatefulSubstrate/LiveBwUtil",
                               live['live_bw_util'], global_step)
        self.writer.add_scalar("StatefulSubstrate/CommittedVNRs",
                               live['committed_vnrs'], global_step)
```

**Modification to `PPOConfig` CLI argument parsing (in `_build_parser()`):**

```python
# Add to _build_parser():
p.add_argument("--stateful-substrate",     action="store_true",
               help="Enable stateful substrate wrapper for distribution alignment")
p.add_argument("--ss-warmup",              type=int,   default=50)
p.add_argument("--ss-lifetime-episodes",   type=int,   default=5)
p.add_argument("--ss-overflow-threshold",  type=float, default=0.88)

# Add to PPOConfig(...) construction in __main__:
stateful_substrate        = args.stateful_substrate,
ss_warmup_episodes        = args.ss_warmup,
ss_vnr_lifetime_episodes  = args.ss_lifetime_episodes,
ss_overflow_cpu_threshold = args.ss_overflow_threshold,
```

---

## 7. Physics Correctness Requirements

The wrapper must replicate the exact same resource accounting as `BatchedVNRSimulator`. The following properties must hold:

### 7.1 Attribute name contract

| Resource | Node attribute | Edge attribute |
|---|---|---|
| CPU | `cpu` (available), `cpu_total` (capacity) | — |
| Bandwidth | — | `bw` (available), `bw_total` (capacity) |

These names are used consistently in `hpso_embed`, `release_node`, `release_path`, and `substrate_utilisation`. The wrapper must use only these same attribute names.

### 7.2 Monotonicity invariants

At all times, the following must hold on every node `n` and edge `(u,v)` in `live_substrate`:

```
0 <= live_substrate.nodes[n]['cpu'] <= live_substrate.nodes[n]['cpu_total']
0 <= live_substrate.edges[u,v]['bw'] <= live_substrate.edges[u,v]['bw_total']
```

Any `_consume_resources` or `_release_resources` call that would violate these bounds must clamp and log a warning. This can happen due to floating-point accumulation if the same episode's env.substrate and live_substrate diverge slightly. Clamping is the correct response.

### 7.3 No double-counting

Each VNR must be committed to `live_substrate` exactly once and released exactly once. The implementation must prevent:
- Committing the same VNR twice if `on_step()` is called twice for the same accepted VNR
- Releasing a VNR that was never committed (e.g., due to overflow reset clearing `committed_vnrs`)

Guard with: track the set of `id(vnr)` objects already committed in the current episode.

### 7.4 Substrate identity

`live_substrate` and `env.substrate` (set inside `VNEOrderingEnv.reset()`) are **different objects**. The env operates on its own copy; `live_substrate` is a separate accumulator. This separation is maintained by `_patched_substrate_fn` always returning `copy_substrate(live_substrate)` rather than `live_substrate` directly.

---

## 8. Parameter Calibration Guide

### 8.1 `vnr_lifetime_episodes` (K)

This is the most impactful parameter. It controls how quickly `live_substrate` fills up relative to how quickly it empties.

**Calibration from dataset stats:**

```
K = round(avg_vnr_lifetime / (avg_inter_arrival_time * avg_batch_size))
```

Where:
- `avg_vnr_lifetime`: mean lifetime from your VNR dataset (check `vnr_stream.json`, field `"lifetime"`)
- `avg_inter_arrival_time`: mean gap between VNR arrival times
- `avg_batch_size`: number of VNRs processed per episode (`vnr_batch_min` to `vnr_batch_max`, use midpoint)

**Example calculation for default `PPOConfig` settings:**
- VNR lifetime drawn from up to 50 time units → avg ≈ 25
- Inter-arrival ≈ 1.0 (exponential mean)
- Batch size ≈ 10 (midpoint of [5, 15])
- K = round(25 / (1.0 × 10)) = **3**

**For fig6 dataset (`generate_vnr_stream_v2` with `max_lifetime=300`, `avg_inter_arrival=1.0`, window_size=10):**
- Pareto lifetime avg ≈ 50–80 time units (Pareto α=2.5 with ×10 scale, clipped to 300)
- K = round(65 / (1.0 × 10)) = **7**

**Verification**: After implementing, run 1000 episodes with the wrapper enabled and compute `mean(live_cpu_util)`. Target: 30 %–55 %. If mean > 70 %, decrease K. If mean < 20 %, increase K.

### 8.2 `overflow_cpu_threshold`

Set 5–10 % above the maximum expected inference utilisation. `BatchedVNRSimulator` typically peaks around 75–80 % CPU utilisation based on the substrate capacity and VNR demands. Default of 0.88 provides reasonable headroom.

**Do not set below 0.80**: this will cause excessive resets that under-expose the agent to high-utilisation states.  
**Do not set above 0.95**: the substrate may become non-embeddable, causing the training to stall (all VNRs rejected, zero gradient signal).

### 8.3 `warmup_episodes`

During warm-up, `live_substrate` is fresh (0 % util) because no commits have been made yet. The warm-up thus happens naturally without any special casing, as long as the wrapper is passive at episode start.

However, explicitly tracking warm-up via `warmup_episodes` allows future enhancements (e.g., logging "warm-up phase" vs "depletion phase" in TensorBoard). Default of 50 is suitable for most configurations.

---

## 9. TensorBoard Metrics Specification

After implementing the wrapper, the following metrics must appear in TensorBoard runs.

### 9.1 New metrics (under `StatefulSubstrate/` namespace)

| Metric name | Description | Expected range | Failure indicator |
|---|---|---|---|
| `StatefulSubstrate/LiveCpuUtil` | CPU utilisation of `live_substrate` at episode start | 0 %–88 % | Always 0 %: wrapper not committing; Always >85 %: overflow threshold too high |
| `StatefulSubstrate/LiveBwUtil` | BW utilisation of `live_substrate` | 0 %–85 % | Similar to CPU |
| `StatefulSubstrate/CommittedVNRs` | Number of VNR records currently in `committed_vnrs` | 0–~50 | Monotonically growing and never decreasing: expiry not working |

### 9.2 Existing metrics to monitor for change

| Metric name | Expected change after enabling wrapper |
|---|---|
| `Metrics/AcceptanceRate` | May initially decrease as agent sees harder states; should recover with training |
| `Metrics/SubstrateCpuUtil` | Per-episode env substrate util; should now show variance around a non-zero mean |
| `Metrics/RevenueCostRatio` | May improve if agent learns better ordering under depletion |

### 9.3 Success criterion

After 5000+ training steps with the wrapper enabled, the TensorBoard plot for `StatefulSubstrate/LiveCpuUtil` must show a distribution that:
- Has mean in the range **[25 %, 60 %]**
- Has visible variance (standard deviation > 5 %)
- Does not appear "stuck" at 0 % or >85 %

This would confirm the wrapper is correctly replicating the inference-time substrate distribution.

---

## 10. Verification Checklist

Work through this list before declaring the implementation complete.

### Unit tests (manual or automated)

- [ ] **Test 1 — No-op when disabled**: Create `StatefulSubstrateConfig(enabled=False)`, run 10 episodes, assert substrate is fresh at start of each episode. Assert `committed_vnrs` is always empty.
- [ ] **Test 2 — Commit on accept**: Run 1 episode where at least 1 VNR is accepted. Assert `len(wrapper.committed_vnrs) >= 1` after episode. Assert `live_substrate` has lower CPU than a fresh substrate.
- [ ] **Test 3 — No commit on reject**: Force a scenario where all VNRs are rejected (e.g., very high CPU demands). Assert `len(wrapper.committed_vnrs) == 0` after episode.
- [ ] **Test 4 — Expiry at K episodes**: Commit 1 VNR at episode 1. Set K=3. Assert the VNR is in `committed_vnrs` at episodes 2, 3. Assert the VNR is released at episode 4. Assert CPU is restored.
- [ ] **Test 5 — Overflow reset**: Fill `live_substrate` above threshold manually. Assert that at the start of the next `on_episode_start()`, `live_substrate` is reset and `committed_vnrs` is empty.
- [ ] **Test 6 — Monotonicity invariant**: After 100 random episodes, assert that for every node and edge in `live_substrate`, `0 <= cpu <= cpu_total` and `0 <= bw <= bw_total`.
- [ ] **Test 7 — Backward compatibility**: Run `PPOTrainerScheduler` with `stateful_substrate=False` for 100 steps. Assert training completes without error. Compare episode reward distribution to pre-patch baseline (should be statistically identical within floating-point noise).
- [ ] **Test 8 — CLI flag**: Run `python -m src.training.train_ppo --stateful-substrate --ss-lifetime-episodes 3 --total-steps 100`. Assert no error, assert TensorBoard directory contains `StatefulSubstrate/LiveCpuUtil` scalar.

### Integration check

- [ ] **TensorBoard check**: After 2000 training steps, open TensorBoard. Confirm `StatefulSubstrate/LiveCpuUtil` is visible and oscillating between ~10 % and ~85 %, not stuck.
- [ ] **Physics sanity**: At any point during training, print `substrate_utilisation(wrapper.live_substrate)`. The result should be a plausible non-zero value.

---

## 11. Rollback & Safety Switches

Because this change modifies training infrastructure, the following safety mechanisms must be in place before merging.

### Master switch
The `stateful_substrate: bool = False` default in `PPOConfig` means **all existing training scripts work without modification**. The feature requires an explicit opt-in. This is the primary safety guarantee.

### Passive warm-up
The `warmup_episodes` parameter means even when the feature is enabled, the first N episodes are identical to baseline. This allows the agent to establish a basic policy before encountering harder states.

### Overflow hard reset
If the substrate becomes over-depleted (> `overflow_cpu_threshold`), the wrapper force-resets to a fresh substrate. This prevents training from getting stuck in an unrecoverable state where every VNR is rejected and the gradient signal disappears.

### Isolation of `live_substrate`
The wrapper never passes `live_substrate` directly to `VNEOrderingEnv`. It always passes `copy_substrate(live_substrate)`. Therefore, even if `hpso_embed` causes unexpected mutations inside the env's substrate, `live_substrate` remains clean. The only modifications to `live_substrate` go through `_consume_resources` and `_release_resources`, which are under the wrapper's control.

---

## 12. Known Edge Cases

| Edge case | Situation | Correct handling |
|---|---|---|
| Empty batch episode | `batch_fn()` returns 0 VNRs; env terminates immediately | `on_step()` is never called; `on_episode_start()` increments counter; expiry check still runs |
| All-reject episode | No VNRs accepted; `env.last_success` always False | No commits to `live_substrate`; util stays unchanged from previous episode |
| Overflow during warm-up | `live_substrate` is fresh during warm-up by definition; overflow cannot trigger | N/A — no commits during warm-up → util stays 0 % → threshold never reached |
| K=1 | VNR expires after just 1 episode | Valid but extreme: live_substrate oscillates rapidly. K=1 with high acceptance rate can cause net-zero depletion. Should not be used in practice. |
| Variable topology across episodes | `substrate_fn()` generates different topologies each call; after overflow reset, the new substrate may have different node IDs | `committed_vnrs` is cleared on overflow reset, so no stale mappings are applied to the new topology |
| `cpu_total` attribute missing | Some substrate generators might not set `cpu_total` | `_release_resources` uses `cpu + cpu_req` as fallback total. Add defensive read in `_consume_resources` as well. |
| Multi-threaded rollout | Future extension may collect rollouts in parallel | The current architecture is single-threaded; `live_substrate` is not thread-safe. Document this as a known limitation; do not add locking yet. |
| Negative CPU after excessive commits | Floating-point drift between env's substrate copy and live_substrate | Clamped to 0 in `_consume_resources`. Log a warning so the operator can diagnose if this fires frequently. |

---

## Appendix A: Complete File Template

The complete `src/training/stateful_substrate.py` should be structured as follows (implementer fills in method bodies per spec above):

```python
"""
src/training/stateful_substrate.py
====================================
StatefulSubstrateWrapper — resolves train/inference distribution mismatch
by maintaining a persistent, depleting substrate across training episodes.

Problem addressed
-----------------
VNEOrderingEnv.reset() always creates a fresh substrate (0% utilisation).
BatchedVNRSimulator inference operates on a substrate that accumulates
depletion across many time windows. This covariate shift degrades policy
performance at inference time.

Solution
--------
This wrapper intercepts the substrate_fn call inside VNEOrderingEnv.reset()
and replaces it with a closure that returns a copy of a live, persistent
substrate. The wrapper commits accepted VNR embeddings to the live substrate
and expires them after K episodes, simulating the load/unload cycle of
BatchedVNRSimulator.

Usage
-----
Enabled via PPOConfig.stateful_substrate = True.
When disabled (default), behaviour is identical to baseline training.

Constraints
-----------
- Does NOT modify VNEOrderingEnv, hpso_embed, BatchedVNRSimulator, or
  any module outside src/training/.
- Backward-compatible: disabled by default.
- Replicates exact resource physics: consume on accept, release on expiry.
"""

from __future__ import annotations

# imports...
# StatefulSubstrateConfig (dataclass)
# VNRRecord (dataclass)
# _consume_resources(substrate, record) -> None
# _release_resources(substrate, record) -> None
# StatefulSubstrateWrapper
#   __init__(self, env, substrate_fn, cfg)
#   _patched_substrate_fn(self) -> nx.Graph
#   on_episode_start(self) -> None
#   on_step(self, action, next_obs, reward, done, info) -> None
#   get_live_util(self) -> dict
```

---

## Appendix B: Quick-Start Command

Once implemented, test with:

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

Then navigate to the `StatefulSubstrate/` section in TensorBoard and verify `LiveCpuUtil` shows the expected oscillating pattern in the range [0 %, 88 %].

---

*End of Implementation Plan*
