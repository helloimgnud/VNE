# Analysis: Train/Inference Distribution Mismatch

## 1. The Core Bug — Confirmed in Code

### Training side (`VNEOrderingEnv.reset()` — `environment.py:134`)
```python
raw_substrate  = self.substrate_fn()       # ← always a brand-new graph
self.substrate = copy_substrate(raw_substrate)
```
`self.substrate_fn` is the closure returned by `make_substrate_fn()` in `generate_data.py`.
Every call to `_fn()` calls `generate_substrate(...)` with a fresh random seed → **always 0 % utilisation**.

### Inference side (`BatchedVNRSimulator` — `simulator.py:352`)
```python
# _embed_batch: after each accepted VNR:
self.active_vnrs.append((vnr_graph, mapping, link_paths, departure_time))
# substrate resources consumed in-place, NOT reset between windows
```
`self.substrate` inside `BatchedVNRSimulator` **persists across all time windows**.
`process_departures()` releases expired VNRs gradually — substrate floats between [0 %, ~80 %].

### Distribution gap
| | Training | Inference |
|---|---|---|
| Substrate state at episode/window start | Always 0 % util | Drawn from broad [0 %–80 %] distribution |
| Substrate topology | Regenerated each episode | Single persistent graph |
| Resource depletion | Never accumulated | Accumulated across whole VNR stream |

---

## 2. Key Verified Facts from Code Inspection

### 2.1 `copy_substrate` is safe for deep copy (graph_utils.py:12–27)
```python
def copy_substrate(sub):
    G = nx.Graph()
    for n, d in sub.nodes(data=True):
        G.add_node(n, **{...d})   # copies dict
    for u, v, d in sub.edges(data=True):
        G.add_edge(u, v, **{...d})
    return G
```
✅ Creates new node/edge dicts — mutations to the copy don't affect the original.
This means `_patched_substrate_fn` returning `copy_substrate(live_substrate)` is safe.

### 2.2 Attribute name contract (graph_utils.py:81, 93, 108, 123)
- Node: `'cpu'` = available, `'max_cpu'` = capacity (used by `substrate_utilisation`)
- Edge: `'bw'` = available, `'max_bw'` = capacity (used by `substrate_utilisation`)

> [!WARNING]
> `substrate_utilisation()` reads `max_cpu` / `max_bw` as capacity, but the plan's
> `_release_resources` reads `cpu_total` / `bw_total`. These are **different attribute names**.
> The plan's spec must be adjusted to use `max_cpu` / `max_bw` to match the live codebase.

### 2.3 `env.substrate_fn` is a plain instance attribute (environment.py:93)
```python
self.substrate_fn = substrate_fn
```
✅ Can be monkey-patched by the wrapper directly: `env.substrate_fn = self._patched_substrate_fn`

### 2.4 `env.accepted` structure (environment.py:178)
```python
self.accepted.append((vnr, mapping, link_paths))
```
✅ Matches the plan's `env.accepted[-1]` → `(vnr, mapping, link_paths)` tuple access.

### 2.5 `env.last_success` (environment.py:179, 187)
```python
self.last_success = True   # on accept
self.last_success = False  # on reject
```
✅ Reliable signal for `on_step()` to detect new commitments.

### 2.6 The rollout loop structure (train_ppo.py:204–244)
```python
for _ in range(n_steps):
    if not obs["vnr_list"]:
        self._obs, _ = self.env.reset()   # ← Hook 1 goes BEFORE this
        obs = self._obs
    ...
    next_obs, reward, done, _, info = self.env.step(action.item())  # ← Hook 2 goes after
    ...
    if done:
        # TensorBoard logging ← add StatefulSubstrate metrics here
        self._obs, _ = self.env.reset()   # ← Hook 1 also goes BEFORE this
```

> [!IMPORTANT]
> There are **two** `env.reset()` call sites in `_collect_rollout`:
> 1. Line 209: when `obs["vnr_list"]` is empty (episode boundary detected mid-loop)
> 2. Line 242: when `done == True`
> Both need `on_episode_start()` called **before** them.

### 2.7 No `cpu_total` / `bw_total` attributes in generators
From `substrate_utilisation()`:
```python
total_cpu = sum(d.get('max_cpu', d.get('cpu', 1e-9)) ...)
total_bw  = sum(d.get('max_bw',  d.get('bw',  1e-9)) ...)
```
The fallback is `d.get('cpu', 1e-9)` — meaning if `max_cpu` is absent, the capacity
**equals current available CPU** (a moving target). This will be wrong for `_release_resources`.

**Fix in `_release_resources`**: use `max_cpu` / `max_bw` as the cap, matching `substrate_utilisation`:
```python
cpu_total = float(substrate.nodes[snode].get('max_cpu', substrate.nodes[snode].get('cpu', 0.0) + cpu_req))
bw_total  = float(substrate.edges[a, b].get('max_bw', substrate.edges[a, b].get('bw', 0.0) + bw_req))
```

---

## 3. Implementation Gaps / Corrections Needed

| # | Plan says | Reality | Fix needed |
|---|---|---|---|
| G1 | Use `cpu_total` / `bw_total` | Codebase uses `max_cpu` / `max_bw` | `_release_resources` must use `max_cpu`/`max_bw` |
| G2 | Hook 1 "before env.reset()" | Two reset call sites in `_collect_rollout` | Both sites need the hook |
| G3 | "On episode start" increment counter | The early-exit reset (empty vnr_list) is between-step, not a clean boundary | Guard against double-increment if both fire in same iteration |
| G4 | `env.accepted[-1]` access | `env.accepted` is reset to `[]` in `env.reset()` | Must read BEFORE `env.reset()` is called; `on_step()` called after `env.step()` is fine |
| G5 | `StatefulSubstrateWrapper.__init__` patches `env.substrate_fn` | `env` is constructed before wrapper in `PPOTrainerScheduler.__init__` | Order preserved; env is passed to wrapper, which then patches it |

---

## 4. Integration Points in `train_ppo.py`

### Where to insert `self.stateful_wrapper = None` block
After line 181 (env construction), before line 192 (`self._obs, _ = self.env.reset()`):
```
line 181: )                          ← end of VNEOrderingEnv(...)
line 182:                            ← INSERT wrapper block here
...
line 192: self._obs, _ = self.env.reset()
```

### Where to insert Hook 1 (on_episode_start)
**Site A** — line 208–210 (empty vnr_list mid-loop):
```python
if not obs["vnr_list"]:
    if self.stateful_wrapper is not None:   # ← INSERT
        self.stateful_wrapper.on_episode_start()
    self._obs, _ = self.env.reset()
```

**Site B** — line 241–242 (done=True):
```python
if done:
    if self.stateful_wrapper is not None:   # ← INSERT
        self.stateful_wrapper.on_episode_start()
    self._obs, _ = self.env.reset()
```

### Where to insert Hook 2 (on_step)
After line 219:
```python
next_obs, reward, done, _, info = self.env.step(action.item())
if self.stateful_wrapper is not None:           # ← INSERT
    self.stateful_wrapper.on_step(action.item(), next_obs, reward, done, info)
```

### Where to insert TensorBoard logging
Inside the `if done:` block at line 230, after existing `writer.add_scalar` calls (lines 234–238):
```python
if self.stateful_wrapper is not None:
    live = self.stateful_wrapper.get_live_util()
    self.writer.add_scalar("StatefulSubstrate/LiveCpuUtil", live['live_cpu_util'], global_step)
    self.writer.add_scalar("StatefulSubstrate/LiveBwUtil",  live['live_bw_util'],  global_step)
    self.writer.add_scalar("StatefulSubstrate/CommittedVNRs", live['committed_vnrs'], global_step)
```

---

## 5. PPOConfig additions (line 104, after `load_checkpoint`)

```python
# ── Stateful substrate (distribution alignment) ──────────────────────────
stateful_substrate:          bool  = False
ss_warmup_episodes:          int   = 50
ss_vnr_lifetime_episodes:    int   = 5
ss_overflow_cpu_threshold:   float = 0.88
ss_regenerate_on_overflow:   bool  = True
```

And CLI args in `_build_parser()` (after line 511):
```python
p.add_argument("--stateful-substrate",    action="store_true")
p.add_argument("--ss-warmup",             type=int,   default=50)
p.add_argument("--ss-lifetime-episodes",  type=int,   default=5)
p.add_argument("--ss-overflow-threshold", type=float, default=0.88)
```

---

## 6. K Calibration for Default Config

Using the formula `K = round(avg_lifetime / (avg_inter_arrival × avg_batch_size))`:

- `make_batch_fn` defaults: `cpu_range=(10,40)`, `bw_range=(10,50)`, no lifetime field set on VNRs
- VNR graphs from `generate_single_vnr` — check if `lifetime` is set in graph attributes
- Batch size: uniform [5, 15] → avg = **10**
- If VNR lifetime is not set, `BatchedVNRSimulator` defaults to 50 (simulator.py:140)
  → avg = 50/2 = **25** (if uniform 0–50)
- Inter-arrival: 1.0 (assumed)
- **K = round(25 / (1.0 × 10)) = 3** ← matches plan's recommendation

For fig6 datasets (max_lifetime=300, Pareto): K ≈ 7.

---

## 7. Ready to Implement

All insertion points are now precisely identified. Implementation requires:
1. **New file**: `src/training/stateful_substrate.py` — per spec, with `max_cpu`/`max_bw` fix
2. **Modified file**: `src/training/train_ppo.py` — 4 insertion sites + 5 new `PPOConfig` fields + CLI args
