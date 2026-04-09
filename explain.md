# Training Explained: RL-Based VNR Scheduler

## Q1: Why does 200 VNRs train for 300 **episodes**?

These are **two completely different concepts**. The 200 VNRs are a **pool** (a fixed reservoir to sample from), not a training batch.

```
VNR Pool (200 VNRs) ─── fixed reservoir, created once ─────────────────────────┐
                                                                                 │
 Episode 1:  sample 10 random VNRs  →  schedule  →  HPSO  →  rewards  →  PPO  │
 Episode 2:  sample 10 random VNRs  →  schedule  →  HPSO  →  rewards  →  PPO  │
 Episode 3:  sample 10 random VNRs  →  schedule  →  HPSO  →  rewards  →  PPO  │
   ...                                                                           │
 Episode 300: sample 10 random VNRs →  schedule  →  HPSO  →  rewards  →  PPO  ┘
```

- **`window_size=10`** → each episode picks 10 VNRs from the pool via `random.sample()`
- **`episodes=300`** → the PPO agent gets 300 gradient updates
- With C(200,10) ≈ 22 **trillion** possible subsets, the pool provides near-infinite variety

The 200-VNR pool is analogous to a training dataset; each episode draws a mini-batch from it.

---

## Q2: How does one episode work? (Full Data Flow)

```
STEP 0 – SAMPLE
  window = random.sample(vnr_pool, 10)       # 10 VNRs for this episode
  sub_copy = deepcopy(substrate)             # fresh substrate (no resource leakage)

STEP 1 – ENCODE (no gradient)
  g_sub  = build_substrate_dgl(sub_copy)     # NetworkX → DGLGraph
  h_p    = SubstrateEncoder(g_sub)           # GAT × 3 → mean-pool → R^128

  for each VNR_i in window:
      g_vi    = build_vnr_dgl(VNR_i)         # NetworkX → DGLGraph
      h_vi    = VNREncoder(g_vi)             # GAT × 3 → max-pool  → R^64
  keys    = KeyProjection(h_vi)              # project to GRU key space R^256

STEP 2 – ROLLOUT (actor, no gradient, stochastic)
  mask    = [T, T, T, T, T, T, T, T, T, T, F, ..., F]  # 10 real, 10 padding
  context = ContextMLP([h_p || mean(h_vi)])  # initialise GRU state R^256
  h_gru   = context

  for t = 0 → 9:
      logits, h_gru = PointerDecoder(keys, h_gru, last_key, mask)
      probs         = softmax(logits)        # only active slots have prob > 0
      action_t      = sample(probs)          # index of next VNR to embed
      mask[action_t] = False                 # remove from queue
      record (action_t, log_pi_old, V(s_t))

STEP 3 – EXECUTE HPSO (black-box solver)
  for each action_t in [a_0, a_1, ..., a_9]:
      result = hpso_embed(sub_copy, VNR[action_t])   # MUTATES sub_copy
      r_t = Revenue/Cost  if accepted
            -0.1           if rejected

STEP 4 – COMPUTE ADVANTAGES (GAE)
  delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
  A_t     = sum_{k>=0} (gamma*lambda)^k * delta_{t+k}   # gamma=0.98, lambda=0.95

STEP 5 – PPO UPDATE (k_epochs=4 gradient steps, WITH gradient)
  Re-run evaluate_actions() with current policy weights to get log_pi_new:

  for epoch in range(4):
      log_pi_new, V_new, entropy = agent.evaluate_actions(g_sub, vnr_dgls, actions)
      ratio      = exp(log_pi_new - log_pi_old)          # importance ratio
      surr1      = ratio * Advantage
      surr2      = clip(ratio, 1-eps, 1+eps) * Advantage  # eps=0.2
      L_policy   = -min(surr1, surr2).mean()
      L_value    = MSE(V_new, returns)
      L_entropy  = -entropy.mean()
      Loss = L_policy + 0.5*L_value + 0.01*L_entropy
      Loss.backward(); clip_grad_norm(0.5); Adam.step()
```

---

## Q3: Why is AccRate=1.000, RC=1.000 throughout?

The synthetic data is **too easy** for HPSO. Here's why:

| Parameter | Value | Effect |
|-----------|-------|--------|
| Substrate | 30 nodes, 141 edges | Dense, well-connected |
| Substrate CPU per node | 50–150 | Total ≈ 3000 CPU units |
| VNR CPU demand | 5–20 per virtual node | Each VNR needs ≈ 30–90 total |
| VNR size | 3–8 nodes | Small relative to substrate |

A 30-node substrate with 3000 CPU can trivially fit 10 VNRs requiring ~90 CPU each. **HPSO never fails**, so `accepted = 10/10` every episode.

This means **the ordering doesn't matter** — whether you process VNR_1 first or VNR_7 first, they all succeed anyway. The reward `r_t = R/C` is positive and nearly constant → the agent receives no gradient signal that distinguishes one ordering from another.

> **The agent is not learning; it is stuck in reward saturation.**

### How to produce a meaningful training signal

```bash
# Option 1: tighter substrate (harder to fit all VNRs)
python -m src.scripts.train_rl_scheduler \
    --substrate_nodes 15 \
    --vnr_pool_size 200 \
    --vnr_min_nodes 4 \
    --vnr_max_nodes 10 \
    --episodes 500

# Option 2: use a real VNR stream from the dataset generator
python -m src.generators.dataset_generator --experiments fig8
python -m src.scripts.train_rl_scheduler \
    --data_path dataset/fig8/vnr_stream.pkl \
    --episodes 1000
```

**Target**: AccRate should fluctuate between 0.5–0.9 for the reward difference across orderings to be measurable.

---

## Q4: What does the network actually learn?

The agent learns a **context-dependent priority function** over VNRs:

> *"Given the current substrate state h_p and the set of pending VNRs, which VNR should I embed next?"*

The intuition behind good ordering:
1. **Resource-heavy VNRs first** — embed while the substrate is fresh with full capacity
2. **Competing VNRs serialised** — avoid two VNRs fighting over the same bottleneck node
3. **Likely-to-fail VNRs last** — defer if deferral has no cost, or skip if success is unlikely

The pointer network generalises this via a **soft attention over all VNR embeddings** weighted by the current substrate state h_p — an expressive function that is strictly richer than the fixed revenue-sort heuristic.

---

## Summary Table

| Concept | Value | Meaning |
|---------|-------|---------|
| VNR pool | 200 | Reservoir to sample windows from |
| episodes | 300 | Number of PPO weight updates |
| window_size | 10 | VNRs chosen per episode |
| AccRate=1.0 | Bad sign | Substrate too easy; no learning signal |
| Loss fluctuates | Expected | Policy entropy term prevents collapse |
| Next step | Harder data | Substrate ~50% saturated → useful signal |
