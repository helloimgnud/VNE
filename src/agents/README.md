# VNR Scheduler — Graph Pointer Network + PPO

> Thay thế tiêu chí sắp xếp đơn giản (`sort by revenue`) trong `hpso_batch.py` bằng một
> **agent RL được train** để quyết định thứ tự xử lý VNR tối ưu trong mỗi time window.

---

## Mục Lục

1. [Tổng Quan Vấn Đề](#1-tổng-quan-vấn-đề)
2. [Kiến Trúc Model](#2-kiến-trúc-model)
3. [MDP Formulation](#3-mdp-formulation)
4. [Chi Tiết Các Thành Phần](#4-chi-tiết-các-thành-phần)
5. [Luồng Hoạt Động Đầy Đủ](#5-luồng-hoạt-động-đầy-đủ)
6. [Cấu Trúc File](#6-cấu-trúc-file)
7. [Cách Sử Dụng](#7-cách-sử-dụng)
8. [Tích Hợp Với hpso_batch.py](#8-tích-hợp-với-hpso_batchpy)
9. [Hyperparameters](#9-hyperparameters)
10. [So Sánh Với GPRL Gốc](#10-so-sánh-với-gprl-gốc)

---

## 1. Tổng Quan Vấn Đề

### Vấn Đề Hiện Tại (`hpso_batch.py`)

```python
# hpso_batch.py dòng 40-41 — sắp xếp đơn giản bằng revenue
vnr_list.sort(key=lambda x: revenue_of_vnr(x), reverse=True)
```

Cách sắp xếp này **bỏ qua trạng thái substrate** và **bỏ qua tương tác giữa các VNR**.
Kết quả: thứ tự có thể không tối ưu, dẫn đến acceptance rate thấp hơn mức có thể đạt được.

### Vấn Đề Cần Giải Quyết

Trong một time window với K VNR đang chờ:
- Khi PSO embed VNR_i trước → chiếm tài nguyên substrate → thay đổi khả năng embed các VNR_j sau
- **Thứ tự quyết định kết quả**: `{VNR2→4→5}` có thể accept 3/3, còn `{VNR1→4→5}` chỉ accept 2/3
- Cần agent học được cách chọn VNR tiếp theo **dựa trên trạng thái tài nguyên hiện tại** và **đặc điểm của từng VNR đang chờ**

### Giải Pháp

```
RL Agent (Graph Pointer Network) quyết định thứ tự
         ↓
   PSO thực hiện embedding theo thứ tự đó
```

---

## 2. Kiến Trúc Model

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                     VNRSchedulerAgent                               │
 │                                                                     │
 │  Substrate G_p(t)          VNR Queue [G_v1 ... G_vK]               │
 │       │                            │                                │
 │       ▼                            ▼                                │
 │  ┌──────────────┐          ┌───────────────────┐                   │
 │  │ Substrate    │          │  VNR Encoder       │                   │
 │  │ Encoder      │          │  (GAT×3, shared    │                   │
 │  │ (GAT×3)      │          │   weights)         │                   │
 │  │              │          │                    │                   │
 │  │ mean-pool    │          │  max-pool per VNR  │                   │
 │  └──────┬───────┘          └─────────┬──────────┘                  │
 │         │ h_p∈R^128                  │ {h_vi}∈R^{K×64}             │
 │         │                            │                              │
 │         └──────────┬─────────────────┘                             │
 │                    │                                                │
 │                    ▼                                                │
 │           ┌─────────────────┐                                       │
 │           │  Context MLP    │  [h_p ‖ mean({h_vi})] → R^256        │
 │           └────────┬────────┘                                       │
 │                    │ context (= GRU init hidden)                    │
 │                    ▼                                                │
 │           ┌─────────────────────────────────────────┐              │
 │           │         Pointer Decoder (GRU)            │              │
 │           │                                          │              │
 │           │  For step t:                             │              │
 │           │    GRU(last_key, h_{t-1}) → d_t          │              │
 │           │    u^t_i = vᵀ tanh(W1·h_vi + W2·d_t)   │              │
 │           │    mask unavailable → softmax → p_i      │              │
 │           └──────────────────┬──────────────────────┘              │
 │                              │ action: VNR index                    │
 │                              ▼                                      │
 │           ┌─────────────────────────────┐                          │
 │           │       Critic Head           │                          │
 │           │  [h_p ‖ mean(h_active)]     │                          │
 │           │  → MLP → V(s_t)            │                          │
 │           └─────────────────────────────┘                          │
 └─────────────────────────────────────────────────────────────────────┘
```

---

## 3. MDP Formulation

### State `s_t`

```
s_t = (G_p^(t), {G_vi : i ∈ active_queue}, mask_t)
```

| Thành phần | Mô tả | Dimension |
|---|---|---|
| `G_p^(t)` | Substrate graph sau khi đã embed t-1 VNR | N_p nodes × 4 features |
| `{G_vi}` | Tất cả VNR còn trong queue | K_max graphs |
| `mask_t` | Vector bool: True = VNR chưa xử lý | (K_max,) |

**Node features của substrate** (`nfeat` dim=4):
```
[cpu_residual_ratio, mem_residual_ratio, avg_bw_ratio, utilisation_rate]
```

**Node features của VNR** (`nfeat` dim=3):
```
[cpu_demand_norm, mem_demand_norm, vnf_type_norm]
```

### Action `a_t`

```
a_t ∈ {0, 1, ..., K_max-1}   — index của VNR được chọn để HPSO xử lý tiếp theo
```
Chỉ chọn từ VNR có `mask_t[i] = True`.

### Reward `r_t`

```python
if HPSO embed thành công VNR_{a_t}:
    r_t = Revenue(VNR) / Cost(VNR)    # = R/C ratio (giống GPRL eq.21)
else:
    r_t = -0.1                         # penalty nhỏ cho embed thất bại
```

Terminal reward (cuối episode) có thể thêm:
```
r_terminal = α*(Σ R_accepted / Σ R_all) + β*(num_accepted / K)
```

---

## 4. Chi Tiết Các Thành Phần

### 4.1 SubstrateEncoder

```
Input : DGLGraph G_p, node features (N_p, 4)
Layers:
  GAT(4→64,  heads=4, concat) → (N_p, 256)
  GAT(256→64, heads=4, concat) → (N_p, 256)  
  GAT(256→128, heads=1, avg)   → (N_p, 128)
Readout: mean-pool over N_p nodes
Output: h_p ∈ R^128
```

Dùng mean-pool vì muốn tóm tắt **toàn bộ trạng thái tài nguyên** của substrate.

### 4.2 VNREncoder (shared weights)

```
Input : DGLGraph G_vi, node features (N_vi, 3)
Layers:
  GAT(3→32,  heads=4, concat) → (N_vi, 128)
  GAT(128→32, heads=4, concat) → (N_vi, 128)
  GAT(128→64, heads=1, avg)    → (N_vi, 64)
Readout: max-pool over N_vi nodes
Output: h_vi ∈ R^64
```

Dùng **max-pool** (khác GPRL) để capture "node đòi resource nhiều nhất" trong VNR —
node này quyết định khó hay dễ embed VNR đó.

**Shared weights** → cùng 1 bộ tham số encode tất cả VNR trong queue,
cho phép K thay đổi mà không cần retrain.

### 4.3 ContextMLP

```
Input : [h_p ‖ mean({h_vi : i active})]  ∈ R^(128+64) = R^192
Layers: Linear(192→256) → ReLU → Linear(256→256) → ReLU
Output: context ∈ R^256  (dùng làm GRU hidden state khởi tạo)
```

### 4.4 PointerDecoder (GRU + Attention)

Tại mỗi bước scheduling t:

**Bước 1 - KeyProjection** (thực hiện 1 lần trước episode):
```
keys[i] = Linear(64→256)(h_vi)    ∈ R^256    ∀i ∈ 0..K_max-1
```

**Bước 2 - GRU step:**
```
d_t = GRUCell(last_key, h_{t-1})   ∈ R^256
```
- `last_key`: key embedding của VNR đã chọn bước trước
- Bước đầu: `last_key = start_token` (learnable parameter)

**Bước 3 - Pointer Attention** (GPRL Eq. 18-19):
```
u^t_i = vᵀ · tanh(W1·keys[i] + W2·d_t)     ∈ R    ∀i

logits = u^t  masked bởi mask_t  (-inf cho inactive)

p_i = softmax(logits)
```

**Bước 4 - Action:**
```
Training:    a_t ~ Categorical(p)
Inference:   a_t = argmax(p)
```

### 4.5 CriticHead (Value Network)

Chia sẻ SubstrateEncoder và VNREncoder với actor. Chỉ head riêng:

```
Input : [h_p ‖ mean(h_active)]  ∈ R^192
MLP   : Linear(192→128) → ReLU → Linear(128→64) → ReLU → Linear(64→1)
Output: V(s_t)  ∈ R  (scalar value estimate)
```

### 4.6 PPO Training

**GAE Advantage:**
```
δ_t   = r_t + γ·V(s_{t+1}) - V(s_t)
Â_t   = Σ_{k=0}^{T-t} (γλ)^k · δ_{t+k}     [λ=0.95]
```

**Actor Loss (PPO-Clip):**
```
ratio_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

L_CLIP = E_t[ min(ratio_t · Â_t,  clip(ratio_t, 1-ε, 1+ε) · Â_t) ]
```

**Critic Loss:**
```
L_value = MSE(V_θ(s_t),  returns_t)    [returns = Â + V_old]
```

**Entropy Bonus:**
```
H_t = -Σ_i p_i · log(p_i)
```

**Total Loss (minimized):**
```
L_total = -L_CLIP + 0.5·L_value - 0.01·H
          ↑ actor     ↑ critic    ↑ exploration
```

**Backprop flow:**
```
L_total
 ├─► Actor head (GRU + W1/W2/v + start_token)
 │    └─► Shared (SubstrateEncoder + VNREncoder + ContextMLP + KeyProjection)
 └─► Critic head (CriticMLP)
      └─► Shared (SubstrateEncoder + VNREncoder)
```

---

## 5. Luồng Hoạt Động Đầy Đủ

### 5.1 Training Loop (1 Episode = 1 Time Window)

```
Episode bắt đầu:
  substrate_copy = copy(substrate)
  window = [VNR_1, VNR_2, ..., VNR_K]   (K ≤ K_max)

┌─────────────────────────────────────────────────────┐
│  ROLLOUT PHASE (no gradient)                        │
│                                                     │
│  Encode: h_p = SubstrateEncoder(G_p_init)           │
│          {h_vi} = VNREncoder(G_v1..K)               │
│          keys = KeyProj({h_vi})                     │
│          h_gru = ContextMLP(h_p, mean(h_vi))        │
│          last_key = start_token                     │
│                                                     │
│  For t = 1..K:                                      │
│    logits, h_gru = Decoder(keys, h_gru, last_key,   │
│                             mask)                   │
│    a_t ~ Categorical(softmax(logits))               │
│    log_prob_t = log π(a_t|s_t)                      │
│    value_t = CriticHead(h_p, mean(h_active))        │
│                                                     │
│    result = HPSO(substrate_copy, VNR_{a_t})         │
│    r_t = R/C if success else -penalty               │
│                                                     │
│    mask[a_t] = False                                │
│    last_key = keys[a_t]                             │
│    traces.append(s_t, a_t, r_t, log_prob_t, val_t) │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  UPDATE PHASE (with gradient, k_epochs times)       │
│                                                     │
│  Compute GAE: Â_t, returns_t                        │
│  Normalize: Â ← (Â - mean) / std                   │
│                                                     │
│  For epoch = 1..k_epochs:                           │
│    Re-run episode với fixed actions (evaluate_actions)│
│    → new log_probs, values, entropies               │
│                                                     │
│    ratio = exp(log_probs_new - log_probs_old)       │
│    L_CLIP = min(ratio·Â, clip(ratio,1±ε)·Â).mean() │
│    L_value = MSE(values_new, returns)               │
│    H = entropy.mean()                               │
│                                                     │
│    L = -L_CLIP + 0.5·L_value - 0.01·H              │
│    optimizer.zero_grad()                            │
│    L.backward()                                     │
│    clip_grad_norm_(params, 0.5)                     │
│    optimizer.step()                                 │
└─────────────────────────────────────────────────────┘
```

### 5.2 Inference (Sử Dụng Model Đã Train)

```python
# Thay thế hpso_batch() bằng hpso_batch_rl()
from vnr_scheduler import hpso_batch_rl, VNRSchedulerAgent
import torch

# Load model
ckpt  = torch.load('result/scheduler_model.pt')
agent = VNRSchedulerAgent(ckpt['cfg'])
agent.load_state_dict(ckpt['model_state'])
agent.eval()

# Sử dụng y như hpso_batch cũ
accepted, rejected = hpso_batch_rl(
    substrate=substrate_graph,
    batch=vnr_batch,
    agent=agent,
    verbose=True
)
```

---

## 6. Cấu Trúc File

```
project/
├── vnr_scheduler.py          ← Model chính (file mới)
│   ├── GATEncoder            # 3-layer GAT
│   ├── SubstrateEncoder      # GAT + mean-pool → h_p
│   ├── VNREncoder            # GAT + max-pool → h_vi  (shared)
│   ├── ContextMLP            # [h_p‖mean(h_vi)] → GRU init
│   ├── KeyProjection         # h_vi → pointer key space
│   ├── PointerDecoder        # GRU + attention → logits
│   ├── CriticHead            # MLP → V(s)
│   ├── VNRSchedulerAgent     # Full actor-critic agent
│   ├── PPOTrainer            # PPO training loop
│   ├── build_vnr_dgl()       # nx VNR → DGLGraph
│   ├── build_substrate_dgl() # nx substrate → DGLGraph
│   └── hpso_batch_rl()       # Drop-in cho hpso_batch.py
│
├── train_scheduler.py        ← Training script (file mới)
├── hpso_batch.py             ← GIỮ NGUYÊN (vẫn dùng được)
│
├── src/
│   ├── algorithms/
│   │   └── fast_hpso.py      # PSO solver (không thay đổi)
│   └── evaluation/
│       └── eval.py           # revenue_of_vnr, cost_of_vnr
│
└── result/
    └── scheduler_model.pt    # Model đã train
```

---

## 7. Cách Sử Dụng

### 7.1 Cài Đặt Dependencies

```bash
pip install torch dgl networkx
# DGL: https://www.dgl.ai/pages/start.html (chọn đúng CUDA version)
```

### 7.2 Chuẩn Bị Dữ Liệu

Model cần substrate graph và VNR graphs theo định dạng NetworkX với node attributes:

**Substrate nodes** phải có:
```python
node_data = {
    'cpu':     50.0,    # total CPU capacity
    'mem':     50.0,    # total memory capacity  
    'cpu_res': 30.0,    # residual CPU
    'mem_res': 25.0,    # residual memory
}
# edges phải có:
edge_data = {'bw': 100.0, 'bw_res': 80.0}
```

**VNR nodes** phải có:
```python
node_data = {
    'cpu':      15.0,   # CPU demand (hoặc 'cpu_demand')
    'mem':      10.0,   # memory demand
    'vnf_type': 2,      # VNF type ID (0..4)
}
```

Nếu attributes của bạn có tên khác, override `build_vnr_dgl()` và `build_substrate_dgl()`:

```python
def my_build_vnr(vnr_nx):
    # ... tùy chỉnh attribute extraction ...
    g.ndata['nfeat'] = torch.tensor(feats, dtype=torch.float32)
    return g
```

### 7.3 Training

```bash
python train_scheduler.py \
    --episodes 500 \
    --K_max 20 \
    --window_size 10 \
    --save_path result/scheduler_model.pt \
    --lr 3e-4 \
    --gamma 0.98
```

### 7.4 Tích Hợp Vào Code Của Bạn

**Option A: Drop-in replacement (đơn giản nhất)**

```python
# Trước đây:
from hpso_batch import hpso_embed_batch
accepted, rejected = hpso_embed_batch(substrate, batch)

# Sau khi tích hợp:
from vnr_scheduler import hpso_batch_rl, VNRSchedulerAgent
import torch

agent = VNRSchedulerAgent()
ckpt  = torch.load('result/scheduler_model.pt')
agent.load_state_dict(ckpt['model_state'])

accepted, rejected = hpso_batch_rl(substrate, batch, agent=agent)
```

**Option B: Custom builders** (khi attribute names khác)

```python
accepted, rejected = hpso_batch_rl(
    substrate, batch, agent,
    build_sub_dgl_fn=my_build_substrate,
    build_vnr_dgl_fn=my_build_vnr,
)
```

**Option C: Chỉ lấy ordering từ agent** (tích hợp thủ công)

```python
import torch
from vnr_scheduler import VNRSchedulerAgent, build_vnr_dgl, build_substrate_dgl

agent.eval()
g_sub = build_substrate_dgl(substrate)
vnr_dgl = [build_vnr_dgl(v) for v in vnr_list]
padded  = vnr_dgl + [None] * (agent.K_max - len(vnr_list))

with torch.no_grad():
    traj = agent.rollout(g_sub, padded, len(vnr_list), deterministic=True)

order = traj['actions']   # [3, 1, 0, 2, ...] — thứ tự index VNR

# Bây giờ bạn có thể dùng order với bất kỳ solver nào
for idx in order:
    result = your_solver(substrate, vnr_list[idx])
```

---

## 8. Tích Hợp Với hpso_batch.py

File `hpso_batch.py` **giữ nguyên không thay đổi**. `vnr_scheduler.py` chỉ **thêm vào**:

```
hpso_batch.py  (giữ nguyên)         vnr_scheduler.py  (thêm mới)
─────────────────────────           ──────────────────────────────
hpso_embed_batch()                  hpso_batch_rl()   ← gọi hpso_embed() nội bộ
  sort by revenue (heuristic)         agent.rollout() → order
  HPSO embed loop                     HPSO embed loop (same)
```

`hpso_batch_rl()` gọi `hpso_embed()` từ `fast_hpso.py` theo đúng cách `hpso_batch.py` đang làm —
chỉ thay đổi **thứ tự** VNR được đưa vào solver.

Khi `agent=None`, `hpso_batch_rl()` tự động fallback về behaviour gốc (sort by revenue).

---

## 9. Hyperparameters

| Parameter | Default | Mô Tả |
|---|---|---|
| `substrate_hidden` | 128 | Dim output SubstrateEncoder |
| `vnr_hidden` | 64 | Dim output VNREncoder |
| `substrate_gat_heads` | [4,4,1] | Số attention heads mỗi lớp GAT |
| `vnr_gat_heads` | [4,4,1] | Số attention heads mỗi lớp GAT |
| `context_dim` | 256 | Dim ContextMLP output (= GRU hidden) |
| `gru_hidden` | 256 | GRU hidden size trong PointerDecoder |
| `critic_hidden` | 128 | Hidden size CriticHead MLP |
| `K_max` | 20 | Max VNRs per window (padding target) |
| `gamma` | 0.98 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_eps` | 0.2 | PPO clipping epsilon |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |
| `value_coef` | 0.5 | Value loss coefficient |
| `lr` | 3e-4 | Learning rate (Adam) |
| `k_epochs` | 4 | PPO update epochs per episode |

---

## 10. So Sánh Với GPRL Gốc

| Khía Cạnh | GPRL (paper) | VNRScheduler (này) |
|---|---|---|
| **Quyết định** | Node-level: chọn physical node cho virtual node | Request-level: chọn VNR tiếp theo |
| **Encoder input** | 1 VNR + 1 substrate | K VNRs + 1 substrate |
| **Pointer output** | Physical node index (N_p options) | VNR index (K options) |
| **Episode length** | \|N_v\| steps (per VNR) | K steps (per time window) |
| **Readout VNR** | Không rõ trong paper | max-pool (capture hardest node) |
| **Readout substrate** | Không rõ trong paper | mean-pool (global resource state) |
| **Solver** | PPO trực tiếp map nodes | PPO orders → HPSO maps nodes |
| **Shared encoder** | 2 separate GATs | Same GAT shared for all K VNRs |
| **GRU context init** | x0 (zeros) | ContextMLP(h_p, mean(h_vi)) |

---

## Lưu Ý Quan Trọng

### Attribute Mapping

Hàm `build_vnr_dgl()` và `build_substrate_dgl()` cần **match attribute names** của graph của bạn.
Kiểm tra các keys trong `vnr.nodes[n]` và `substrate.nodes[n]` rồi điều chỉnh:

```python
# Xem attributes hiện tại
import networkx as nx
vnr_sample = vnr_list[0]
print(dict(vnr_sample.nodes[0]))       # → {'cpu': 15.0, 'mem': 10.0, ...}
print(dict(substrate.nodes[0]))        # → {'cpu_res': 30.0, ...}
```

### reward_of_vnr và cost_of_vnr

`PPOTrainer.compute_reward()` import từ `src.evaluation.eval`. Đảm bảo:
```python
from src.evaluation.eval import revenue_of_vnr, cost_of_vnr
```
Nếu function signatures khác, override static method này.

### Substrate Mutation

`hpso_embed()` từ `fast_hpso.py` **mutate** substrate khi thành công (reserve resources).
`collect_and_update()` dùng `copy.deepcopy(substrate)` để tránh ảnh hưởng substrate gốc
trong training. Trong inference, substrate được update trực tiếp (như `hpso_batch.py` gốc).
