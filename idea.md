# Ý Tưởng: RL-Based VNR Scheduling với Graph Pointer Network

## 1. Tổng Quan Bài Toán

### 1.1 Bài Toán Gốc (GPRL)
- **Quyết định**: Node-level — chọn physical node nào để map virtual node lên
- **Input**: 1 VNR + substrate network
- **Output**: Sequence ánh xạ node ảo → node thực

### 1.2 Bài Toán Của Bạn
- **Quyết định**: Request-level — trong 1 time window có N VNR đang chờ, chọn VNR nào để PSO xử lý tiếp theo
- **PSO**: Đã xử lý việc ánh xạ node ảo → node thực (fixed, không thay đổi)
- **RL**: Học cách sắp xếp thứ tự giải quyết các VNR để maximize revenue/acceptance rate

### 1.3 Tại Sao Thứ Tự Quan Trọng?
Tài nguyên substrate là hữu hạn và dùng chung. Khi PSO giải VNR_i trước, nó sẽ chiếm tài nguyên, làm thay đổi trạng thái mạng cho các VNR_j sau. Do đó:
- {VNR2, VNR4, VNR5} theo thứ tự 2→4→5 có thể accept cả 3
- {VNR1, VNR4, VNR5} theo thứ tự 1→4→5 có thể chỉ accept được 2

---

## 2. Kiến Trúc Tổng Thể

```
Time Window t: [VNR_1, VNR_2, ..., VNR_K] chờ xử lý

                    ┌─────────────────────────────┐
                    │       RL AGENT (Actor)       │
                    │                              │
  Substrate         │  ┌──────────────────────┐   │
  Network  ────────►│  │  Substrate Encoder   │   │
  G_p(t)            │  │  (GAT layers)        │   │
                    │  └──────────┬───────────┘   │
                    │             │ h_p            │
                    │             ▼                │
  Pending VNRs      │  ┌──────────────────────┐   │
  [G_v1,...,G_vK]──►│  │  VNR Set Encoder     │   │
                    │  │  (GAT per VNR)       │   │
                    │  └──────────┬───────────┘   │
                    │             │ {h_v1,...h_vK} │
                    │             ▼                │
                    │  ┌──────────────────────┐   │
                    │  │  Pointer Network     │   │
                    │  │  Decoder             │   │──► chọn VNR_i
                    │  └──────────────────────┘   │
                    └─────────────────────────────┘
                                  │
                                  ▼ VNR_i được chọn
                    ┌─────────────────────────────┐
                    │    PSO Solver               │
                    │    Map VNR_i → Substrate    │
                    └─────────────────────────────┘
                                  │
                                  ▼
                    Cập nhật trạng thái substrate G_p(t+1)
                    Loại VNR_i khỏi danh sách chờ
                    Lặp lại với VNR còn lại
```

---

## 3. Markov Decision Process (MDP) Formulation

### 3.1 State s_t

State tại bước t bao gồm:

**Substrate Network State** `S_substrate`:
- Topology: adjacency matrix A_p ∈ R^{N_p × N_p}
- Node features ma trận X_p ∈ R^{N_p × d_node}:
  - CPU residual ratio: CPU_res / CPU_total
  - Memory residual ratio: MEM_res / MEM_total
  - Số VNF đang chạy hiện tại
  - Node utilization rate
- Edge features ma trận E_p ∈ R^{|E_p| × d_edge}:
  - Bandwidth residual ratio: BW_res / BW_total
  - Link utilization

**VNR Queue State** `S_queue`:
- Danh sách K VNR đang chờ, mỗi VNR_i gồm:
  - DAG topology: A_vi ∈ R^{n_i × n_i}
  - Virtual node features X_vi ∈ R^{n_i × d_vnode}:
    - CPU demand
    - Memory demand
    - VNF type (one-hot)
  - Virtual edge features: bandwidth demand
  - Deadline / time remaining
  - Số node ảo |N_vi|

**Mask vector** `m_t` ∈ {0,1}^K:
- m_t[i] = 0 nếu VNR_i đã được xử lý hoặc không còn khả thi
- m_t[i] = 1 nếu VNR_i vẫn trong queue

### 3.2 Action a_t

```
a_t ∈ {1, 2, ..., K}  — chỉ số của VNR được chọn để PSO xử lý tiếp theo
```

- Chỉ chọn từ các VNR có mask m_t[i] = 1
- Pointer network output xác suất p(a_t | s_t) cho từng VNR

### 3.3 Reward r_t

Reward được thiết kế để khuyến khích maximize revenue và acceptance rate:

```
r_t = R/C_i     nếu PSO embed thành công VNR_i
r_t = -penalty  nếu PSO thất bại (không đủ tài nguyên)
r_t = 0         nếu không còn VNR nào trong queue
```

**Chi tiết:**
```
R/C_i = Revenue_i / Cost_i

Revenue_i = Σ_{nv ∈ Nv_i} Σ_r d^r_nv  +  Σ_{ev ∈ Ev_i} b_ev

Cost_i     = Σ_{nv,np} x^np_nv * Σ_r d^r_nv * c^r_np
           + Σ_{ev,uv} y^uv_ev * b^uv_ev * c_uv
```

**Có thể thêm shaping reward tại cuối episode:**
```
r_terminal = α * (Σ accepted R_i / Σ all Revenue_i)   # acceptance-weighted revenue
           + β * (num_accepted / K)                     # acceptance rate
```

Điều này giúp agent học quan điểm dài hạn (long-term return), không chỉ greedy từng bước.

---

## 4. Actor Network — Chi Tiết Kiến Trúc

### 4.1 Substrate Graph Encoder

**Input**: G_p = (X_p, A_p, E_p)

**Architecture**: 3 lớp GAT stacked

```
Layer 1: GAT(d_in=d_node, d_out=64, heads=4, concat=True)  → 256-dim
Layer 2: GAT(d_in=256,    d_out=64, heads=4, concat=True)  → 256-dim
Layer 3: GAT(d_in=256,    d_out=128, heads=1, concat=False) → 128-dim
```

**Attention weight** (như paper GPRL):
```
α_ij = softmax_j( LeakyReLU( a^T [W*h_i || W*h_j] ) )

h_i^(l+1) = σ( Σ_{j∈N_i} α_ij * W * h_j^(l) )
```

**Global graph embedding** của substrate:
```
h_p = mean_pool( {h^(3)_np : np ∈ N_p} )   ∈ R^128
```

Đây là vector tóm tắt toàn bộ trạng thái tài nguyên hiện tại của substrate.

### 4.2 VNR Graph Encoder

**Input**: Mỗi VNR_i = (X_vi, A_vi) — DAG topology

**Architecture**: Tương tự substrate encoder nhưng chia sẻ weights cho tất cả VNR (parameter sharing):

```
Layer 1: GAT(d_in=d_vnode, d_out=32, heads=4) → 128-dim
Layer 2: GAT(d_in=128,     d_out=32, heads=4) → 128-dim  
Layer 3: GAT(d_in=128,     d_out=64, heads=1) → 64-dim
```

**VNR embedding** (readout):
```
h_vi = max_pool( {h^(3)_nv : nv ∈ N_vi} )   ∈ R^64
```

Sử dụng max_pool để capture đặc trưng "nút khó nhất" trong VNR (critical resource node).

**Optional**: thêm summary features sau pooling:
```
feat_vi = [h_vi || log(|N_vi|) || max_demand_cpu || max_demand_bw]  ∈ R^67
```

### 4.3 Context Vector

Trước khi đưa vào decoder, tổng hợp context:

```
context = MLP( [h_p || mean({h_vi : i active}) || step_encoding] )
        ∈ R^256
```

- `h_p`: trạng thái substrate hiện tại
- `mean({h_vi})`: trạng thái trung bình các VNR còn lại
- `step_encoding`: positional encoding của bước hiện tại trong episode

### 4.4 Pointer Network Decoder

**Mục tiêu**: Tính xác suất p(a_t = i | s_t) cho mỗi VNR_i còn lại.

**GRU Decoder** (stateful, theo dõi lịch sử các lựa chọn):
```
d_t, c_t = GRU(input=h_v_{last_chosen}, hidden=c_{t-1})
```
- `h_v_{last_chosen}`: embedding của VNR đã chọn ở bước trước (teacher forcing khi train)
- Khởi tạo c_0 = context vector

**Pointer Attention** (như GPRL eq 18-19):
```
u^t_i = v^T * tanh( W_1 * h_vi + W_2 * d_t )    ∀i ∈ active_queue

p_i = softmax( u^t_i )    với mask m_t (set u^t_i = -∞ nếu m_t[i]=0)
```

**Action sampling**:
```
# Khi training: sample từ distribution
a_t ~ Categorical(p_1, ..., p_K)

# Khi inference: greedy
a_t = argmax_i p_i
```

---

## 5. Critic Network

### 5.1 Kiến Trúc

Critic chia sẻ 2 encoder (substrate + VNR) với actor network (shared backbone), chỉ có head riêng.

```
Shared:
  h_p      = SubstrateEncoder(G_p)          ∈ R^128
  {h_vi}   = VNREncoder({G_vi})             ∈ R^{K×64}

Critic-specific head:
  h_queue  = mean({h_vi : i active})        ∈ R^64
  h_concat = [h_p || h_queue]               ∈ R^192

  V(s_t) = MLP(h_concat)
  MLP: Linear(192→128) → ReLU → Linear(128→64) → ReLU → Linear(64→1)
```

### 5.2 Critic Loss (Value Loss)

```
L_critic = MSE( V(s_t), R̂_t )

Trong đó R̂_t là rewards-to-go (GAE):
R̂_t = Σ_{k=0}^{T-t} (γ^k * r_{t+k})
```

**Generalized Advantage Estimation (GAE)**:
```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)

Â_t = Σ_{k=0}^{T-t} (γλ)^k * δ_{t+k}
```
với λ=0.95 (bias-variance tradeoff).

---

## 6. PPO Training

### 6.1 Actor Loss (Policy Loss)

```
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

L_CLIP(θ) = E_t[ min(
    r_t(θ) * Â_t,
    clip(r_t(θ), 1-ε, 1+ε) * Â_t
) ]
```

với ε = 0.2 (clipping parameter).

**Entropy bonus** (khuyến khích exploration):
```
H_t = -Σ_i p_i * log(p_i)   (entropy của policy distribution)

L_policy(θ) = -L_CLIP(θ) - α_H * H_t
```

### 6.2 Total PPO Loss

```
L_total = L_policy(θ_actor) + c_v * L_critic(θ_critic)

c_v = 0.5  (value loss coefficient)
```

### 6.3 Backpropagation Flow

```
L_total
    │
    ├── L_policy → Actor Head (Pointer Decoder + GRU)
    │       └── (shared) GAT Encoders
    │
    └── L_critic → Critic Head (MLP)
            └── (shared) GAT Encoders
```

**Chú ý**: Shared encoder nhận gradient từ cả actor và critic. Điều này giúp encoder học representations tốt hơn.

### 6.4 PPO Training Algorithm

```
Algorithm: PPO Training cho VNR Scheduler

Initialize: Actor θ_A, Critic θ_C với random weights

For episode = 1 to N_episodes:
    Reset: Lấy time window mới với K VNR, substrate G_p
    
    # Rollout phase
    traces = []
    For t = 1 to K:  # K bước trong 1 episode (K = số VNR)
        s_t = (G_p_current, active_VNRs, mask)
        
        a_t ~ π_θ_A(·|s_t)           # Actor chọn VNR
        log_prob_t = log π_θ_A(a_t|s_t)
        
        PSO_embed(VNR_{a_t}, G_p_current)  # PSO xử lý VNR
        r_t = compute_reward(result)
        
        Update G_p_current (tài nguyên bị chiếm)
        Remove VNR_{a_t} từ active_VNRs
        
        s_{t+1} = (G_p_current, active_VNRs_updated, mask_updated)
        
        traces.append((s_t, a_t, r_t, log_prob_t, V(s_t)))
    
    # Compute advantages
    Compute R̂_t, Â_t từ traces (GAE)
    
    # Update phase (multiple epochs)
    For epoch = 1 to K_epochs:
        Sample mini-batches từ traces
        
        Compute L_CLIP, L_value, H
        L_total = -L_CLIP + 0.5*L_value - 0.01*H
        
        Gradient step: θ_A, θ_C ← optimizer(L_total)
        
    Update θ_old ← θ_A  (for next episode's ratio computation)
```

---

## 7. Xử Lý Variable-Size Queue

Một thách thức quan trọng: K (số VNR trong queue) có thể thay đổi giữa các time window.

### 7.1 Masking Strategy
```python
# Pointer output trước mask
logits = [u^t_1, u^t_2, ..., u^t_K_max]

# Apply mask (đã processed hoặc không tồn tại)
logits_masked = logits + (1 - mask) * (-1e9)

# Softmax
probs = softmax(logits_masked)
```

### 7.2 Padding
- Pad queue đến K_max VNR với dummy entries
- Dummy VNR có mask = 0, không bao giờ được chọn
- K_max = max VNR per window dựa trên thiết lập thực nghiệm

---

## 8. Interaction với PSO

### 8.1 Interface Design
```
RL Agent → PSO Interface:
    Input:  VNR_i (DAG + resource demands)
            G_p_current (substrate state)
    Output: Embedding result {success/fail, node_mapping, link_mapping}
            Updated G_p (sau khi cấp phát tài nguyên)
```

### 8.2 Handling PSO Failure
Khi PSO thất bại (không tìm được embedding hợp lệ):
- Reward = -penalty (e.g., penalty = 0.1)
- VNR bị loại khỏi queue (mask = 0)
- Tiếp tục với VNR kế tiếp

**Lưu ý thiết kế**: RL agent sẽ học được rằng nên tránh chọn các VNR có ít khả năng embed thành công (do resource constraints), từ đó tối ưu acceptance rate một cách gián tiếp.

---

## 9. Sự Khác Biệt Với GPRL Gốc

| Khía Cạnh | GPRL | Proposed |
|---|---|---|
| Quyết định | Chọn physical node cho virtual node | Chọn VNR tiếp theo từ queue |
| Encoder input | 1 VNR + 1 substrate | K VNRs + 1 substrate |
| Pointer output | Physical node index | VNR index |
| Decoder target | Node embedding sequence | VNR scheduling sequence |
| PSO | Không dùng | Sub-solver cho node embedding |
| Episode length | |N_v| steps (per VNR) | K steps (per time window) |

---

## 10. Thiết Kế Thực Nghiệm

### 10.1 Hyperparameters Gợi Ý

```yaml
# Encoders
gat_layers: 3
gat_heads: [4, 4, 1]
substrate_embed_dim: 128
vnr_embed_dim: 64

# Decoder
gru_hidden_dim: 256
pointer_dim: 128

# PPO
lr: 3e-4
gamma: 0.98
lambda_gae: 0.95
epsilon_clip: 0.2
entropy_coef: 0.01
value_coef: 0.5
k_epochs: 4
batch_size: 64

# Training
episodes: 500
steps_per_episode: K (= số VNR per window)
```

### 10.2 Evaluation Metrics
- **Acceptance Rate**: num_accepted / K per window
- **Revenue-Cost Ratio**: Σ Revenue_accepted / Σ Cost_accepted
- **Long-term Average Revenue**: theo rolling window
- **Comparison**: So sánh với FIFO ordering, Random ordering, Greedy (chọn VNR nhỏ nhất trước), GPRL-adapted

### 10.3 Ablation Studies Gợi Ý
1. Thay GAT encoder → GCN encoder (như GPRL baseline Kolin)
2. Bỏ GRU decoder → dùng MLP đơn giản
3. Bỏ entropy bonus → kiểm tra exploration
4. Thay PSO → random embedding
5. Thay max_pool → mean_pool cho VNR readout

---

## 11. Ưu Điểm Của Thiết Kế Này

1. **Tái sử dụng kiến trúc**: GAT encoder từ GPRL đã được chứng minh hiệu quả trong việc encode graph topology, tái dùng cho cả substrate và VNR encoder.

2. **Pointer network phù hợp**: Variable-length input (K VNR thay đổi giữa các window) được xử lý tự nhiên bởi pointer attention.

3. **Tách biệt concerns**: RL chịu trách nhiệm scheduling (thứ tự), PSO chịu trách nhiệm embedding (node mapping) — mỗi module làm tốt việc của mình.

4. **Online learning**: Agent có thể cập nhật policy liên tục khi network state thay đổi, tương tự GPRL.

5. **Scalability**: Parameter sharing giữa các VNR encoder cho phép xử lý queue size thay đổi mà không cần retrain.
