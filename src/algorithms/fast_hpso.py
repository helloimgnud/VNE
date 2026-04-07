# v2
# src/algorithms/fast_hpso.py
import random
import math
import copy
from utils.graph_utils import shortest_path_with_capacity, reserve_node, reserve_path
import networkx as nx
INFEASIBLE_PENALTY = 1e9

def fast_fitness(particle, substrate_graph, vnr_graph):
    mapping = {}
    vnodes = list(vnr_graph.nodes())

    # === Injective check: không cho phép 2 node ảo khác nhau ánh xạ vào cùng 1 node thực ===
    if len(set(particle)) < len(particle):
        return INFEASIBLE_PENALTY

    # Check Node Constraints
    for i, v in enumerate(vnodes):
        s = particle[i]
        if substrate_graph.nodes[s]['cpu'] < vnr_graph.nodes[v]['cpu']:
            return INFEASIBLE_PENALTY
        mapping[v] = s

    # === Estimate link cost via hop distance (proxy) ===
    est_link_cost = 0
    for (u, v) in vnr_graph.edges():
        s_u = mapping[u]
        s_v = mapping[v]
        
        if s_u != s_v:
            # Tính số hop giữa 2 node thực bằng shortest path length
            try:
                hop_distance = nx.shortest_path_length(substrate_graph, s_u, s_v)
            except nx.NetworkXNoPath:
                # Không có đường đi giữa 2 node thực
                return INFEASIBLE_PENALTY
            
            # Chi phí ước lượng = số hop × bandwidth yêu cầu
            bw_demand = vnr_graph.edges[u, v]['bw']
            est_link_cost += hop_distance * bw_demand

    return est_link_cost


def build_and_reserve(particle, substrate_graph, vnr_graph):
    """
    Hàm này chạy Dijkstra thật sự để kiểm tra tính khả thi toàn diện.
    Chỉ gọi khi cần kiểm tra kết quả cuối cùng.
    """
    mapping = {}
    vnodes = list(vnr_graph.nodes())

    if len(set(particle)) < len(particle):
        return None, None

    # 1. Node Mapping & Check CPU
    for i, v in enumerate(vnodes):
        s = particle[i]
        if substrate_graph.nodes[s]['cpu'] < vnr_graph.nodes[v]['cpu']:
            return None, None
        mapping[v] = s

    # 2. Link Mapping (Dijkstra)
    link_paths = {}
    for (u, v) in vnr_graph.edges():
        bw = vnr_graph.edges[u, v]['bw']
        path = shortest_path_with_capacity(
            substrate_graph, mapping[u], mapping[v], bw
        )
        if path is None:
            return None, None
        link_paths[(u, v)] = path

    # 3. Reserve Resources (Update Graph State)
    for v, s in mapping.items():
        reserve_node(substrate_graph, s, vnr_graph.nodes[v]['cpu'])
    for (u, v), path in link_paths.items():
        reserve_path(substrate_graph, path, vnr_graph.edges[u, v]['bw'])

    return mapping, link_paths

# =========================================================
# 2. Particle Initialization Strategy
# =========================================================
def init_particles_hpso(substrate_graph, vnr_graph, particles):
    vnodes = sorted(
        vnr_graph.nodes(),
        key=lambda v: vnr_graph.nodes[v]['cpu'],
        reverse=True
    )

    sub_nodes_sorted = sorted(
        substrate_graph.nodes(),
        key=lambda s: substrate_graph.nodes[s]['cpu'],
        reverse=True
    )

    swarm = []

    for _ in range(particles):
        mapping = {}
        used = set() # Tránh trùng lặp node vật lý trong 1 particle (tùy chọn)

        valid_particle = True
        temp_particle = [None] * len(vnodes)
        
        # Mapping index mapping for correct order in particle array
        vnode_to_idx = {node: i for i, node in enumerate(vnr_graph.nodes())}

        for v in vnodes:
            v_cpu = vnr_graph.nodes[v]['cpu']
            # Lọc candidate
            candidates = [
                s for s in sub_nodes_sorted
                if substrate_graph.nodes[s]['cpu'] >= v_cpu
                and s not in used
            ]
            
            if not candidates:
                valid_particle = False
                break

            # Chọn ngẫu nhiên trong Top-K để đảm bảo chất lượng khởi tạo
            k = random.randint(1, min(3, len(candidates)))
            s = random.choice(candidates[:k])
            
            # Gán vào vị trí tương ứng
            idx = vnode_to_idx[v]
            temp_particle[idx] = s
            used.add(s) 

        if valid_particle:
            swarm.append(temp_particle)

    # Nếu không tạo đủ, fill bằng random thuần túy để tránh crash
    while len(swarm) < particles:
        sub_keys = list(substrate_graph.nodes())
        if len(sub_keys) < len(vnr_graph.nodes()):
            break  # không thể injective

        random.shuffle(sub_keys)
        swarm.append(sub_keys[:len(vnr_graph.nodes())])


    return swarm

# =========================================================
# 3. PSO Operators
# =========================================================
def operation_minus(Xi, Xj):
    return [1 if Xi[k] == Xj[k] else 0 for k in range(len(Xi))]

def operation_plus(p, Vi, q, Vj):
    if p + q == 0:
        return Vi.copy()

    p_norm = p / (p + q)
    res = []
    for i in range(len(Vi)):
        if Vi[i] == Vj[i]:
            res.append(Vi[i])
        else:
            res.append(Vi[i] if random.random() < p_norm else Vj[i])
    return res


def operation_multiply(Xi, V, vnr_graph, substrate_graph):
    Xnew = Xi.copy()
    vnodes = list(vnr_graph.nodes())
    used = set(Xnew)

    for i in range(len(V)):
        if V[i] == 0:
            v = vnodes[i]
            v_cpu = vnr_graph.nodes[v]['cpu']

            used.discard(Xnew[i])

            candidates = [
                s for s in substrate_graph.nodes()
                if substrate_graph.nodes[s]['cpu'] >= v_cpu
                and s not in used
            ]

            if candidates:
                Xnew[i] = random.choice(candidates)

            used.add(Xnew[i])

    return Xnew



# =========================================================
# 4. Simulated Annealing Neighbor
# =========================================================
def sa_neighbor(particle, substrate_graph, vnr_graph):
    neighbor = particle.copy()
    used = set(neighbor)

    i = random.randrange(len(particle))
    v = list(vnr_graph.nodes())[i]
    v_cpu = vnr_graph.nodes[v]['cpu']

    used.discard(particle[i])

    candidates = [
        s for s in substrate_graph.nodes()
        if substrate_graph.nodes[s]['cpu'] >= v_cpu
        and s not in used
    ]

    if candidates:
        neighbor[i] = random.choice(candidates)

    return neighbor



# =========================================================
# 5. HPSO Main Algorithm (Optimized Speed)
# =========================================================

def hpso_embed(
    substrate_graph,
    vnr_graph,
    particles=20,
    iterations=30,
    w_max=0.9,
    w_min=0.5,
    beta=0.3,
    gamma=0.3,
    T0=100,
    cooling_rate=0.95
):
    vnodes = list(vnr_graph.nodes())
    num_v = len(vnodes)

    # 1. Init Swarm
    swarm = init_particles_hpso(substrate_graph, vnr_graph, particles)
    if not swarm:
        return None

    velocities = [
        [random.randint(0, 1) for _ in range(num_v)]
        for _ in range(len(swarm))
    ]

    pbest = copy.deepcopy(swarm)
    # Dùng fast_fitness để init cost
    pbest_cost = [fast_fitness(p, substrate_graph, vnr_graph) for p in swarm]

    gbest = None
    gbest_cost = INFEASIBLE_PENALTY
    
    # Tìm gbest ban đầu
    for i in range(len(swarm)):
        if pbest_cost[i] < gbest_cost:
            gbest_cost = pbest_cost[i]
            gbest = swarm[i].copy()

    T = T0

    # ================= MAIN LOOP =================
    for it in range(iterations):
        alpha = w_max - (w_max - w_min) * it / iterations
        total = alpha + beta + gamma
        # Normalize weights
        if total == 0: a, b, c = 0.33, 0.33, 0.33
        else: a, b, c = alpha / total, beta / total, gamma / total

        for i in range(len(swarm)):
            # --- 1. PSO UPDATE ---
            dp = operation_minus(pbest[i], swarm[i])
            dg = operation_minus(gbest, swarm[i]) if gbest else [0]*num_v

            tmp = operation_plus(a, velocities[i], b, dp)
            velocities[i] = operation_plus(1 - c, tmp, c, dg)

            new_pos = operation_multiply(
                swarm[i], velocities[i], vnr_graph, substrate_graph
            )
            
            # Dùng Fast Fitness để đánh giá PSO
            new_cost = fast_fitness(new_pos, substrate_graph, vnr_graph)
            
            # Update pBest/gBest
            if new_cost < pbest_cost[i]:
                pbest[i] = new_pos.copy()
                pbest_cost[i] = new_cost
                if new_cost < gbest_cost:
                    gbest = new_pos.copy()
                    gbest_cost = new_cost
            
            swarm[i] = new_pos

            # --- 2. SA STEP (Simulated Annealing) ---
            # Chỉ thực hiện SA nếu nhiệt độ còn đủ lớn để tránh lãng phí tính toán
            if T > 0.1:
                sa_cand = sa_neighbor(swarm[i], substrate_graph, vnr_graph)
                # Dùng Fast Fitness cho SA
                sa_cost = fast_fitness(sa_cand, substrate_graph, vnr_graph)
                
                current_cost = new_cost 
                delta_C = sa_cost - current_cost

                accept = False
                if delta_C < 0:
                    accept = True
                else:
                    r = random.random()
                    try:
                        prob = math.exp(-delta_C / T)
                    except OverflowError:
                        prob = 0
                    if r < prob:
                        accept = True
                
                if accept:
                    swarm[i] = sa_cand
                    # Update Pbest/Gbest nếu SA tìm được cái tốt hơn
                    if sa_cost < pbest_cost[i]:
                        pbest[i] = sa_cand.copy()
                        pbest_cost[i] = sa_cost
                        if sa_cost < gbest_cost:
                            gbest = sa_cand.copy()
                            gbest_cost = sa_cost

        # Giảm nhiệt độ
        T *= cooling_rate

    # ----- validate final gbest after main loop -----
    if gbest is not None:
        mapping, link_paths = build_and_reserve(gbest, substrate_graph, vnr_graph)
        if mapping is not None:
            return mapping, link_paths
         
    return None