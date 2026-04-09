# src/algorithms/hpso.py
import random
import math
import copy
from src.utils.graph_utils import shortest_path_with_capacity, reserve_node, reserve_path

INFEASIBLE_PENALTY = 1e9

def build_embedding(particle, substrate_graph, vnr_graph):
    mapping = {}
    vnodes = list(vnr_graph.nodes())

    # reserve nodes
    for i, v in enumerate(vnodes):
        s = particle[i]
        if substrate_graph.nodes[s]['cpu'] < vnr_graph.nodes[v]['cpu']:
            return None, None
        mapping[v] = s

    link_paths = {}
    for (u, v) in vnr_graph.edges():
        bw = vnr_graph.edges[u, v]['bw']
        path = shortest_path_with_capacity(
            substrate_graph, mapping[u], mapping[v], bw
        )
        if path is None:
            return None, None
        link_paths[(u, v)] = path

    # reserve resources
    for v, s in mapping.items():
        reserve_node(substrate_graph, s, vnr_graph.nodes[v]['cpu'])
    for (u, v), path in link_paths.items():
        reserve_path(substrate_graph, path, vnr_graph.edges[u, v]['bw'])

    return mapping, link_paths


# =========================================================
# 1. Particle Initialization Allocation Strategy (Paper IV-D)
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
        used = set()

        for v in vnodes:
            v_cpu = vnr_graph.nodes[v]['cpu']
            candidates = [
                s for s in sub_nodes_sorted
                if substrate_graph.nodes[s]['cpu'] >= v_cpu and s not in used
            ]
            if not candidates:
                mapping = None
                break

            k = random.randint(1, min(3, len(candidates)))
            s = random.choice(candidates[:k])
            mapping[v] = s
            used.add(s)

        if mapping is not None:
            swarm.append([mapping[v] for v in vnr_graph.nodes()])

    return swarm


# =========================================================
# 2. Full Fitness Function (Node + Link Embedding)
# =========================================================
def hpso_fitness(particle, substrate_graph, vnr_graph):
    mapping = {}
    vnodes = list(vnr_graph.nodes())

    # --- Node feasibility ---
    for i, v in enumerate(vnodes):
        s = particle[i]
        if substrate_graph.nodes[s]['cpu'] < vnr_graph.nodes[v]['cpu']:
            return INFEASIBLE_PENALTY
        mapping[v] = s

    cost = 0

    # --- Link embedding ---
    for (u, v) in vnr_graph.edges():
        bw = vnr_graph.edges[u, v]['bw']
        path = shortest_path_with_capacity(
            substrate_graph, mapping[u], mapping[v], bw
        )
        if path is None:
            return INFEASIBLE_PENALTY
        cost += (len(path) - 1) * bw

    return cost


# =========================================================
# 3. PSO Operators (Paper Eq. 14–16)
# =========================================================
def operation_minus(Xi, Xj):
    return [1 if Xi[k] == Xj[k] else 0 for k in range(len(Xi))]


def operation_plus(p, Vi, q, Vj):
    res = []
    for i in range(len(Vi)):
        if Vi[i] == Vj[i]:
            res.append(Vi[i])
        else:
            res.append(Vi[i] if random.random() < p else Vj[i])
    return res


def operation_multiply(Xi, V, vnr_graph, substrate_graph):
    Xnew = Xi.copy()
    vnodes = list(vnr_graph.nodes())

    for i in range(len(V)):
        if V[i] == 0:
            v = vnodes[i]
            v_cpu = vnr_graph.nodes[v]['cpu']
            candidates = [
                s for s in substrate_graph.nodes()
                if substrate_graph.nodes[s]['cpu'] >= v_cpu
            ]
            if candidates:
                Xnew[i] = random.choice(candidates)

    return Xnew


# =========================================================
# 4. Simulated Annealing Neighbor (Paper Algorithm 1)
# =========================================================
def velocity_update(v, x, pbest, gbest, w, c1, c2):
    inertia = operation_multiply(w, v)
    cognitive = operation_multiply(c1 * random.random(),
                                   operation_minus(pbest, x))
    social = operation_multiply(c2 * random.random(),
                                operation_minus(gbest, x))
    return operation_plus(inertia,
                           operation_plus(cognitive, social))

def sa_neighbor(particle, substrate_graph, vnr_graph):
    neighbor = particle.copy()
    i = random.randrange(len(particle))
    v = list(vnr_graph.nodes())[i]
    v_cpu = vnr_graph.nodes[v]['cpu']

    candidates = [
        s for s in substrate_graph.nodes()
        if substrate_graph.nodes[s]['cpu'] >= v_cpu and s != particle[i]
    ]
    if candidates:
        neighbor[i] = random.choice(candidates)

    return neighbor


# =========================================================
# 5. HPSO Main Algorithm (Algorithm 1 – Paper)
# =========================================================

def hpso_embed(
    substrate_graph,
    vnr_graph,
    particles=20,
    iterations=30,
    w_max=0.9,
    w_min=0.4,
    beta=0.3,
    gamma=0.3,
    T0=100,
    cooling_rate=0.95
):
    vnodes = list(vnr_graph.nodes())
    num_v = len(vnodes)

    swarm = init_particles_hpso(
        substrate_graph, vnr_graph, particles
    )
    if not swarm:
        return None

    velocities = [
        [random.randint(0, 1) for _ in range(num_v)]
        for _ in range(len(swarm))
    ]

    pbest = copy.deepcopy(swarm)
    pbest_cost = [
        hpso_fitness(p, substrate_graph, vnr_graph) for p in swarm
    ]

    gbest = None
    gbest_cost = INFEASIBLE_PENALTY
    
    # Tìm gbest ban đầu từ swarm khởi tạo
    for i in range(len(swarm)):
        cost = hpso_fitness(swarm[i], substrate_graph, vnr_graph)
        pbest_cost[i] = cost
        if cost < gbest_cost:
            gbest_cost = cost
            gbest = swarm[i].copy()

    T = T0

    # ================= MAIN LOOP =================
    for it in range(iterations):
        alpha = w_max - (w_max - w_min) * it / iterations
        total = alpha + beta + gamma
        a, b, c = alpha / total, beta / total, gamma / total

        for i in range(len(swarm)):
            # Dùng gbest thay vì pbest[i] cho thành phần thứ 3
            dp = operation_minus(pbest[i], swarm[i])
            dg = operation_minus(gbest, swarm[i]) 

            tmp = operation_plus(a, velocities[i], b, dp)
            velocities[i] = operation_plus(1 - c, tmp, c, dg)

            # Tạo vị trí mới từ PSO
            new_pos = operation_multiply(
                swarm[i], velocities[i], vnr_graph, substrate_graph
            )
            
            # Tính fitness cho vị trí PSO mới
            new_cost = hpso_fitness(new_pos, substrate_graph, vnr_graph)
            
            # Cập nhật pBest/gBest ngay (theo flow chuẩn PSO)
            if new_cost < pbest_cost[i]:
                pbest[i] = new_pos.copy()
                pbest_cost[i] = new_cost
                if new_cost < gbest_cost:
                    gbest = new_pos.copy()
                    gbest_cost = new_cost
            
            # Cập nhật vị trí hiện tại
            swarm[i] = new_pos

            # 2. SA STEP (Theo Algorithm 1: Sau khi PSO, thử mutation)
            # Tạo một neighbor ngẫu nhiên từ vị trí hiện tại
            sa_cand = sa_neighbor(swarm[i], substrate_graph, vnr_graph)
            sa_cost = hpso_fitness(sa_cand, substrate_graph, vnr_graph)
            
            current_cost = new_cost # Cost hiện tại sau bước PSO
            delta_C = sa_cost - current_cost

            # Logic chấp nhận của SA
            if delta_C < 0:
                swarm[i] = sa_cand
                # Cần update lại pBest/gBest nếu bước SA tìm ra cái tốt hơn
                if sa_cost < pbest_cost[i]:
                    pbest[i] = sa_cand.copy()
                    pbest_cost[i] = sa_cost
                    if sa_cost < gbest_cost:
                        gbest = sa_cand.copy()
                        gbest_cost = sa_cost
            else:
                # Chấp nhận với xác suất
                r = random.random()
                if r < min(1, math.exp(-delta_C / T)):
                     swarm[i] = sa_cand
            
        # Giảm nhiệt độ
        T *= cooling_rate

    # Trả về kết quả tốt nhất tìm thấy (gbest)
    # Lưu ý: Hàm gốc trả về mapping, cần build lại từ gbest
    if gbest is not None and gbest_cost < INFEASIBLE_PENALTY:
         return build_embedding(gbest, substrate_graph, vnr_graph)
         
    return None