
# #v2
# # src/mapping_vne_pso.py
# import random
# import numpy as np
# from utils.graph_utils import shortest_path_with_capacity, reserve_node, reserve_path
# import heapq

# INFEASIBLE_PENALTY = 1e9
# STALL_LIMIT = 5
# TOP_K_FALLBACK = 5

# def fast_fitness(particle, substrate_graph, vnr_graph):
#     """
#     Fast fitness:
#     - check CPU feasibility
#     - estimate link cost by hop proxy
#     """
#     mapping = {}
#     vnodes = list(vnr_graph.nodes())

#     for i, v in enumerate(vnodes):
#         s = particle[i]
#         if substrate_graph.nodes[s]['cpu'] < vnr_graph.nodes[v]['cpu']:
#             return INFEASIBLE_PENALTY
#         mapping[v] = s

#     est_link_cost = 0
#     for (u, v) in vnr_graph.edges():
#         if mapping[u] != mapping[v]:
#             est_link_cost += vnr_graph.edges[u, v]['bw']

#     return est_link_cost

# def create_candidate_lists(substrate_graph, vnr_graph):
#     candidate_lists = {}
#     sub_nodes = list(substrate_graph.nodes())

#     for v in vnr_graph.nodes():
#         v_cpu = vnr_graph.nodes[v]['cpu']
#         v_domain = vnr_graph.nodes[v].get('domain')

#         candidates = []
#         for s in sub_nodes:
#             if substrate_graph.nodes[s]['cpu'] < v_cpu:
#                 continue
#             s_domain = substrate_graph.nodes[s].get('domain')
#             if v_domain is not None and s_domain is not None and v_domain != s_domain:
#                 continue
#             candidates.append(s)

#         if not candidates:
#             return None

#         candidate_lists[v] = candidates

#     return candidate_lists


# def build_full_solution(particle, substrate_graph, vnr_graph, path_cache):
#     mapping = {}
#     vnodes = list(vnr_graph.nodes())

#     for i, v in enumerate(vnodes):
#         s = particle[i]
#         if substrate_graph.nodes[s]['cpu'] < vnr_graph.nodes[v]['cpu']:
#             return None, None
#         mapping[v] = s

#     link_paths = {}
#     for (u, v) in vnr_graph.edges():
#         bw = vnr_graph.edges[u, v]['bw']
#         key = (mapping[u], mapping[v], bw)

#         if key in path_cache:
#             path = path_cache[key]
#         else:
#             path = shortest_path_with_capacity(
#                 substrate_graph, mapping[u], mapping[v], bw
#             )
#             path_cache[key] = path

#         if path is None:
#             return None, None

#         link_paths[(u, v)] = path

#     return mapping, link_paths

# def operation_minus(Xi, Xj):
#     return [1 if Xi[k] == Xj[k] else 0 for k in range(len(Xi))]


# def operation_plus(p, Vi, q, Vj):
#     res = []
#     for i in range(len(Vi)):
#         if Vi[i] == Vj[i]:
#             res.append(Vi[i])
#         else:
#             res.append(Vi[i] if random.random() < p else Vj[i])
#     return res


# def operation_multiply(Xi, V, vnr_graph, candidate_lists):
#     Xnew = Xi.copy()
#     vnodes = list(vnr_graph.nodes())

#     for i in range(len(V)):
#         if V[i] == 0:
#             v = vnodes[i]
#             candidates = candidate_lists[v]
#             alt = [s for s in candidates if s != Xi[i]]
#             if alt:
#                 Xnew[i] = random.choice(alt)
#     return Xnew

# def build_and_reserve(particle, substrate_graph, vnr_graph):
#     mapping = {}
#     vnodes = list(vnr_graph.nodes())

#     # node mapping
#     for i, v in enumerate(vnodes):
#         s = particle[i]
#         if substrate_graph.nodes[s]['cpu'] < vnr_graph.nodes[v]['cpu']:
#             return None, None
#         mapping[v] = s

#     link_paths = {}
#     for (u, v) in vnr_graph.edges():
#         bw = vnr_graph.edges[u, v]['bw']
#         path = shortest_path_with_capacity(
#             substrate_graph, mapping[u], mapping[v], bw
#         )
#         if path is None:
#             return None, None
#         link_paths[(u, v)] = path

#     # reserve
#     for v, s in mapping.items():
#         reserve_node(substrate_graph, s, vnr_graph.nodes[v]['cpu'])
#     for (u, v), path in link_paths.items():
#         reserve_path(substrate_graph, path, vnr_graph.edges[u, v]['bw'])

#     return mapping, link_paths

# def pso_embed(
#     substrate_graph,
#     vnr_graph,
#     particles=30,
#     iterations=50,
#     beta=0.3,
#     gamma=0.4,
#     w_max=0.9,
#     w_min=0.4
# ):
#     vnodes = list(vnr_graph.nodes())
#     num_v = len(vnodes)

#     candidate_lists = create_candidate_lists(substrate_graph, vnr_graph)
#     if candidate_lists is None:
#         return None

#     # ----- init swarm -----
#     swarm = []
#     velocities = []
#     for _ in range(particles):
#         swarm.append([random.choice(candidate_lists[v]) for v in vnodes])
#         velocities.append([random.randint(0, 1) for _ in range(num_v)])

#     pbest = [p.copy() for p in swarm]
#     pbest_cost = [INFEASIBLE_PENALTY] * particles
#     gbest = None
#     gbest_cost = INFEASIBLE_PENALTY

#     # ----- PSO loop -----
#     for it in range(iterations):
#         alpha = w_max - (w_max - w_min) * it / iterations
#         total = alpha + beta + gamma
#         a, b, c = alpha / total, beta / total, gamma / total

#         for i in range(particles):
#             # ========== EARLY STOP CHECK ==========
#             mapping, link_paths = build_and_reserve(
#                 swarm[i], substrate_graph, vnr_graph
#             )
#             if mapping is not None:
#                 return mapping, link_paths
#             # =====================================

#             cost = fast_fitness(swarm[i], substrate_graph, vnr_graph)

#             if cost < pbest_cost[i]:
#                 pbest_cost[i] = cost
#                 pbest[i] = swarm[i].copy()

#             if cost < gbest_cost:
#                 gbest_cost = cost
#                 gbest = swarm[i].copy()

#         # ----- velocity + position update -----
#         for i in range(particles):
#             dp = operation_minus(pbest[i], swarm[i])
#             dg = operation_minus(gbest, swarm[i])

#             tmp = operation_plus(a, velocities[i], b, dp)
#             velocities[i] = operation_plus(1 - c, tmp, c, dg)

#             swarm[i] = operation_multiply(
#                 swarm[i], velocities[i], vnr_graph, candidate_lists
#             )

#     # nếu chạy hết mà không embed được
#     return None


# v5
# src/algorithms/pso.py
import random
import numpy as np
import networkx as nx
from utils.graph_utils import shortest_path_with_capacity, reserve_node, reserve_path
import heapq

INFEASIBLE_PENALTY = 1e9

def fast_fitness(particle, substrate_graph, vnr_graph):
    """
    Fast fitness:
    - check injective constraint (no duplicate substrate nodes)
    - check CPU feasibility
    - estimate link cost by hop distance × bandwidth
    """
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


def create_candidate_lists(substrate_graph, vnr_graph):
    """
    Tạo danh sách các node thực có thể ánh xạ cho mỗi node ảo
    Kiểm tra CPU và domain constraints
    """
    candidate_lists = {}
    sub_nodes = list(substrate_graph.nodes())

    for v in vnr_graph.nodes():
        v_cpu = vnr_graph.nodes[v]['cpu']
        v_domain = vnr_graph.nodes[v].get('domain')

        candidates = []
        for s in sub_nodes:
            if substrate_graph.nodes[s]['cpu'] < v_cpu:
                continue
            s_domain = substrate_graph.nodes[s].get('domain')
            if v_domain is not None and s_domain is not None and v_domain != s_domain:
                continue
            candidates.append(s)

        if not candidates:
            return None

        candidate_lists[v] = candidates

    return candidate_lists


def initialize_particle(vnr_graph, candidate_lists, max_attempts=100):
    """
    Khởi tạo một particle ngẫu nhiên với injective constraint
    """
    vnodes = list(vnr_graph.nodes())
    
    for attempt in range(max_attempts):
        particle = []
        used_nodes = set()
        
        success = True
        for v in vnodes:
            # Chọn node thực chưa được dùng từ candidate list
            available = [s for s in candidate_lists[v] if s not in used_nodes]
            
            if not available:
                success = False
                break
            
            s = random.choice(available)
            particle.append(s)
            used_nodes.add(s)
        
        if success:
            return particle
    
    # Nếu không tìm được particle hợp lệ sau max_attempts
    return None


def build_full_solution(particle, substrate_graph, vnr_graph, path_cache):
    mapping = {}
    vnodes = list(vnr_graph.nodes())

    # === Injective check ===
    if len(set(particle)) < len(particle):
        return None, None

    for i, v in enumerate(vnodes):
        s = particle[i]
        if substrate_graph.nodes[s]['cpu'] < vnr_graph.nodes[v]['cpu']:
            return None, None
        mapping[v] = s

    link_paths = {}
    for (u, v) in vnr_graph.edges():
        bw = vnr_graph.edges[u, v]['bw']
        key = (mapping[u], mapping[v], bw)

        if key in path_cache:
            path = path_cache[key]
        else:
            path = shortest_path_with_capacity(
                substrate_graph, mapping[u], mapping[v], bw
            )
            path_cache[key] = path

        if path is None:
            return None, None

        link_paths[(u, v)] = path

    return mapping, link_paths


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


def operation_multiply(Xi, V, vnr_graph, candidate_lists):
    """
    Mutation operator with injective constraint enforcement
    """
    Xnew = Xi.copy()
    vnodes = list(vnr_graph.nodes())
    
    for i in range(len(V)):
        if V[i] == 0:
            v = vnodes[i]
            # Chọn node thực từ candidate list mà chưa được dùng
            used_nodes = set(Xnew)
            available = [s for s in candidate_lists[v] if s not in used_nodes or s == Xi[i]]
            
            if available:
                Xnew[i] = random.choice(available)
    
    return Xnew


def build_and_reserve(particle, substrate_graph, vnr_graph):
    mapping = {}
    vnodes = list(vnr_graph.nodes())

    # === Injective check ===
    if len(set(particle)) < len(particle):
        return None, None

    # node mapping
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

    # reserve
    for v, s in mapping.items():
        reserve_node(substrate_graph, s, vnr_graph.nodes[v]['cpu'])
    for (u, v), path in link_paths.items():
        reserve_path(substrate_graph, path, vnr_graph.edges[u, v]['bw'])

    return mapping, link_paths


def pso_embed(
    substrate_graph,
    vnr_graph,
    particles=20,
    iterations=30,
    beta=0.3,
    gamma=0.4,
    w=0.7
):
    vnodes = list(vnr_graph.nodes())
    num_v = len(vnodes)

    # ----- Tạo candidate lists (kiểm tra CPU và domain) -----
    candidate_lists = create_candidate_lists(substrate_graph, vnr_graph)
    if candidate_lists is None:
        return None

    # ----- Khởi tạo swarm ngẫu nhiên với injective constraint -----
    swarm = []
    velocities = []
    
    for _ in range(particles):
        particle = initialize_particle(vnr_graph, candidate_lists)
        
        if particle is None:
            # Không thể tạo particle hợp lệ
            return None
        
        swarm.append(particle)
        velocities.append([random.randint(0, 1) for _ in range(num_v)])

    pbest = [p.copy() for p in swarm]
    pbest_cost = [INFEASIBLE_PENALTY] * particles
    gbest = None
    gbest_cost = INFEASIBLE_PENALTY

    path_cache = {}

    # ----- PSO loop -----
    for it in range(iterations):
        total = w + beta + gamma
        a, b, c = w / total, beta / total, gamma / total

        for i in range(particles):
            cost = fast_fitness(swarm[i], substrate_graph, vnr_graph)

            if cost < pbest_cost[i]:
                pbest_cost[i] = cost
                pbest[i] = swarm[i].copy()

            if cost < gbest_cost:
                gbest_cost = cost
                gbest = swarm[i].copy()

        # ----- velocity + position update -----
        for i in range(particles):
            dp = operation_minus(pbest[i], swarm[i])
            dg = operation_minus(gbest, swarm[i])

            tmp = operation_plus(a, velocities[i], b, dp)
            velocities[i] = operation_plus(1 - c, tmp, c, dg)

            swarm[i] = operation_multiply(
                swarm[i], velocities[i], vnr_graph, candidate_lists
            )

    # ----- validate final gbest -----
    if gbest is not None:
        mapping, link_paths = build_full_solution(
            gbest, substrate_graph, vnr_graph, path_cache
        )
        if mapping is not None:
            # reserve resources
            for v, s in mapping.items():
                reserve_node(substrate_graph, s, vnr_graph.nodes[v]['cpu'])
            for (u, v), path in link_paths.items():
                reserve_path(substrate_graph, path, vnr_graph.edges[u, v]['bw'])
            return mapping, link_paths

    # if no feasible solution found
    return None