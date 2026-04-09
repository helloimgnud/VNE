import random
import math
from copy import deepcopy
from src.algorithms.pso import build_and_reserve

INFEASIBLE_PENALTY = 1e9
V_MAX = 4

def fast_fitness(particle, substrate_graph, vnr_graph):
    """
    Fitness tương đương fitness_particle của VNE_PSO
    - CPU aggregation
    - CPU violation penalty
    - Node cost proxy (CPU usage)
    """
    usage = {}
    cost = 0
    vnodes = list(vnr_graph.nodes())

    # ---- aggregate CPU usage ----
    for i, v in enumerate(vnodes):
        s = particle[i]
        demand = vnr_graph.nodes[v]['cpu']
        usage[s] = usage.get(s, 0) + demand

    # ---- CPU violation penalty ----
    for s, used in usage.items():
        if used > substrate_graph.nodes[s]['cpu']:
            cost += 1000

    # ---- node cost (proxy) ----
    for i, v in enumerate(vnodes):
        s = particle[i]
        cost += vnr_graph.nodes[v]['cpu']

    return cost


def create_candidate_lists(substrate_graph, vnr_graph):
    """
    Candidate selection tương đương initialize_particles của VNE_PSO
    nhưng chỉ dùng API NetworkX
    """
    candidate_lists = {}
    sub_nodes = list(substrate_graph.nodes())

    for v in vnr_graph.nodes():
        v_cpu = vnr_graph.nodes[v]['cpu']
        v_domain = vnr_graph.nodes[v].get('domain')

        candidates = []
        for s in sub_nodes:
            # CPU feasibility
            if substrate_graph.nodes[s]['cpu'] < v_cpu:
                continue

            # Domain constraint (nếu tồn tại)
            s_domain = substrate_graph.nodes[s].get('domain')
            if v_domain is not None and s_domain is not None:
                if v_domain != s_domain:
                    continue

            candidates.append(s)

        if not candidates:
            return None

        candidate_lists[v] = candidates

    return candidate_lists



def pso_embed(
    substrate_graph,
    vnr_graph,
    particles=20,
    iterations=100,
    beta=1.5,     # c1
    gamma=1.5,    # c2
    w_max=0.7,
    w_min=0.7
):
    """
    PSO node-mapping + Dijkstra link-mapping
    LOGIC TƯƠNG ĐƯƠNG VNE_PSO
    """

    vnodes = list(vnr_graph.nodes())
    num_v = len(vnodes)

    # ---------------- candidate lists ----------------
    candidate_lists = create_candidate_lists(substrate_graph, vnr_graph)
    if candidate_lists is None:
        return None

    # ---------------- init swarm ----------------
    swarm = []
    velocities = []

    for _ in range(particles):
        particle = []
        vel = []
        for v in vnodes:
            particle.append(random.choice(candidate_lists[v]))
            vel.append(0.0)
        swarm.append(particle)
        velocities.append(vel)

    pbest = [p.copy() for p in swarm]
    pbest_cost = [INFEASIBLE_PENALTY] * particles
    gbest = None
    gbest_cost = INFEASIBLE_PENALTY

    # ---------------- PSO loop ----------------
    for it in range(iterations):

        # inertia weight (giữ nguyên nếu w_max == w_min)
        w = w_max - (w_max - w_min) * it / iterations

        for i in range(particles):

            # ===== EARLY STOP (GIỐNG handle_mapping_request) =====
            mapping, link_paths = build_and_reserve(
                swarm[i], substrate_graph, vnr_graph
            )
            if mapping is not None:
                return mapping, link_paths
            # =====================================================

            cost = fast_fitness(swarm[i], substrate_graph, vnr_graph)

            if cost < pbest_cost[i]:
                pbest_cost[i] = cost
                pbest[i] = swarm[i].copy()

            if cost < gbest_cost:
                gbest_cost = cost
                gbest = swarm[i].copy()

        # ---------------- velocity + position update ----------------
        for i in range(particles):
            for d in range(num_v):
                r1 = random.random()
                r2 = random.random()

                candidates = candidate_lists[vnodes[d]]

                cur = swarm[i][d]
                cur_idx = candidates.index(cur)

                p_idx = candidates.index(pbest[i][d])
                g_idx = candidates.index(gbest[d])

                # === GIỐNG update_particles() ===
                cognitive = beta * r1 * (p_idx - cur_idx)
                social = gamma * r2 * (g_idx - cur_idx)

                vel = w * velocities[i][d] + cognitive + social
                vel = max(-V_MAX, min(V_MAX, vel))
                velocities[i][d] = vel

                new_idx = cur_idx + int(round(vel))
                new_idx = max(0, min(len(candidates) - 1, new_idx))

                swarm[i][d] = candidates[new_idx]

    # nếu chạy hết mà không embed được
    return None
