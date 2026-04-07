import random
import math
from copy import deepcopy
from algorithms.pso import build_and_reserve

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

def hpso_embed(
    substrate_graph,
    vnr_graph,
    particles=20,
    iterations=100,
    beta=1.5,     # c1
    gamma=1.5,    # c2
    w_max=0.7,
    w_min=0.7,
    cooling_rate=0.9
):
    """
    Discrete HPSO = PSO + Simulated Annealing
    (theo Algorithm 1 – VNE-HPSO)
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
    pbest_cost = [fast_fitness(p, substrate_graph, vnr_graph) for p in swarm]

    gbest_idx = pbest_cost.index(min(pbest_cost))
    gbest = pbest[gbest_idx].copy()
    gbest_cost = pbest_cost[gbest_idx]

    # ---------------- SA temperature ----------------
    f_max = max(pbest_cost)
    f_min = min(pbest_cost)
    t = max(f_max - f_min, 1e-9)

    # ---------------- HPSO loop ----------------
    for it in range(iterations):

        w = w_max - (w_max - w_min) * it / iterations

        for i in range(particles):

            # ===== EARLY STOP =====
            mapping, link_paths = build_and_reserve(
                swarm[i], substrate_graph, vnr_graph
            )
            if mapping is not None:
                return mapping, link_paths
            # ======================

            old_particle = swarm[i].copy()
            old_cost = fast_fitness(old_particle, substrate_graph, vnr_graph)

            # -------- PSO UPDATE --------
            for d in range(num_v):
                r1 = random.random()
                r2 = random.random()

                candidates = candidate_lists[vnodes[d]]

                cur = swarm[i][d]
                cur_idx = candidates.index(cur)
                p_idx = candidates.index(pbest[i][d])
                g_idx = candidates.index(gbest[d])

                cognitive = beta * r1 * (p_idx - cur_idx)
                social = gamma * r2 * (g_idx - cur_idx)

                vel = w * velocities[i][d] + cognitive + social
                vel = max(-V_MAX, min(V_MAX, vel))
                velocities[i][d] = vel

                new_idx = cur_idx + int(round(vel))
                new_idx = max(0, min(len(candidates) - 1, new_idx))
                swarm[i][d] = candidates[new_idx]

            new_cost = fast_fitness(swarm[i], substrate_graph, vnr_graph)
            delta_c = new_cost - old_cost

            # -------- SA ACCEPTANCE --------
            accept = False
            if delta_c < 0:
                accept = True
            else:
                r = random.random()
                if r < min(1.0, math.exp(-delta_c / t)):
                    accept = True

            if not accept:
                swarm[i] = old_particle
                new_cost = old_cost

            # -------- update pBest --------
            if new_cost < pbest_cost[i]:
                pbest_cost[i] = new_cost
                pbest[i] = swarm[i].copy()

            # -------- update gBest --------
            if new_cost < gbest_cost:
                gbest_cost = new_cost
                gbest = swarm[i].copy()

        # -------- cooling --------
        t *= cooling_rate

    # ---------------- final attempt ----------------
    mapping, link_paths = build_and_reserve(
        gbest, substrate_graph, vnr_graph
    )
    if mapping is not None:
        return mapping, link_paths

    return None
