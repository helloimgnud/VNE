# src/algorithms/hpso.py
import random
import math
import copy
from utils.graph_utils import shortest_path_with_capacity, reserve_node, reserve_path

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

    T = T0

    # ================= MAIN LOOP =================
    for it in range(iterations):
        alpha = w_max - (w_max - w_min) * it / iterations
        total = alpha + beta + gamma
        a, b, c = alpha / total, beta / total, gamma / total

        for i in range(len(swarm)):
            # ---------- EARLY STOP CHECK ----------
            cost = hpso_fitness(swarm[i], substrate_graph, vnr_graph)
            if cost < INFEASIBLE_PENALTY:
                mapping, link_paths = build_embedding(
                    swarm[i], substrate_graph, vnr_graph
                )
                if mapping is not None:
                    return mapping, link_paths

            # ---------- PSO UPDATE ----------
            dp = operation_minus(pbest[i], swarm[i])
            dg = operation_minus(pbest[i], swarm[i])

            tmp = operation_plus(a, velocities[i], b, dp)
            velocities[i] = operation_plus(1 - c, tmp, c, dg)

            pso_candidate = operation_multiply(
                swarm[i], velocities[i], vnr_graph, substrate_graph
            )

            sa_candidate = sa_neighbor(
                swarm[i], substrate_graph, vnr_graph
            )

            for cand in [pso_candidate, sa_candidate]:
                old_cost = hpso_fitness(
                    swarm[i], substrate_graph, vnr_graph
                )
                new_cost = hpso_fitness(
                    cand, substrate_graph, vnr_graph
                )

                if new_cost < old_cost or random.random() < math.exp(-(new_cost - old_cost) / T):
                    swarm[i] = cand

            # ---------- UPDATE PBEST ----------
            cost = hpso_fitness(swarm[i], substrate_graph, vnr_graph)
            if cost < pbest_cost[i]:
                pbest[i] = swarm[i].copy()
                pbest_cost[i] = cost

        T *= cooling_rate

    # nếu chạy hết mà không embed được
    return None
