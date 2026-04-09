# proposed_v2.py  –  drop-in replacement for proposed.py (batch embed entry)
#
# Key improvements over v1:
#   1. Cost-aware fitness  (uses node.cost + edge.bw_cost from eval.py)
#   2. Fixed gbest direction  (min-cost, not max)
#   3. Inertia-decay PSO with correct w_max→w_min schedule
#   4. Repair runs on a *snapshot* of substrate BEFORE reservation so it
#      never over-commits resources
#   5. reserve_with_topk: correct score (cost/revenue, lower → better),
#      sorted ascending so cheapest-relative-to-revenue goes first
#   6. Parallel PSO per VNR (unchanged), but gbest tracking is now correct
#   7. Two-stage repair now also checks snapshot, then reserves on the live
#      substrate only once we are sure the solution is feasible

import random
import copy
import math
from multiprocessing import Pool, cpu_count

import networkx as nx

from src.evaluation.eval import revenue_of_vnr
from src.algorithms.fast_hpso import (
    operation_minus,
    operation_plus,
    operation_multiply,
    sa_neighbor,
    init_particles_hpso,
)
from src.algorithms.proposed import (
    Individual,
    build_full_solution_check,
    verify_and_allocate,
    INFEASIBLE_FITNESS,
)
from src.utils.graph_utils import (
    can_place_node,
    reserve_node,
    reserve_path,
    shortest_path_with_capacity,
)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
INFEASIBLE_PENALTY = 1e9


# ===========================================================================
# 1.  COST-AWARE FITNESS
#     Mirrors eval.py: node_cost = cpu * node['cost']
#                      link_cost = bw  * edge['bw_cost'] * hops (proxy)
# ===========================================================================

def cost_aware_fitness(particle, substrate, vnr):
    """
    Lower  → better (we are minimising cost).
    Returns INFEASIBLE_PENALTY for infeasible particles.
    """
    vnodes = list(vnr.nodes())

    # Injective check
    if len(set(particle)) < len(particle):
        return INFEASIBLE_PENALTY

    mapping = {}
    node_cost = 0.0

    for i, v in enumerate(vnodes):
        s = particle[i]
        v_cpu = vnr.nodes[v]['cpu']
        if substrate.nodes[s]['cpu'] < v_cpu:
            return INFEASIBLE_PENALTY
        p_cost = substrate.nodes[s].get('cost', 1.0)
        node_cost += v_cpu * p_cost
        mapping[v] = s

    link_cost = 0.0
    for (u, v) in vnr.edges():
        s_u, s_v = mapping[u], mapping[v]
        if s_u == s_v:
            continue
        try:
            path = nx.shortest_path(substrate, s_u, s_v)
        except nx.NetworkXNoPath:
            return INFEASIBLE_PENALTY

        bw_demand = vnr.edges[u, v]['bw']
        for i in range(len(path) - 1):
            bw_cost = substrate.edges[path[i], path[i + 1]].get('bw_cost', 1.0)
            link_cost += bw_demand * bw_cost

    return node_cost + link_cost


# ===========================================================================
# 2.  EVOLVE ONE VNR  (fixed gbest direction + inertia decay)
# ===========================================================================

def evolve_vnr_full(args):
    vnr, substrate, pop_size, generations, top_k, w_max, w_min = args

    particles = init_particles_hpso(substrate, vnr, pop_size)
    if not particles:
        return []

    population = []
    for p in particles:
        vel = [random.randint(0, 1) for _ in p]
        fit = cost_aware_fitness(p, substrate, vnr)          # lower = better
        population.append(Individual(p, vel, fit))

    pbests = copy.deepcopy(population)

    # gbest = particle with MINIMUM cost
    gbest = min(population, key=lambda x: x.fitness)

    for gen in range(generations):
        # Inertia weight decays linearly
        w = w_max - (w_max - w_min) * gen / max(generations - 1, 1)

        new_pop, new_pbest = [], []

        for i, ind in enumerate(population):
            dp = operation_minus(pbests[i].particle, ind.particle)
            dg = operation_minus(gbest.particle, ind.particle)

            # velocity update with inertia
            v = operation_plus(w,   ind.velocity, 0.0, ind.velocity)   # inertia
            v = operation_plus(1.0, v,            0.3, dp)             # cognitive
            v = operation_plus(1.0, v,            0.3, dg)             # social

            new_particle = operation_multiply(ind.particle, v, vnr, substrate)
            fit = cost_aware_fitness(new_particle, substrate, vnr)

            # SA refinement
            sa_p   = sa_neighbor(new_particle, substrate, vnr)
            sa_fit = cost_aware_fitness(sa_p, substrate, vnr)
            if sa_fit < fit:                                 # SA: accept if lower cost
                new_particle, fit = sa_p, sa_fit

            new_ind = Individual(new_particle, v, fit)

            # Update personal best (lower cost wins)
            if fit < pbests[i].fitness:
                new_pbest.append(copy.deepcopy(new_ind))
                if fit < gbest.fitness:
                    gbest = copy.deepcopy(new_ind)
            else:
                new_pbest.append(pbests[i])

            new_pop.append(new_ind)

        population = new_pop
        pbests     = new_pbest

    # TOP-K: feasible particles with LOWEST cost
    feasible = [p for p in population if p.fitness < INFEASIBLE_PENALTY]
    feasible.sort(key=lambda x: x.fitness)           # ascending = best first

    return feasible[:top_k]


# ===========================================================================
# 3.  PARALLEL PSO FOR ALL VNRS
# ===========================================================================

def solve_all_vnrs_parallel(
    vnr_list, substrate, pop_size, generations, top_k,
    w_max=0.9, w_min=0.4
):
    tasks = [
        (vnr, substrate, pop_size, generations, top_k, w_max, w_min)
        for vnr in vnr_list
    ]
    n_proc = min(cpu_count(), len(tasks))
    with Pool(processes=n_proc) as pool:
        results = pool.map(evolve_vnr_full, tasks)
    return results


# ===========================================================================
# 4.  RESERVATION (TOP-K TRY)  –  fixed score direction
# ===========================================================================

def reserve_with_topk(substrate, vnr_list, revenues, candidates_per_vnr,
                       ac_controller=None):
    substrate_reserved = copy.deepcopy(substrate)
    accepted, rejected = {}, set()

    scores = []
    for i, cands in enumerate(candidates_per_vnr):
        if not cands:
            rejected.add(i)
            continue

        if ac_controller is not None:
            # ---- RL-based admission control ----
            ac_score = ac_controller.score(substrate_reserved, vnr_list[i])
            if ac_score < 0.5:
                rejected.add(i)
                continue                       # AC says reject
            scores.append((-ac_score, i))      # higher AC score → first
        else:
            # ---- Fallback: heuristic cost/revenue ----
            best_cost = cands[0].fitness       # already sorted ascending
            score = best_cost / max(revenues[i], 1e-6)
            scores.append((score, i))

    scores.sort()                              # ascending

    for _, k in scores:
        vnr   = vnr_list[k]
        placed = False

        for ind in candidates_per_vnr[k]:
            sol = build_full_solution_check(vnr, ind.particle, substrate_reserved)
            if sol is None:
                continue

            # Node capacity check
            feasible = True
            for v_node, s_node in sol.mapping.items():
                if not can_place_node(substrate_reserved, s_node, vnr.nodes[v_node]['cpu']):
                    feasible = False
                    break

            if not feasible:
                continue

            # Link capacity check
            for (u, v), path in sol.link_paths.items():
                bw = vnr.edges[u, v]['bw']
                for i in range(len(path) - 1):
                    if substrate_reserved[path[i]][path[i + 1]]['bw'] < bw:
                        feasible = False
                        break
                if not feasible:
                    break

            if not feasible:
                continue

            # Reserve
            for v_node, s_node in sol.mapping.items():
                reserve_node(substrate_reserved, s_node, vnr.nodes[v_node]['cpu'])
            for (u, v), path in sol.link_paths.items():
                reserve_path(substrate_reserved, path, vnr.edges[u, v]['bw'])

            accepted[k] = sol
            placed = True
            break

        if not placed:
            rejected.add(k)

    return accepted, rejected


# ===========================================================================
# 5.  TWO-STAGE REPAIR  (operates on a snapshot, reserves on live substrate)
# ===========================================================================

def _detect_infeasible_nodes(mapping, substrate, vnr):
    return [
        v for v, s in mapping.items()
        if substrate.nodes[s]['cpu'] < vnr.nodes[v]['cpu']
    ]


def _detect_infeasible_links(mapping, substrate, vnr, link_paths):
    bad, reasons = [], {}
    for (u, v) in vnr.edges():
        bw  = vnr.edges[u, v]['bw']
        pth = link_paths.get((u, v))
        if pth is None or len(pth) < 2:
            bad.append((u, v)); reasons[(u, v)] = "no path"; continue
        for i in range(len(pth) - 1):
            if substrate[pth[i]][pth[i + 1]].get('bw', 0) < bw:
                bad.append((u, v))
                reasons[(u, v)] = f"BW @ ({pth[i]},{pth[i+1]})"
                break
    return bad, reasons


def _rebuild_paths(mapping, substrate, vnr):
    paths = {}
    for (u, v) in vnr.edges():
        try:
            p = nx.shortest_path(substrate, mapping[u], mapping[v])
        except nx.NetworkXNoPath:
            return None
        paths[(u, v)] = p
    return paths


def _repair_nodes(mapping, substrate, vnr, bad_nodes):
    new_map = copy.deepcopy(mapping)
    used    = set(new_map.values())
    for vn in bad_nodes:
        need_cpu = vnr.nodes[vn]['cpu']
        used.discard(new_map[vn])
        cands = sorted(
            [s for s in substrate.nodes()
             if s not in used and substrate.nodes[s]['cpu'] >= need_cpu],
            key=lambda s: substrate.nodes[s]['cpu']          # smallest-fit
        )
        if not cands:
            return None
        new_map[vn] = cands[0]
        used.add(cands[0])
    return new_map


def _repair_links(mapping, substrate, vnr, bad_links, link_paths):
    new_paths = dict(link_paths)
    for (u, v) in bad_links:
        bw  = vnr.edges[u, v]['bw']
        src, dst = mapping[u], mapping[v]
        # Build bandwidth-filtered graph
        flt = nx.DiGraph() if substrate.is_directed() else nx.Graph()
        flt.add_nodes_from(substrate.nodes(data=True))
        for a, b, d in substrate.edges(data=True):
            if d.get('bw', 0) >= bw:
                flt.add_edge(a, b, **d)
        try:
            p = nx.shortest_path(flt, src, dst)
        except nx.NetworkXNoPath:
            return None
        new_paths[(u, v)] = p
    return new_paths


def two_stage_repair(vnr, gbest_particle, substrate, vnr_idx=0, verbose=False):
    """
    Returns (mapping, link_paths, success).
    Works on a snapshot so the live substrate is never touched here.
    """
    snap    = copy.deepcopy(substrate)   # read-only snapshot for feasibility checks
    vnodes  = list(vnr.nodes())
    mapping = {v: gbest_particle[i] for i, v in enumerate(vnodes)}

    # ---- Stage 1: Node repair ----
    bad_nodes = _detect_infeasible_nodes(mapping, snap, vnr)
    if bad_nodes:
        mapping = _repair_nodes(mapping, snap, vnr, bad_nodes)
        if mapping is None:
            return None, None, False

    link_paths = _rebuild_paths(mapping, snap, vnr)
    if link_paths is None:
        return None, None, False

    # ---- Stage 2: Link repair ----
    bad_links, _ = _detect_infeasible_links(mapping, snap, vnr, link_paths)
    if bad_links:
        link_paths = _repair_links(mapping, snap, vnr, bad_links, link_paths)
        if link_paths is None:
            return None, None, False

    # ---- Final check ----
    if _detect_infeasible_nodes(mapping, snap, vnr):
        return None, None, False
    if _detect_infeasible_links(mapping, snap, vnr, link_paths)[0]:
        return None, None, False

    return mapping, link_paths, True


# ===========================================================================
# 6.  PUBLIC API  –  embed_batch
# ===========================================================================

def embed_batch(
    substrate,
    batch,
    pop_size   = 20,
    generations= 30,
    top_k      = 5,
    w_max      = 0.9,
    w_min      = 0.4,
    ac_model_path = None,
    verbose    = False,
):
    """
    Embed a batch of VNRs.

    Returns
    -------
    accepted : list of (vnr, mapping, link_paths)
    rejected : list of vnr
    """
    accepted, rejected = [], []

    vnr_list = [vnr for vnr, _ in batch]
    revenues = [revenue_of_vnr(vnr) for vnr in vnr_list]

    # ---- Phase 1: parallel PSO ----
    candidates = solve_all_vnrs_parallel(
        vnr_list, substrate, pop_size, generations, top_k, w_max, w_min
    )

    # ---- Phase 2: reservation with optional RL admission control ----
    ac = None
    if ac_model_path is not None:
        from src.algorithms.ac_controller import AdmissionController
        ac = AdmissionController(ac_model_path)
        if verbose:
            print('[AC] RL admission controller loaded')

    locked, rejected_idx = reserve_with_topk(
        substrate, vnr_list, revenues, candidates,
        ac_controller=ac
    )

    for i, vnr in enumerate(vnr_list):
        if i in locked:
            sol = locked[i]
            if verify_and_allocate(substrate, sol):
                accepted.append((sol.vnr, sol.mapping, sol.link_paths))
            else:
                rejected_idx.add(i)

    rejected_vnrs = [vnr_list[i] for i in rejected_idx]

    if not rejected_vnrs:
        return accepted, rejected_vnrs

    # ---- Phase 3: two-stage repair for remaining rejects ----
    if verbose:
        print(f"\n[REPAIR] {len(rejected_vnrs)} VNRs to repair ...")

    repair_accepted, final_rejected = [], []

    for vnr in rejected_vnrs:
        original_idx = vnr_list.index(vnr)
        cands = candidates[original_idx] if original_idx < len(candidates) else []

        if not cands:
            final_rejected.append(vnr)
            continue

        mapping, link_paths, ok = two_stage_repair(
            vnr, cands[0].particle, substrate, original_idx, verbose
        )

        if not ok:
            final_rejected.append(vnr)
            continue

        # Check live substrate capacity before committing
        can_reserve = True
        for vn, sn in mapping.items():
            if not can_place_node(substrate, sn, vnr.nodes[vn]['cpu']):
                can_reserve = False; break

        if can_reserve:
            for (u, v), path in link_paths.items():
                bw = vnr.edges[u, v]['bw']
                for i in range(len(path) - 1):
                    if substrate[path[i]][path[i + 1]]['bw'] < bw:
                        can_reserve = False; break
                if not can_reserve:
                    break

        if can_reserve:
            for vn, sn in mapping.items():
                reserve_node(substrate, sn, vnr.nodes[vn]['cpu'])
            for (u, v), path in link_paths.items():
                reserve_path(substrate, path, vnr.edges[u, v]['bw'])
            repair_accepted.append((vnr, mapping, link_paths))
            if verbose:
                print(f"  VNR {original_idx}: RECOVERED")
        else:
            final_rejected.append(vnr)
            if verbose:
                print(f"  VNR {original_idx}: repair ok but substrate exhausted")

    accepted.extend(repair_accepted)

    if verbose:
        print(f"[REPAIR] recovered {len(repair_accepted)}/{len(rejected_vnrs)}")

    return accepted, final_rejected