# src/algorithms/parallel_hpso_priority.py
"""
Parallel HPSO with Priority Vector Encoding + Two-Stage Repair
Combines:
- Priority vector encoding from hpso_priority.py
- Parallel PSO processing for multiple VNRs
- Two-stage repair mechanism (node repair + link repair)
"""

from evaluation.eval import revenue_of_vnr
import random
import copy
import numpy as np
import networkx as nx
from multiprocessing import Pool, cpu_count

from algorithms.hpso_priority import (
    build_domain_masks,
    build_available_masks,
    decode_priority_vector,
    fast_fitness_priority,
    init_particles_priority,
    update_velocity_priority,
    update_position_priority,
    sa_neighbor_priority,
    INFEASIBLE_PENALTY
)

from utils.graph_utils import (
    can_place_node,
    reserve_node,
    reserve_path,
    shortest_path_with_capacity
)

# ============================================================
# 1. DATA STRUCTURES
# ============================================================

class PriorityIndividual:
    """Individual with priority vector encoding"""
    def __init__(self, priority_vec, velocity, fitness):
        self.priority_vec = priority_vec
        self.velocity = velocity
        self.fitness = fitness

class Solution:
    """Solution wrapper"""
    def __init__(self, vnr, mapping, link_paths):
        self.vnr = vnr
        self.mapping = mapping
        self.link_paths = link_paths

# ============================================================
# 2. EVOLVE ONE VNR (PRIORITY VECTOR PSO)
# ============================================================

def evolve_vnr_priority(args):
    """
    Run PSO with priority vector encoding for one VNR.
    Returns top-K best solutions.
    
    Args:
        args: tuple (vnr, substrate, pop_size, generations, top_k, w_max, w_min, c1, c2, mutation_rate)
        
    Returns:
        list of (priority_vec, fitness) tuples - top K candidates
    """
    (vnr, substrate, pop_size, generations, top_k, 
     w_max, w_min, c1, c2, mutation_rate) = args
    
    # Pre-processing
    domain_masks, node_to_idx = build_domain_masks(substrate)
    available_masks = build_available_masks(substrate, vnr, node_to_idx)
    m = len(node_to_idx)
    
    # Initialize swarm
    swarm = init_particles_priority(m, pop_size)
    velocities = [np.random.uniform(-0.1, 0.1, m) for _ in range(pop_size)]
    
    # Evaluate initial fitness
    fitnesses = [
        fast_fitness_priority(p, substrate, vnr, domain_masks, available_masks, node_to_idx)
        for p in swarm
    ]
    
    # Initialize population
    population = []
    for i in range(pop_size):
        population.append(PriorityIndividual(swarm[i], velocities[i], fitnesses[i]))
    
    # Initialize pbest
    pbests = copy.deepcopy(population)
    
    # Find gbest
    gbest_idx = int(np.argmin([p.fitness for p in population]))
    gbest = population[gbest_idx].priority_vec.copy()
    gbest_cost = population[gbest_idx].fitness
    
    # ---- PSO Loop ----
    for it in range(generations):
        # Linearly decreasing inertia weight
        w = w_max - (w_max - w_min) * it / generations
        
        for i in range(pop_size):
            # --- 1. PSO UPDATE ---
            velocities[i] = update_velocity_priority(
                velocities[i], 
                population[i].priority_vec, 
                pbests[i].priority_vec, 
                gbest,
                w, c1, c2
            )
            
            new_pos = update_position_priority(population[i].priority_vec, velocities[i])
            
            # Evaluate fitness
            new_cost = fast_fitness_priority(
                new_pos, substrate, vnr,
                domain_masks, available_masks, node_to_idx
            )
            
            # Update pbest
            if new_cost < pbests[i].fitness:
                pbests[i] = PriorityIndividual(new_pos.copy(), velocities[i], new_cost)
                if new_cost < gbest_cost:
                    gbest = new_pos.copy()
                    gbest_cost = new_cost
            
            # --- 2. SA STEP ---
            sa_cand = sa_neighbor_priority(new_pos, mutation_rate)
            sa_cost = fast_fitness_priority(
                sa_cand, substrate, vnr,
                domain_masks, available_masks, node_to_idx
            )
            
            # Accept based on SA criterion (simplified - always accept better)
            if sa_cost < new_cost:
                new_pos = sa_cand
                new_cost = sa_cost
                
                if sa_cost < pbests[i].fitness:
                    pbests[i] = PriorityIndividual(sa_cand.copy(), velocities[i], sa_cost)
                    if sa_cost < gbest_cost:
                        gbest = sa_cand.copy()
                        gbest_cost = sa_cost
            
            # Update population
            population[i] = PriorityIndividual(new_pos, velocities[i], new_cost)
    
    # ---- Select TOP-K ----
    # Filter feasible solutions (fitness < INFEASIBLE_PENALTY)
    feasible = [(ind.priority_vec, ind.fitness) for ind in population 
                if ind.fitness < INFEASIBLE_PENALTY]
    
    # Sort by fitness (lower is better for priority encoding)
    feasible.sort(key=lambda x: x[1])
    
    return feasible[:top_k]


# ============================================================
# 3. PARALLEL PSO FOR ALL VNRS
# ============================================================

def solve_all_vnrs_priority_parallel(
    vnr_list, 
    substrate, 
    pop_size, 
    generations, 
    top_k,
    w_max=0.9,
    w_min=0.4,
    c1=2.0,
    c2=2.0,
    mutation_rate=0.1
):
    """
    Run priority vector PSO in parallel for all VNRs.
    
    Returns:
        results: list of list of (priority_vec, fitness) tuples
                 results[k] = top-k candidates for VNR k
    """
    tasks = [
        (vnr, substrate, pop_size, generations, top_k, w_max, w_min, c1, c2, mutation_rate)
        for vnr in vnr_list
    ]
    
    n_proc = min(cpu_count(), len(tasks))
    with Pool(processes=n_proc) as pool:
        results = pool.map(evolve_vnr_priority, tasks)
    
    return results


# ============================================================
# 4. DECODE AND BUILD SOLUTION
# ============================================================

def build_solution_from_priority(priority_vec, substrate, vnr):
    """
    Decode priority vector to full solution with link mapping.
    
    Args:
        priority_vec: np.array - priority vector
        substrate: substrate network
        vnr: virtual network request
        
    Returns:
        mapping: dict {vnode: snode} or None
        link_paths: dict {(u,v): path} or None
    """
    # Decode to node mapping
    domain_masks, node_to_idx = build_domain_masks(substrate)
    available_masks = build_available_masks(substrate, vnr, node_to_idx)
    
    mapping = decode_priority_vector(
        priority_vec, substrate, vnr,
        domain_masks, available_masks, node_to_idx
    )
    
    if mapping is None:
        return None, None
    
    # Build link paths using shortest path with capacity
    link_paths = {}
    for (u, v) in vnr.edges():
        bw = vnr.edges[u, v]['bw']
        path = shortest_path_with_capacity(
            substrate, mapping[u], mapping[v], bw
        )
        if path is None:
            return None, None
        link_paths[(u, v)] = path
    
    return mapping, link_paths


# ============================================================
# 5. RESERVATION (TOP-K TRY)
# ============================================================

def reserve_with_topk_priority(
    substrate,
    vnr_list,
    revenues,
    candidates_per_vnr
):
    """
    Try to reserve resources using top-K candidates for each VNR.
    Uses revenue/cost ratio for prioritization.
    
    Args:
        substrate: substrate network (will be modified)
        vnr_list: list of VNRs
        revenues: list of revenues for each VNR
        candidates_per_vnr: list of list of (priority_vec, fitness) for each VNR
        
    Returns:
        accepted: dict {vnr_idx: Solution}
        rejected: set of vnr indices
    """
    substrate_reserved = copy.deepcopy(substrate)
    
    accepted = {}
    rejected = set()
    
    # ---- Compute scores for prioritization ----
    scores = []
    for i, cands in enumerate(candidates_per_vnr):
        if not cands:
            continue
        
        # Best candidate (lowest fitness/cost)
        best_priority_vec, best_cost = cands[0]
        
        # Score = cost / revenue (lower is better)
        score = best_cost / (revenues[i] + 1e-6)
        scores.append((score, i))
    
    scores.sort()  # Lower score = higher priority
    
    # ---- Reservation loop ----
    for _, k in scores:
        vnr = vnr_list[k]
        placed = False
        
        for priority_vec, fitness in candidates_per_vnr[k]:
            # Build solution
            mapping, link_paths = build_solution_from_priority(
                priority_vec, substrate_reserved, vnr
            )
            
            if mapping is None or link_paths is None:
                continue
            
            # Check feasibility
            feasible = True
            
            # Check node capacity
            for v_node, s_node in mapping.items():
                cpu = vnr.nodes[v_node]['cpu']
                if not can_place_node(substrate_reserved, s_node, cpu):
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            # Check link capacity
            for (u, v), path in link_paths.items():
                bw = vnr.edges[u, v]['bw']
                for i in range(len(path) - 1):
                    if substrate_reserved[path[i]][path[i+1]]['bw'] < bw:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if not feasible:
                continue
            
            # ---- Reserve resources ----
            for v_node, s_node in mapping.items():
                reserve_node(substrate_reserved, s_node, vnr.nodes[v_node]['cpu'])
            
            for (u, v), path in link_paths.items():
                reserve_path(substrate_reserved, path, vnr.edges[u, v]['bw'])
            
            # Store solution
            sol = Solution(vnr, mapping, link_paths)
            accepted[k] = sol
            placed = True
            break
        
        if not placed:
            rejected.add(k)
    
    # Update original substrate with reserved state
    substrate.clear()
    substrate.update(substrate_reserved)
    
    return accepted, rejected


# ============================================================
# 6. TWO-STAGE REPAIR HELPERS
# ============================================================

def detect_infeasible_nodes(mapping, substrate, vnr):
    """Detect nodes that don't have enough CPU capacity."""
    infeasible = []
    
    for vnode, snode in mapping.items():
        required_cpu = vnr.nodes[vnode]['cpu']
        available_cpu = substrate.nodes[snode]['cpu']
        
        if available_cpu < required_cpu:
            infeasible.append(vnode)
    
    return infeasible


def detect_infeasible_links(mapping, substrate, vnr, link_paths):
    """Detect links that cannot be satisfied due to insufficient bandwidth."""
    infeasible = []
    reasons = {}
    
    for (u, v) in vnr.edges():
        required_bw = vnr.edges[u, v]['bw']
        
        # Check if path exists
        if (u, v) not in link_paths:
            infeasible.append((u, v))
            reasons[(u, v)] = "No path found"
            continue
        
        path = link_paths[(u, v)]
        
        # Check if path is valid
        if path is None or len(path) < 2:
            infeasible.append((u, v))
            reasons[(u, v)] = "Invalid path"
            continue
        
        # Check if path starts/ends at correct nodes
        if path[0] != mapping[u] or path[-1] != mapping[v]:
            infeasible.append((u, v))
            reasons[(u, v)] = f"Path mismatch"
            continue
        
        # Check bandwidth on each edge
        for i in range(len(path) - 1):
            edge_bw = substrate[path[i]][path[i+1]].get('bw', 0)
            if edge_bw < required_bw:
                infeasible.append((u, v))
                reasons[(u, v)] = f"Insufficient BW on edge ({path[i]},{path[i+1]})"
                break
    
    return infeasible, reasons


def find_feasible_nodes_same_domain(substrate, vnode, vnr, current_mapping):
    """Find all substrate nodes in the same domain that can host vnode."""
    required_cpu = vnr.nodes[vnode]['cpu']
    domain = vnr.nodes[vnode].get('domain', None)
    
    feasible = []
    used_snodes = set(current_mapping.values())
    
    for snode in substrate.nodes():
        # Skip if already used
        if snode in used_snodes:
            continue
        
        # Check domain match
        if domain is not None:
            snode_domain = substrate.nodes[snode].get('domain', None)
            if snode_domain != domain:
                continue
        
        # Check CPU capacity
        available_cpu = substrate.nodes[snode]['cpu']
        if available_cpu >= required_cpu:
            feasible.append(snode)
    
    return feasible


def repair_node_mapping(mapping, substrate, vnr, infeasible_nodes):
    """Try to repair node mapping by remapping infeasible nodes."""
    new_mapping = copy.deepcopy(mapping)
    changed_nodes = []
    
    for vnode in infeasible_nodes:
        # Find feasible alternatives
        feasible_snodes = find_feasible_nodes_same_domain(substrate, vnode, vnr, new_mapping)
        
        if not feasible_snodes:
            return None, []
        
        # Choose the one with most available CPU
        best_snode = max(feasible_snodes, key=lambda s: substrate.nodes[s]['cpu'])
        
        # Update mapping
        new_mapping[vnode] = best_snode
        changed_nodes.append(vnode)
    
    return new_mapping, changed_nodes


def rebuild_all_paths(mapping, substrate, vnr):
    """Rebuild all paths for all virtual links using shortest path."""
    link_paths = {}
    
    for (u, v) in vnr.edges():
        src = mapping[u]
        dst = mapping[v]
        
        try:
            path = nx.shortest_path(substrate, src, dst)
            link_paths[(u, v)] = path
        except nx.NetworkXNoPath:
            return None
    
    return link_paths


def check_path_capacity(path, substrate, bw):
    """Check if path has sufficient bandwidth on all edges."""
    if path is None or len(path) < 2:
        return False
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if substrate[u][v].get('bw', 0) < bw:
            return False
    return True


def repair_link_paths(mapping, substrate, vnr, infeasible_links):
    """Try to repair link paths by finding alternative routes."""
    new_link_paths = {}
    
    # First, add all feasible links (those not in infeasible_links)
    for (u, v) in vnr.edges():
        if (u, v) in infeasible_links:
            continue
        
        src = mapping[u]
        dst = mapping[v]
        bw = vnr.edges[u, v]['bw']
        
        try:
            path = nx.shortest_path(substrate, src, dst)
            if check_path_capacity(path, substrate, bw):
                new_link_paths[(u, v)] = path
            else:
                return None
        except nx.NetworkXNoPath:
            return None
    
    # Now repair infeasible links using bandwidth-filtered graph
    for (u, v) in infeasible_links:
        src = mapping[u]
        dst = mapping[v]
        bw = vnr.edges[u, v]['bw']
        
        # Create a filtered substrate graph
        filtered_substrate = nx.DiGraph() if substrate.is_directed() else nx.Graph()
        filtered_substrate.add_nodes_from(substrate.nodes(data=True))
        
        # Add only edges with sufficient bandwidth
        for (s_u, s_v, data) in substrate.edges(data=True):
            edge_bw = data.get('bw', 0)
            if edge_bw >= bw:
                filtered_substrate.add_edge(s_u, s_v, **data)
        
        # Find shortest path on filtered graph
        try:
            path = nx.shortest_path(filtered_substrate, src, dst)
            if check_path_capacity(path, substrate, bw):
                new_link_paths[(u, v)] = path
            else:
                return None
        except nx.NetworkXNoPath:
            return None
    
    return new_link_paths


# ============================================================
# 7. TWO-STAGE REPAIR
# ============================================================

def two_stage_repair_priority(vnr, priority_vec, substrate, vnr_idx, verbose=True):
    """
    Two-stage repair process for priority vector encoding.
    
    Stage 1: Repair infeasible node mappings
    Stage 2: Repair infeasible link paths
    
    Args:
        vnr: virtual network request
        priority_vec: np.array - best priority vector from PSO
        substrate: substrate network
        vnr_idx: index of VNR for logging
        verbose: whether to print detailed logs
    
    Returns:
        mapping: dict {vnode: snode} or None
        link_paths: dict {(u,v): path} or None
        success: bool
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[TWO-STAGE REPAIR] Starting repair for VNR {vnr_idx}")
        print(f"{'='*60}")
    
    # Build initial mapping from priority vector
    mapping, link_paths = build_solution_from_priority(priority_vec, substrate, vnr)
    
    if mapping is None:
        if verbose:
            print(f"[STAGE 0] Initial decoding FAILED")
        return None, None, False
    
    if verbose:
        print(f"\n[STAGE 0: INITIAL MAPPING]")
        for vnode, snode in mapping.items():
            required_cpu = vnr.nodes[vnode]['cpu']
            available_cpu = substrate.nodes[snode]['cpu']
            status = "" if available_cpu >= required_cpu else ""
            print(f"  {status} vnode {vnode} -> snode {snode} (require {required_cpu}, available {available_cpu})")
    
    # ============================================================
    # STAGE 1: NODE REPAIR
    # ============================================================
    if verbose:
        print(f"\n[STAGE 1: NODE REPAIR]")
    
    infeasible_nodes = detect_infeasible_nodes(mapping, substrate, vnr)
    
    if infeasible_nodes:
        if verbose:
            print(f"  Found {len(infeasible_nodes)} infeasible nodes: {infeasible_nodes}")
        
        # Try to repair
        new_mapping, changed_nodes = repair_node_mapping(mapping, substrate, vnr, infeasible_nodes)
        
        if new_mapping is None:
            if verbose:
                print(f"   Node repair FAILED")
            return None, None, False
        
        if verbose:
            print(f"   Node repair SUCCESS: Remapped {len(changed_nodes)} nodes")
        
        mapping = new_mapping
        
        # Rebuild paths due to node changes
        link_paths = rebuild_all_paths(mapping, substrate, vnr)
        
        if link_paths is None:
            if verbose:
                print(f"   Path rebuilding FAILED")
            return None, None, False
    else:
        if verbose:
            print(f"   All nodes are feasible")
    
    # ============================================================
    # STAGE 2: LINK REPAIR
    # ============================================================
    if verbose:
        print(f"\n[STAGE 2: LINK REPAIR]")
    
    infeasible_links, reasons = detect_infeasible_links(mapping, substrate, vnr, link_paths)
    
    if infeasible_links:
        if verbose:
            print(f"  Found {len(infeasible_links)} infeasible links")
        
        # Try to repair
        new_link_paths = repair_link_paths(mapping, substrate, vnr, infeasible_links)
        
        if new_link_paths is None:
            if verbose:
                print(f"   Link repair FAILED")
            return None, None, False
        
        if verbose:
            print(f"   Link repair SUCCESS")
        
        link_paths = new_link_paths
    else:
        if verbose:
            print(f"   All links are feasible")
    
    # ============================================================
    # FINAL VERIFICATION
    # ============================================================
    if verbose:
        print(f"\n[FINAL VERIFICATION]")
    
    infeasible_nodes_final = detect_infeasible_nodes(mapping, substrate, vnr)
    infeasible_links_final, _ = detect_infeasible_links(mapping, substrate, vnr, link_paths)
    
    if infeasible_nodes_final or infeasible_links_final:
        if verbose:
            print(f"   FAILED: Still have infeasible elements")
        return None, None, False
    
    if verbose:
        print(f"   SUCCESS: All nodes and links are feasible!")
        print(f"{'='*60}\n")
    
    return mapping, link_paths, True


# ============================================================
# 8. PUBLIC API
# ============================================================

def embed_batch(
    substrate, 
    batch, 
    pop_size=50, 
    generations=100, 
    top_k=5,
    w_max=0.9,
    w_min=0.4,
    c1=2.0,
    c2=2.0,
    mutation_rate=0.3,
    verbose=True
):
    accepted, rejected = [], []
    
    vnr_list = [vnr for vnr, _ in batch]
    revenues = [revenue_of_vnr(vnr) for vnr in vnr_list]
    
    # ============================================================
    # PHASE 1: PARALLEL PSO
    # ============================================================
    if verbose:
        print(f"\n{'='*60}")
        print(f"PHASE 1: Running parallel PSO for {len(vnr_list)} VNRs")
        print(f"{'='*60}")
    
    candidates = solve_all_vnrs_priority_parallel(
        vnr_list, substrate, pop_size, generations, top_k,
        w_max, w_min, c1, c2, mutation_rate
    )
    
    if verbose:
        for i, cands in enumerate(candidates):
            print(f"VNR {i}: {len(cands)} candidates")
    
    # ============================================================
    # PHASE 2: RESERVATION
    # ============================================================
    if verbose:
        print(f"\n{'='*60}")
        print(f"PHASE 2: Resource reservation")
        print(f"{'='*60}")
    
    locked, rejected_idx = reserve_with_topk_priority(
        substrate, vnr_list, revenues, candidates
    )
    
    # Build accepted and rejected lists
    for i in range(len(vnr_list)):
        if i in locked:
            sol = locked[i]
            accepted.append((sol.vnr, sol.mapping, sol.link_paths))
            if verbose:
                print(f"VNR {i}: ACCEPTED (direct reservation)")
        else:
            rejected.append(vnr_list[i])
            if verbose:
                print(f"VNR {i}: REJECTED (will try repair)")
    
    # If all accepted, return early
    if len(rejected) == 0:
        if verbose:
            print(f"\n{'='*60}")
            print(f"All VNRs accepted! No repair needed.")
            print(f"{'='*60}\n")
        return accepted, rejected
    
    # ============================================================
    # PHASE 3: TWO-STAGE REPAIR
    # ============================================================
    if verbose:
        print(f"\n{'#'*60}")
        print(f"# PHASE 3: TWO-STAGE REPAIR")
        print(f"# Processing {len(rejected)} rejected VNRs...")
        print(f"{'#'*60}")
    
    repair_accepted = []
    final_rejected = []
    
    for vnr in rejected:
        # Get original index
        original_idx = vnr_list.index(vnr)
        
        # Get best candidate
        if original_idx >= len(candidates) or not candidates[original_idx]:
            if verbose:
                print(f"\n[VNR {original_idx}] No candidates available, skipping repair")
            final_rejected.append(vnr)
            continue
        
        best_priority_vec, _ = candidates[original_idx][0]
        
        # Try two-stage repair
        mapping, link_paths, success = two_stage_repair_priority(
            vnr, best_priority_vec, substrate, original_idx, verbose=verbose
        )
        
        if not success:
            if verbose:
                print(f"[VNR {original_idx}] Two-stage repair FAILED")
            final_rejected.append(vnr)
            continue
        
        # Try to reserve resources
        if verbose:
            print(f"\n[VNR {original_idx}] Attempting resource reservation...")
        
        can_reserve = True
        
        # Check node capacity
        for vnode, snode in mapping.items():
            if not can_place_node(substrate, snode, vnr.nodes[vnode]['cpu']):
                can_reserve = False
                if verbose:
                    print(f"   Cannot reserve node {snode}")
                break
        
        if can_reserve:
            # Check link capacity
            for (u, v), path in link_paths.items():
                bw = vnr.edges[u, v]['bw']
                for i in range(len(path) - 1):
                    if substrate[path[i]][path[i+1]]['bw'] < bw:
                        can_reserve = False
                        if verbose:
                            print(f"   Cannot reserve edge ({path[i]},{path[i+1]})")
                        break
                if not can_reserve:
                    break
        
        if can_reserve:
            # Reserve resources
            if verbose:
                print(f"   Reserving resources...")
            
            for vnode, snode in mapping.items():
                reserve_node(substrate, snode, vnr.nodes[vnode]['cpu'])
            
            for (u, v), path in link_paths.items():
                reserve_path(substrate, path, vnr.edges[u, v]['bw'])
            
            repair_accepted.append((vnr, mapping, link_paths))
            if verbose:
                print(f"   VNR {original_idx} RECOVERED via two-stage repair!")
        else:
            if verbose:
                print(f"   Reservation failed for VNR {original_idx}")
            final_rejected.append(vnr)
    
    # Merge repair accepted with original accepted
    accepted.extend(repair_accepted)
    
    if verbose:
        print(f"\n{'#'*60}")
        print(f"# TWO-STAGE REPAIR SUMMARY")
        print(f"# Total rejected VNRs: {len(rejected)}")
        print(f"# Recovered VNRs: {len(repair_accepted)}")
        print(f"# Still rejected: {len(final_rejected)}")
        if len(rejected) > 0:
            print(f"# Recovery rate: {len(repair_accepted)/len(rejected)*100:.1f}%")
        print(f"{'#'*60}\n")
    
    return accepted, final_rejected