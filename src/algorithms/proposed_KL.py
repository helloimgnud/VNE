# src/algorithms/proposed_KL.py
"""
Proposed VNE algorithm with KL divergence-based similarity for knowledge transfer.
Based on proposed.py, replacing task_similarity with kl_similarity.
"""
import random
import copy
import numpy as np

# Updated imports from the new hpso.py
from src.algorithms.hpso import (
    operation_minus,
    operation_plus,
    operation_multiply,
    sa_neighbor,
    init_particles_hpso
)

from src.utils.graph_utils import (
    shortest_path_with_capacity,
    can_place_node,
    reserve_node,
    reserve_path
)

from src.evaluation.eval import cost_of_embedding

# Fitness thấp vô cùng cho giải pháp không khả thi
INFEASIBLE_FITNESS = -1e12


# ============================================================
# Data Structures
# ============================================================

class Individual:
    def __init__(self, particle, velocity, fitness):
        self.particle = particle
        self.velocity = velocity
        self.fitness = fitness


class MappingSolution:
    def __init__(self, vnr, particle, mapping, link_paths, fitness):
        self.vnr = vnr
        self.particle = particle
        self.mapping = mapping
        self.link_paths = link_paths
        self.fitness = fitness


class VNRContext:
    """
    Each VNR is solved by exactly ONE algorithm.
    Added 'final_solution' to support Early Stopping.
    Added 'allocated' to support Dynamic Resource Reservation.
    """
    def __init__(self, vnr, algo, pop, pbest=None):
        self.vnr = vnr
        self.algo = algo          # "PSO" or "VNS"
        self.pop = pop
        self.pbest = pbest        # only for PSO
        self.hist = []            # best fitness history
        self.final_solution = None # Early stopping result
        self.allocated = False     # Dynamic reservation: resources already reserved


def get_candidates(vnr, vnode, substrate):
    return {
        s for s in substrate.nodes
        if substrate.nodes[s]['cpu'] >= vnr.nodes[vnode]['cpu']
    }


def kl_divergence(p, q, epsilon=1e-10):
    """
    Compute KL divergence D_KL(P || Q).
    
    Args:
        p: Probability distribution P (numpy array)
        q: Probability distribution Q (numpy array)
        epsilon: Small value to avoid log(0)
    
    Returns:
        KL divergence value
    """
    # Add epsilon to avoid log(0)
    p = np.asarray(p, dtype=np.float64) + epsilon
    q = np.asarray(q, dtype=np.float64) + epsilon
    
    # Normalize to ensure they sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return np.sum(p * np.log(p / q))


def js_divergence(p, q, epsilon=1e-10):
    """
    Compute Jensen-Shannon divergence (symmetric version of KL).
    
    JS(P, Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
    where M = 0.5 * (P + Q)
    
    Args:
        p: Probability distribution P (numpy array)
        q: Probability distribution Q (numpy array)
        epsilon: Small value to avoid log(0)
    
    Returns:
        JS divergence value (0 = identical, higher = more different)
    """
    p = np.asarray(p, dtype=np.float64) + epsilon
    q = np.asarray(q, dtype=np.float64) + epsilon
    
    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Compute M
    m = 0.5 * (p + q)
    
    return 0.5 * kl_divergence(p, m, epsilon=0) + 0.5 * kl_divergence(q, m, epsilon=0)


def kl_similarity(vnr_a, vnr_b, substrate):

    vnodes_a = list(vnr_a.nodes())
    vnodes_b = list(vnr_b.nodes())
    
    min_len = min(len(vnodes_a), len(vnodes_b))
    
    if min_len == 0:
        return 0.0
    
    # Get all substrate nodes for building distributions
    sub_nodes = list(substrate.nodes())
    n_sub = len(sub_nodes)
    
    if n_sub == 0:
        return 0.0
    
    # Create node to index mapping
    node_to_idx = {node: idx for idx, node in enumerate(sub_nodes)}
    
    total_similarity = 0.0
    
    for i in range(min_len):
        # Get candidates for this vnode position
        cand_a = get_candidates(vnr_a, vnodes_a[i], substrate)
        cand_b = get_candidates(vnr_b, vnodes_b[i], substrate)
        
        # Build probability distributions over substrate nodes
        # Uniform distribution over candidates
        p_a = np.zeros(n_sub)
        p_b = np.zeros(n_sub)
        
        for node in cand_a:
            if node in node_to_idx:
                p_a[node_to_idx[node]] = 1.0
        
        for node in cand_b:
            if node in node_to_idx:
                p_b[node_to_idx[node]] = 1.0
        
        # Normalize to get probability distributions
        sum_a = np.sum(p_a)
        sum_b = np.sum(p_b)
        
        if sum_a > 0 and sum_b > 0:
            p_a = p_a / sum_a
            p_b = p_b / sum_b
            
            # Compute JS divergence (0 = identical, max ~0.69 for disjoint)
            js_div = js_divergence(p_a, p_b)
            
            # Convert divergence to similarity: lower divergence = higher similarity
            # Similarity = 1 / (1 + JS_div) maps [0, inf) to (0, 1]
            similarity = 1.0 / (1.0 + js_div)
            total_similarity += similarity
        elif sum_a == 0 and sum_b == 0:
            # Both have no candidates - they are similar in this regard
            total_similarity += 1.0
        # else: one has candidates, other doesn't - adds 0 similarity
    
    return total_similarity


# ============================================================
# Population-Based KL Divergence for Knowledge Transfer
# ============================================================

def population_mapping_distribution(pop, vnode_idx, substrate):
    """
    Build probability distribution of where particles in a population
    map a specific virtual node position to substrate nodes.
    
    Args:
        pop: List of Individual objects (population)
        vnode_idx: Index of the virtual node to analyze
        substrate: Substrate network graph
        
    Returns:
        numpy array of probabilities over all substrate nodes
    """
    sub_nodes = list(substrate.nodes())
    n_sub = len(sub_nodes)
    
    if n_sub == 0:
        return np.zeros(0)
    
    node_to_idx = {node: idx for idx, node in enumerate(sub_nodes)}
    counts = np.zeros(n_sub)
    
    for ind in pop:
        if vnode_idx < len(ind.particle):
            snode = ind.particle[vnode_idx]
            if snode in node_to_idx:
                counts[node_to_idx[snode]] += 1
    
    total = np.sum(counts)
    if total > 0:
        return counts / total
    return counts


def population_kl_divergence(pop_a, pop_b, substrate, epsilon=1e-10):
    """
    Compute KL divergence between mapping distributions of two populations.
    
    Measures how differently two populations map virtual nodes to substrate nodes.
    - Low divergence = populations map virtual nodes similarly
    - High divergence = populations are diverse in their mappings
    
    Args:
        pop_a: First population (list of Individual)
        pop_b: Second population (list of Individual)
        substrate: Substrate network graph
        epsilon: Small value to avoid log(0)
        
    Returns:
        Average JS divergence across all comparable virtual node positions
    """
    if not pop_a or not pop_b:
        return 0.0
    
    # Get the minimum particle length across both populations
    max_len_a = max(len(ind.particle) for ind in pop_a) if pop_a else 0
    max_len_b = max(len(ind.particle) for ind in pop_b) if pop_b else 0
    min_len = min(max_len_a, max_len_b)
    
    if min_len == 0:
        return 0.0
    
    total_divergence = 0.0
    valid_positions = 0
    
    for i in range(min_len):
        dist_a = population_mapping_distribution(pop_a, i, substrate)
        dist_b = population_mapping_distribution(pop_b, i, substrate)
        
        # Only compute if both distributions have data
        if np.sum(dist_a) > 0 and np.sum(dist_b) > 0:
            js_div = js_divergence(dist_a, dist_b, epsilon)
            total_divergence += js_div
            valid_positions += 1
    
    if valid_positions > 0:
        return total_divergence / valid_positions
    return 0.0


# ============================================================
# 1. Optimization: Fast Fitness (No Dijkstra)
# ============================================================

def compute_fast_fitness(vnr, particle, substrate):
    """
    [OPTIMIZATION] Fast fitness from pso.py:
    - Check CPU feasibility.
    - Estimate link cost via hop proxy (no Dijkstra).
    - Returns negative cost for maximization.
    """
    vnodes = list(vnr.nodes())
    mapping = {vnodes[i]: particle[i] for i in range(len(vnodes))}
    
    # 1. Check Node Constraints (CPU)
    for v, s in mapping.items():
        if substrate.nodes[s]['cpu'] < vnr.nodes[v]['cpu']:
            return INFEASIBLE_FITNESS

    # 2. Estimate Cost (Node Cost + Link Proxy Cost)
    # Link Proxy: Sum of bandwidths for inter-node links (assuming 1 hop)
    est_cost = 0
    
    # Node cost component (Simplified)
    for v, s in mapping.items():
        est_cost += vnr.nodes[v]['cpu'] # Or substrate.nodes[s]['cost'] if available

    # Link cost component
    for u, v in vnr.edges():
        if mapping[u] != mapping[v]:
            est_cost += vnr.edges[u, v]['bw']
    
    return -est_cost


# ============================================================
# 2. Optimization: Full Check for Early Stopping
# ============================================================

def build_full_solution_check(vnr, particle, substrate):
    """
    [OPTIMIZATION] Try to build a valid solution with Dijkstra.
    Used for Early Stopping check.
    Does NOT reserve resources, just checks feasibility.
    """
    vnodes = list(vnr.nodes())
    mapping = {vnodes[i]: particle[i] for i in range(len(vnodes))}
    
    # 1. Check Node CPU again (Strict)
    for v, s in mapping.items():
        if substrate.nodes[s]['cpu'] < vnr.nodes[v]['cpu']:
            return None

    # 2. Check Link Mapping (Dijkstra)
    link_paths = {}
    total_cost = 0
    
    # Calc node cost
    # Assuming cost_per_unit is 1 for simplicity or read from substrate
    for v, s in mapping.items():
         total_cost += vnr.nodes[v]['cpu']

    for u, v in vnr.edges():
        su = mapping[u]
        sv = mapping[v]
        bw_req = vnr.edges[u, v]['bw']

        path = shortest_path_with_capacity(substrate, su, sv, bw_req)
        if path is None:
            return None
        
        link_paths[(u, v)] = path
        total_cost += (len(path) - 1) * bw_req

    # Return valid solution object
    return MappingSolution(vnr, particle, mapping, link_paths, -total_cost)


# ============================================================
# Resource Commit (Final Phase)
# ============================================================

def verify_and_allocate(substrate, solution):
    vnr = solution.vnr

    for v, s in solution.mapping.items():
        if not can_place_node(substrate, s, vnr.nodes[v]['cpu']):
            return False

    for (u, v), path in solution.link_paths.items():
        bw = vnr.edges[u, v]['bw']
        for i in range(len(path) - 1):
            if substrate.edges[path[i], path[i + 1]]['bw'] < bw:
                return False

    for v, s in solution.mapping.items():
        reserve_node(substrate, s, vnr.nodes[v]['cpu'])

    for (u, v), path in solution.link_paths.items():
        reserve_path(substrate, path, vnr.edges[u, v]['bw'])

    return True


# ============================================================
# Population Initialization
# ============================================================

def initialize_population(vnr, substrate, pop_size):
    # Use HPSO's initialization strategy
    particles = init_particles_hpso(substrate, vnr, pop_size)
    if particles is None:
        particles = []
    
    vnodes = list(vnr.nodes())
    sub_nodes = list(substrate.nodes())
    
    # Nếu init_particles_hpso trả về ít hơn pop_size (hoặc rỗng),
    # ta điền thêm bằng các particle ngẫu nhiên.
    while len(particles) < pop_size:
        if not sub_nodes: # Trường hợp mạng vật lý rỗng 
            break 
        # Tạo particle ngẫu nhiên: gán mỗi vnode vào một snode bất kỳ
        p = [random.choice(sub_nodes) for _ in vnodes]
        particles.append(p)
    pop = []
    vnodes = list(vnr.nodes())
    
    for particle in particles:
        velocity = [random.randint(0, 1) for _ in vnodes]
        # Use Fast Fitness for initialization
        fit = compute_fast_fitness(vnr, particle, substrate)
        pop.append(Individual(particle, velocity, fit))
        
    return pop


# ============================================================
# PSO Update
# ============================================================

def update_pso(pop, pbest, gbest, vnr, substrate,
               alpha=0.5, beta=0.3, gamma=0.2):

    for i, ind in enumerate(pop):
        diff_p = operation_minus(pbest[i].particle, ind.particle)
        diff_g = operation_minus(gbest.particle, ind.particle)

        v = operation_plus(alpha, ind.velocity, beta, diff_p)
        v = operation_plus(1 - gamma, v, gamma, diff_g)

        ind.particle = operation_multiply(
            ind.particle, v, vnr, substrate
        )
        ind.velocity = v


# ============================================================
# VNS Update
# ============================================================

def update_vns(pop, vnr, substrate, max_iter=5):
    for ind in pop:
        best_particle = ind.particle
        # Use Fast Fitness
        best_fit = compute_fast_fitness(vnr, best_particle, substrate)

        for _ in range(max_iter):
            neigh = sa_neighbor(best_particle, substrate, vnr)
            fit = compute_fast_fitness(vnr, neigh, substrate)
            if fit > best_fit:
                best_particle = neigh
                best_fit = fit

        ind.particle = best_particle
        ind.fitness = best_fit


# ============================================================
# Knowledge Transfer (Domain-aware, Discrete)
# ============================================================

def transfer_knowledge(source_best, target_pop,
                       vnr, substrate,
                       rate=0.5):

    vnodes = list(vnr.nodes())
    sub_nodes = list(substrate.nodes())
    source_len = len(source_best.particle)

    for ind in target_pop:
        for i, v in enumerate(vnodes):
            if i >= source_len:
                continue

            if random.random() > rate:
                continue

            gene = source_best.particle[i]

            # -------- 1. Candidate check --------
            candidates = [
                s for s in sub_nodes
                if substrate.nodes[s]['cpu'] >= vnr.nodes[v]['cpu']
            ]

            if gene not in candidates:
                # fallback: random candidate
                if candidates:
                    ind.particle[i] = random.choice(candidates)
                continue

            ind.particle[i] = gene


def is_stagnant(hist, window=5, eps=1e-4):
    if len(hist) < window:
        return False
    return abs(hist[-1] - hist[-window]) < eps


# ============================================================
# Core MP-PVA (Cross-VNR) with Early Stopping & Dynamic Reservation - Using KL Similarity
# ============================================================

def run_algorithm(vnr_list, substrate,
               pop_size=20, generations=30):
    """
    Run MP-PVA optimization with Dynamic Resource Reservation.
    Uses KL divergence-based similarity for knowledge transfer.
    
    When a VNR finds a valid solution, resources are immediately reserved
    on the LIVE substrate so other VNRs optimize against updated state.
    
    Args:
        vnr_list: List of VNR graphs to embed
        substrate: LIVE substrate graph (resources will be modified)
        pop_size: Population size for optimization
        generations: Number of generations to run
        
    Returns:
        List of (MappingSolution, already_allocated) tuples
    """
    contexts = []

    # ---- Assign ONE algorithm per VNR ----
    for vnr in vnr_list:
        algo = "PSO" if random.random() < 0.5 else "VNS"
        pop = initialize_population(vnr, substrate, pop_size)
        pbest = copy.deepcopy(pop) if algo == "PSO" else None
        contexts.append(VNRContext(vnr, algo, pop, pbest))

    # ---- Evolution ----
    for gen in range(generations):

        # --- Step each VNR ---
        for ctx in contexts:
            # [DYNAMIC RESERVATION] If already allocated, skip
            if ctx.allocated:
                continue

            vnr = ctx.vnr
            
            # 1. Update Population (using Fast Fitness on CURRENT substrate)
            if ctx.algo == "PSO":
                gbest = max(ctx.pop, key=lambda x: x.fitness)
                update_pso(ctx.pop, ctx.pbest, gbest,
                           vnr, substrate)

                for i in range(len(ctx.pop)):
                    fit = compute_fast_fitness(vnr, ctx.pop[i].particle, substrate)
                    ctx.pop[i].fitness = fit
                    if fit > ctx.pbest[i].fitness:
                        ctx.pbest[i] = copy.deepcopy(ctx.pop[i])

                gbest = max(ctx.pop, key=lambda x: x.fitness)
                ctx.hist.append(gbest.fitness)

            else:  # VNS
                update_vns(ctx.pop, vnr, substrate)
                gbest = max(ctx.pop, key=lambda x: x.fitness)
                ctx.hist.append(gbest.fitness)

            
            # [DYNAMIC RESERVATION] 
            if gbest.fitness > INFEASIBLE_FITNESS:
                full_sol = build_full_solution_check(vnr, gbest.particle, substrate)
                if full_sol is not None:
                    # Try to immediately reserve resources on the LIVE substrate
                    if verify_and_allocate(substrate, full_sol):
                        # Success! Resources reserved, stop evolving this VNR
                        ctx.final_solution = full_sol
                        ctx.allocated = True
                    

        # --- Cross-VNR Knowledge Transfer using Population-Based KL Divergence ---
        # Goal: Learn from tasks that are DIVERSE (different mapping behavior)
        #       but have SAME CANDIDATES (structural compatibility)
        for ctx in contexts:
            if ctx.allocated: continue # Skip if already allocated
            
            if not is_stagnant(ctx.hist):
                continue

            donors = [c for c in contexts if c.algo != ctx.algo] 
            
            if not donors:
                continue
            
            scored_donors = []

            for d in donors:
                # 1. Measure population divergence (higher = more diverse mappings)
                divergence = population_kl_divergence(ctx.pop, d.pop, substrate)
                
                # 2. Measure candidate overlap (higher = more compatible substrate nodes)
                candidate_sim = kl_similarity(ctx.vnr, d.vnr, substrate)
                
                # 3. We want: HIGH divergence + HIGH candidate overlap
                # This finds diverse populations that explore the SAME substrate space
                score = divergence * candidate_sim
                scored_donors.append((score, d))

            # Sort descending (higher score = better donor)
            scored_donors.sort(key=lambda x: x[0], reverse=True)

            top_k = scored_donors[:2]

            if not top_k:
                continue

            _, donor = random.choice(top_k)

            # If donor is solved, use its final solution particle, else use its current best
            if donor.final_solution:
                donor_best_particle = donor.final_solution.particle
                # Wrap in dummy individual for transfer function
                donor_best = Individual(donor_best_particle, None, 0)
            else:
                donor_best = max(donor.pop, key=lambda x: x.fitness)

            transfer_knowledge(
                donor_best,
                ctx.pop,
                ctx.vnr,
                substrate
            )

    # ---- Collect solutions ----
    # Solutions that were allocated during optimization are already committed
    # For non-allocated VNRs, try one final allocation attempt
    solutions = []
    for ctx in contexts:
        if ctx.allocated:
            # Already allocated - just add to solutions
            solutions.append((ctx.final_solution, True))  # (solution, already_allocated)
        else:
            # Try one last allocation attempt on the best particle
            best = max(ctx.pop, key=lambda x: x.fitness)
            full_sol = build_full_solution_check(ctx.vnr, best.particle, substrate)
            if full_sol and verify_and_allocate(substrate, full_sol):
                solutions.append((full_sol, True))
            else:
                # Failed to map
                failed_sol = MappingSolution(ctx.vnr, best.particle, None, None, INFEASIBLE_FITNESS)
                solutions.append((failed_sol, False))

    return solutions

def embed_batch(substrate, batch,
                        pop_size=20,
                        generations=30):
    """
    Embed a batch of VNRs with Dynamic Resource Reservation.
    
    Resources are reserved DURING optimization as solutions are found,
    not after all VNRs finish. This allows later VNRs to optimize
    against the updated substrate state.
    
    Args:
        substrate: LIVE substrate graph (will be modified in-place)
        batch: List of (vnr, revenue) tuples
        pop_size: Population size for optimization
        generations: Number of generations
        
    Returns:
        (accepted, rejected) where:
        - accepted: List of (vnr, mapping, link_paths) tuples
        - rejected: List of vnr graphs that failed
    """
    accepted = []
    rejected = []

    vnr_list = [vnr for vnr, _ in batch]
    
    # [DYNAMIC RESERVATION] Pass LIVE substrate, not a snapshot
    # Resources are reserved during optimization as solutions are found
    solutions = run_algorithm(
        vnr_list,
        substrate,  # Live substrate - will be modified in-place
        pop_size,
        generations
    )

    # Collect results - resources already reserved for successful solutions
    for sol, already_allocated in solutions:
        if sol.mapping is None:
            rejected.append(sol.vnr)
        elif already_allocated:
            # Resources already reserved during optimization
            accepted.append((sol.vnr, sol.mapping, sol.link_paths))
        else:
            # Should not happen with new logic, but handle gracefully
            rejected.append(sol.vnr)

    return accepted, rejected
