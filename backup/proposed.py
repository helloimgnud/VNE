# src/algorithms/proposed.py
import random
import copy

# Updated imports from the new hpso.py
from algorithms.hpso import (
    operation_minus,
    operation_plus,
    operation_multiply,
    sa_neighbor,
    init_particles_hpso
)

from utils.graph_utils import (
    shortest_path_with_capacity,
    can_place_node,
    reserve_node,
    reserve_path
)

from evaluation.eval import cost_of_embedding

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
    """
    def __init__(self, vnr, algo, pop, pbest=None):
        self.vnr = vnr
        self.algo = algo          # "PSO" or "VNS"
        self.pop = pop
        self.pbest = pbest        # only for PSO
        self.hist = []            # best fitness history
        self.final_solution = None # Early stopping result

def get_candidates(vnr, vnode, substrate):
    return {
        s for s in substrate.nodes
        if substrate.nodes[s]['cpu'] >= vnr.nodes[vnode]['cpu']
    }

def task_similarity(vnr_a, vnr_b, substrate):
    vnodes_a = list(vnr_a.nodes())
    vnodes_b = list(vnr_b.nodes())

    min_len = min(len(vnodes_a), len(vnodes_b))
    score = 0

    for i in range(min_len):
        cand_a = get_candidates(vnr_a, vnodes_a[i], substrate)
        cand_b = get_candidates(vnr_b, vnodes_b[i], substrate)
        score += len(cand_a & cand_b)

    return score

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
# Core MP-PVA (Cross-VNR) with Early Stopping
# ============================================================

def run_algorithm(vnr_list, substrate_snapshot,
               pop_size=20, generations=30):

    contexts = []

    # ---- Assign ONE algorithm per VNR ----
    for vnr in vnr_list:
        algo = "PSO" if random.random() < 0.5 else "VNS"
        pop = initialize_population(vnr, substrate_snapshot, pop_size)
        pbest = copy.deepcopy(pop) if algo == "PSO" else None
        contexts.append(VNRContext(vnr, algo, pop, pbest))

    # ---- Evolution ----
    for gen in range(generations):

        # --- Step each VNR ---
        for ctx in contexts:
            # [OPTIMIZATION] Early Stopping: If already solved, skip
            if ctx.final_solution is not None:
                continue

            vnr = ctx.vnr
            
            # 1. Update Population (using Fast Fitness)
            if ctx.algo == "PSO":
                gbest = max(ctx.pop, key=lambda x: x.fitness)
                update_pso(ctx.pop, ctx.pbest, gbest,
                           vnr, substrate_snapshot)

                for i in range(len(ctx.pop)):
                    fit = compute_fast_fitness(vnr, ctx.pop[i].particle, substrate_snapshot)
                    ctx.pop[i].fitness = fit
                    if fit > ctx.pbest[i].fitness:
                        ctx.pbest[i] = copy.deepcopy(ctx.pop[i])

                gbest = max(ctx.pop, key=lambda x: x.fitness)
                ctx.hist.append(gbest.fitness)

            else:  # VNS
                update_vns(ctx.pop, vnr, substrate_snapshot)
                gbest = max(ctx.pop, key=lambda x: x.fitness)
                ctx.hist.append(gbest.fitness)

            
            # If the current best solution looks feasible (fitness is not infeasible penalty),
            # try to build the full solution with Dijkstra.
            if gbest.fitness > INFEASIBLE_FITNESS:
                full_sol = build_full_solution_check(vnr, gbest.particle, substrate_snapshot)
                if full_sol is not None:
                    # Found a valid solution! Stop evolving this VNR.
                    ctx.final_solution = full_sol
                    continue

        # --- Cross-VNR Knowledge Transfer ---
        for ctx in contexts:
            if ctx.final_solution is not None: continue # Skip if solved
            
            if not is_stagnant(ctx.hist):
                continue

            donors = [c for c in contexts if c.algo != ctx.algo] 
            
            if not donors:
                continue
            
            scored_donors = []

            for d in donors:
                sim = task_similarity(ctx.vnr, d.vnr, substrate_snapshot)
                scored_donors.append((sim, d))

            # Sắp xếp giảm dần
            scored_donors.sort(key=lambda x: x[0], reverse=True)

            top_k = scored_donors[:2]

            if not top_k:
                continue

            _, donor = random.choice(top_k)

            donor = random.choice(donors)
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
                substrate_snapshot
            )

    # ---- Collect solutions ----
    solutions = []
    for ctx in contexts:
        if ctx.final_solution:
            solutions.append(ctx.final_solution)
        else:
            # If not solved early, try one last check on the best particle
            best = max(ctx.pop, key=lambda x: x.fitness)
            full_sol = build_full_solution_check(ctx.vnr, best.particle, substrate_snapshot)
            if full_sol:
                solutions.append(full_sol)
            else:
                # Failed to map
                solutions.append(MappingSolution(ctx.vnr, best.particle, None, None, INFEASIBLE_FITNESS))

    return solutions

def embed_batch(substrate, batch,
                        pop_size=20,
                        generations=30):

    accepted = []
    rejected = []

    vnr_list = [vnr for vnr, _ in batch]
    snapshot = copy.deepcopy(substrate)

    solutions = run_algorithm(
        vnr_list,
        snapshot,
        pop_size,
        generations
    )

    for sol in solutions:
        if sol.mapping is None:
            rejected.append(sol.vnr)
        # Final Verification & Reservation on the REAL substrate (not snapshot)
        elif verify_and_allocate(substrate, sol):
            accepted.append((sol.vnr, sol.mapping, sol.link_paths))
        else:
            rejected.append(sol.vnr)

    return accepted, rejected