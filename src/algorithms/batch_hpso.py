# src/algorithms/batch_hpso.py
import copy
import random
import math
from multiprocessing import Pool, cpu_count
from utils.graph_utils import reserve_node, reserve_path
from algorithms.fast_hpso import (
    fast_fitness,
    init_particles_hpso,
    operation_minus,
    operation_plus,
    operation_multiply,
    sa_neighbor,
    INFEASIBLE_PENALTY
)

def build_solution(particle, substrate_graph, vnr_graph):
    """
    Build complete solution with actual Dijkstra pathfinding.
    Does NOT modify substrate_graph (read-only check).
    
    Returns:
        (mapping, link_paths, total_cost) if feasible, else (None, None, INFEASIBLE_PENALTY)
    """
    from utils.graph_utils import shortest_path_with_capacity
    
    mapping = {}
    vnodes = list(vnr_graph.nodes())
    
    # 1. Node Mapping & Check CPU
    for i, v in enumerate(vnodes):
        s = particle[i]
        if substrate_graph.nodes[s]['cpu'] < vnr_graph.nodes[v]['cpu']:
            return None, None, INFEASIBLE_PENALTY
        mapping[v] = s
    
    # 2. Link Mapping (Dijkstra)
    link_paths = {}
    total_cost = 0
    
    for (u, v) in vnr_graph.edges():
        bw = vnr_graph.edges[u, v]['bw']
        path = shortest_path_with_capacity(
            substrate_graph, mapping[u], mapping[v], bw
        )
        if path is None:
            return None, None, INFEASIBLE_PENALTY
        link_paths[(u, v)] = path
        
        # Calculate cost: sum of bandwidth * hops
        total_cost += bw * (len(path) - 1)
    
    # Add node cost (CPU usage)
    for v, s in mapping.items():
        total_cost += vnr_graph.nodes[v]['cpu']
    
    return mapping, link_paths, total_cost


def evolve_population_one_generation(args):
    """
    Evolve one population for one generation.
    
    Args:
        args: tuple of (vnr_idx, vnr_graph, substrate_graph, swarm, velocities, 
                        pbest, pbest_cost, gbest, gbest_cost, iteration, 
                        total_iterations, T, cooling_rate, w_max, w_min, beta, gamma)
    
    Returns:
        (vnr_idx, new_swarm, new_velocities, new_pbest, new_pbest_cost, 
         new_gbest, new_gbest_cost, new_T)
    """
    (vnr_idx, vnr_graph, substrate_graph, swarm, velocities, 
     pbest, pbest_cost, gbest, gbest_cost, iteration, 
     total_iterations, T, cooling_rate, w_max, w_min, beta, gamma) = args
    
    num_v = len(list(vnr_graph.nodes()))
    
    # Calculate inertia weight
    alpha = w_max - (w_max - w_min) * iteration / total_iterations
    total_weight = alpha + beta + gamma
    if total_weight == 0:
        a, b, c = 0.33, 0.33, 0.33
    else:
        a = alpha / total_weight
        b = beta / total_weight
        c = gamma / total_weight
    
    new_swarm = []
    new_velocities = []
    new_pbest = copy.deepcopy(pbest)
    new_pbest_cost = pbest_cost.copy()
    new_gbest = copy.deepcopy(gbest) if gbest else None
    new_gbest_cost = gbest_cost
    
    for i in range(len(swarm)):
        # PSO UPDATE
        dp = operation_minus(pbest[i], swarm[i])
        dg = operation_minus(gbest, swarm[i]) if gbest else [0] * num_v
        
        tmp = operation_plus(a, velocities[i], b, dp)
        new_velocity = operation_plus(1 - c, tmp, c, dg)
        
        new_pos = operation_multiply(
            swarm[i], new_velocity, vnr_graph, substrate_graph
        )
        
        # Evaluate using fast fitness
        new_cost = fast_fitness(new_pos, substrate_graph, vnr_graph)
        
        # Update pBest
        if new_cost < new_pbest_cost[i]:
            new_pbest[i] = new_pos.copy()
            new_pbest_cost[i] = new_cost
            
            # Update gBest
            if new_cost < new_gbest_cost:
                new_gbest = new_pos.copy()
                new_gbest_cost = new_cost
        
        # SA STEP
        if T > 0.1:
            sa_cand = sa_neighbor(new_pos, substrate_graph, vnr_graph)
            sa_cost = fast_fitness(sa_cand, substrate_graph, vnr_graph)
            
            delta_C = sa_cost - new_cost
            accept = False
            
            if delta_C < 0:
                accept = True
            else:
                try:
                    prob = math.exp(-delta_C / T)
                    if random.random() < prob:
                        accept = True
                except (OverflowError, ZeroDivisionError):
                    accept = False
            
            if accept:
                new_pos = sa_cand
                if sa_cost < new_pbest_cost[i]:
                    new_pbest[i] = sa_cand.copy()
                    new_pbest_cost[i] = sa_cost
                    if sa_cost < new_gbest_cost:
                        new_gbest = sa_cand.copy()
                        new_gbest_cost = sa_cost
        
        new_swarm.append(new_pos)
        new_velocities.append(new_velocity)
    
    # Cool down temperature
    new_T = T * cooling_rate
    
    return (vnr_idx, new_swarm, new_velocities, new_pbest, new_pbest_cost, 
            new_gbest, new_gbest_cost, new_T)


def batch_hpso_embed(
    substrate_graph,
    vnr_list,
    particles=20,
    max_iterations=50,
    min_iterations=10,
    w_max=0.9,
    w_min=0.4,
    beta=0.3,
    gamma=0.3,
    T0=100,
    cooling_rate=0.95,
    num_processes=None
):
    """
    Batch HPSO algorithm for embedding multiple VNRs in parallel.
    
    Algorithm:
    1. Initialize populations for all VNRs
    2. Evolve all populations in parallel for at least min_iterations
    3. After min_iterations, evaluate cost/revenue ratio and commit best solutions
    4. Continue evolving uncommitted problems on updated substrate
    5. Commit solutions greedily after each generation based on cost/revenue
    
    Args:
        substrate_graph: NetworkX graph (substrate network) 
        vnr_list: List of tuples (vnr_graph, revenue)
        particles: Number of particles per population
        max_iterations: Maximum iterations per population
        min_iterations: Minimum iterations before first commit
        w_max, w_min, beta, gamma: PSO parameters
        T0: Initial temperature for SA
        cooling_rate: Cooling rate for SA
        num_processes: Number of parallel processes (None = use all CPUs)
    
    Returns:
        accepted_requests: List of tuples (vnr_graph, mapping, link_paths)
        rejected_requests: List of vnr_graphs that couldn't be embedded
    """
    if num_processes is None:
        num_processes = cpu_count()
    
    # Create a working copy of substrate graph
    substrate_work = copy.deepcopy(substrate_graph)
    
    # Initialize populations for all VNRs
    populations = {}
    for idx, (vnr_graph, revenue) in enumerate(vnr_list):
        swarm = init_particles_hpso(substrate_work, vnr_graph, particles)
        if not swarm:
            # Skip VNRs that can't initialize
            continue
        
        num_v = len(list(vnr_graph.nodes()))
        velocities = [
            [random.randint(0, 1) for _ in range(num_v)]
            for _ in range(len(swarm))
        ]
        
        pbest = copy.deepcopy(swarm)
        pbest_cost = [fast_fitness(p, substrate_work, vnr_graph) for p in swarm]
        
        gbest = None
        gbest_cost = INFEASIBLE_PENALTY
        for i in range(len(swarm)):
            if pbest_cost[i] < gbest_cost:
                gbest_cost = pbest_cost[i]
                gbest = swarm[i].copy()
        
        populations[idx] = {
            'vnr_graph': vnr_graph,
            'revenue': revenue,
            'swarm': swarm,
            'velocities': velocities,
            'pbest': pbest,
            'pbest_cost': pbest_cost,
            'gbest': gbest,
            'gbest_cost': gbest_cost,
            'T': T0,
            'committed': False,
            'mapping': None,
            'link_paths': None
        }
    
    accepted_requests = []
    rejected_requests = []
    
    # Evolution loop
    for iteration in range(max_iterations):
        # Prepare arguments for parallel evolution
        active_populations = [
            idx for idx, pop in populations.items() 
            if not pop['committed']
        ]
        
        if not active_populations:
            break
        
        evolution_args = []
        for idx in active_populations:
            pop = populations[idx]
            args = (
                idx,
                pop['vnr_graph'],
                substrate_work,
                pop['swarm'],
                pop['velocities'],
                pop['pbest'],
                pop['pbest_cost'],
                pop['gbest'],
                pop['gbest_cost'],
                iteration,
                max_iterations,
                pop['T'],
                cooling_rate,
                w_max,
                w_min,
                beta,
                gamma
            )
            evolution_args.append(args)
        
        # Parallel evolution
        with Pool(processes=min(num_processes, len(evolution_args))) as pool:
            results = pool.map(evolve_population_one_generation, evolution_args)
        
        # Update populations with results
        for result in results:
            (idx, new_swarm, new_velocities, new_pbest, new_pbest_cost,
             new_gbest, new_gbest_cost, new_T) = result
            
            populations[idx]['swarm'] = new_swarm
            populations[idx]['velocities'] = new_velocities
            populations[idx]['pbest'] = new_pbest
            populations[idx]['pbest_cost'] = new_pbest_cost
            populations[idx]['gbest'] = new_gbest
            populations[idx]['gbest_cost'] = new_gbest_cost
            populations[idx]['T'] = new_T
        
        # After min_iterations, start committing solutions
        if iteration >= min_iterations:
            # Build solutions for all active populations
            candidates = []
            for idx in active_populations:
                pop = populations[idx]
                if pop['gbest'] is None:
                    continue
                
                mapping, link_paths, total_cost = build_solution(
                    pop['gbest'],
                    substrate_work,
                    pop['vnr_graph']
                )
                
                if mapping is not None:
                    # Calculate cost/revenue ratio
                    cost_revenue_ratio = total_cost / pop['revenue'] if pop['revenue'] > 0 else float('inf')
                    candidates.append((idx, mapping, link_paths, cost_revenue_ratio, total_cost))
            
            # Sort by cost/revenue ratio (lower is better)
            candidates.sort(key=lambda x: x[3])
            
            # Commit best solution(s) greedily
            for idx, mapping, link_paths, ratio, cost in candidates:
                pop = populations[idx]
                
                # Try to commit this solution
                # Re-verify on current substrate state
                mapping_verify, link_paths_verify, _ = build_solution(
                    pop['gbest'],
                    substrate_work,
                    pop['vnr_graph']
                )
                
                if mapping_verify is not None:
                    # Commit: reserve resources on substrate
                    for v, s in mapping_verify.items():
                        reserve_node(
                            substrate_work, 
                            s, 
                            pop['vnr_graph'].nodes[v]['cpu']
                        )
                    
                    for (u, v), path in link_paths_verify.items():
                        reserve_path(
                            substrate_work,
                            path,
                            pop['vnr_graph'].edges[u, v]['bw']
                        )
                    
                    # Mark as committed
                    populations[idx]['committed'] = True
                    populations[idx]['mapping'] = mapping_verify
                    populations[idx]['link_paths'] = link_paths_verify
                    
                    accepted_requests.append((
                        pop['vnr_graph'],
                        mapping_verify,
                        link_paths_verify
                    ))
                    
                    # Only commit one solution per generation
                    break
    
    # After all iterations, collect results
    for idx, pop in populations.items():
        if not pop['committed']:
            rejected_requests.append(pop['vnr_graph'])
    
    # Update original substrate graph with committed changes
    substrate_graph.clear()
    substrate_graph.update(substrate_work)
    
    return accepted_requests, rejected_requests