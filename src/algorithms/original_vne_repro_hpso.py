# #v2
# # src/mapping_vne_hpso.py
import random
import math
import numpy as np
from src.utils.graph_utils import shortest_path_with_capacity, reserve_node, reserve_path

INFEASIBLE_PENALTY = 1e9
INFEASIBLE_FITNESS = -INFEASIBLE_PENALTY


# def compute_cost(mapping, link_paths, vnr_graph):
#     """
#     Compute the cost of an embedding solution.
#     Cost = CPU cost + bandwidth × path length
#     """
#     cpu_cost = sum(vnr_graph.nodes[n]['cpu'] for n in vnr_graph.nodes())
#     bw_cost = 0
    
#     for (u, v) in vnr_graph.edges():
#         bw = vnr_graph.edges[u, v]['bw']
#         path = link_paths.get((u, v))
#         if path is None:
#             return INFEASIBLE_PENALTY
#         bw_cost += bw * (len(path) - 1)
    
#     return cpu_cost + bw_cost


# def try_build_mapping_from_particle(particle, substrate_graph, vnr_graph):
#     """
#     Try to build a valid mapping from a particle position.
#     Checks CPU capacity and domain constraints.
    
#     Returns:
#         (mapping, link_paths, fitness)
#         fitness is negative cost, or INFEASIBLE_FITNESS if invalid
#     """
#     mapping = {}
#     vnodes = list(vnr_graph.nodes())
    
#     # Check node mapping feasibility
#     for i, v in enumerate(vnodes):
#         s = particle[i]
        
#         # Check CPU capacity
#         if substrate_graph.nodes[s]['cpu'] < vnr_graph.nodes[v]['cpu']:
#             return None, None, INFEASIBLE_FITNESS
        
#         # Check domain constraint
#         v_domain = vnr_graph.nodes[v].get('domain', None)
#         s_domain = substrate_graph.nodes[s].get('domain', None)
        
#         if v_domain is not None and s_domain is not None:
#             if v_domain != s_domain:
#                 return None, None, INFEASIBLE_FITNESS  # Domain mismatch
        
#         mapping[v] = s
    
#     # Check link mapping feasibility
#     link_paths = {}
#     for u, v in vnr_graph.edges():
#         bw_req = vnr_graph.edges[u, v]['bw']
#         s_src = mapping[u]
#         s_dst = mapping[v]
        
#         path = shortest_path_with_capacity(substrate_graph, s_src, s_dst, bw_req)
#         if path is None:
#             return None, None, INFEASIBLE_FITNESS
        
#         link_paths[(u, v)] = path
    
#     # Calculate fitness (negative cost for maximization)
#     cost = compute_cost(mapping, link_paths, vnr_graph)
#     fitness = -cost
    
#     return mapping, link_paths, fitness


# def operation_minus(Xi, Xj):
#     """
#     Velocity operation: returns 1 if positions match, 0 otherwise.
#     """
#     return [1 if Xi[k] == Xj[k] else 0 for k in range(len(Xi))]


# def operation_plus(p, Vi, q, Vj):
#     """
#     Velocity operation: combines two velocities with weights p and q.
#     """
#     result = []
#     for k in range(len(Vi)):
#         if Vi[k] == Vj[k]:
#             result.append(Vi[k])
#         else:
#             if random.random() < p:
#                 result.append(Vi[k])
#             else:
#                 result.append(Vj[k])
#     return result


# def operation_multiply(Xi, Vj, candidate_lists, vnodes):
#     """
#     Apply velocity to position: update position based on velocity.
#     If velocity[k] == 0, change to a different candidate from the same domain.
#     """
#     result = Xi.copy()
    
#     for k in range(len(Vj)):
#         if Vj[k] == 0:
#             v = vnodes[k]
#             candidates = candidate_lists[v]
            
#             # Try to find a different candidate
#             candidates_diff = [s for s in candidates if s != Xi[k]]
#             if candidates_diff:
#                 result[k] = random.choice(candidates_diff)
#             else:
#                 # No different candidate, keep the same
#                 result[k] = random.choice(candidates)
    
#     return result


# def compute_inertia_weight(current_iter, max_iter, w_max=0.9, w_min=0.4):
#     """
#     Compute linearly decreasing inertia weight.
#     """
#     alpha = w_max - (w_max - w_min) * (current_iter / max_iter)
#     return alpha


# def create_candidate_lists(substrate_graph, vnr_graph):
#     """
#     Create candidate lists for each virtual node with domain constraints.
#     Each virtual node can only be mapped to substrate nodes in the same domain.
#     """
#     vnodes = list(vnr_graph.nodes())
#     sub_nodes = list(substrate_graph.nodes())
    
#     # Sort virtual nodes by CPU requirement (descending)
#     vnodes_sorted = sorted(vnodes, 
#                            key=lambda v: vnr_graph.nodes[v]['cpu'], 
#                            reverse=True)
    
#     # Sort substrate nodes by CPU capacity (descending)
#     sub_nodes_sorted = sorted(sub_nodes,
#                               key=lambda s: substrate_graph.nodes[s]['cpu'],
#                               reverse=True)
    
#     # Get minimum CPU requirement
#     min_cpu_req = min(vnr_graph.nodes[v]['cpu'] for v in vnodes)
    
#     # Pre-filter substrate nodes by minimum CPU
#     valid_sub_nodes = [s for s in sub_nodes_sorted 
#                        if substrate_graph.nodes[s]['cpu'] >= min_cpu_req]
    
#     if not valid_sub_nodes:
#         valid_sub_nodes = sub_nodes_sorted
    
#     # Create candidate lists for each virtual node
#     candidate_lists = {}
    
#     for v in vnodes:
#         v_cpu = vnr_graph.nodes[v]['cpu']
#         v_domain = vnr_graph.nodes[v].get('domain', None)
        
#         candidates = []
        
#         for s in valid_sub_nodes:
#             # Check CPU capacity
#             if substrate_graph.nodes[s]['cpu'] < v_cpu:
#                 continue
            
#             # Check domain constraint
#             s_domain = substrate_graph.nodes[s].get('domain', None)
            
#             # If both have domains, they must match
#             if v_domain is not None and s_domain is not None:
#                 if v_domain != s_domain:
#                     continue  # Domain mismatch
            
#             candidates.append(s)
        
#         # Fallback if no candidates found
#         if not candidates:
#             # Try relaxing CPU constraint within the same domain
#             if v_domain is not None:
#                 candidates = [s for s in sub_nodes_sorted
#                              if substrate_graph.nodes[s].get('domain') == v_domain]
            
#             # Last resort: use all valid_sub_nodes
#             if not candidates:
#                 candidates = valid_sub_nodes
        
#         candidate_lists[v] = candidates
    
#     return candidate_lists


# def generate_neighbor_particle(particle, candidate_lists, vnodes):
#     """
#     Generate a neighbor particle by randomly changing one position.
#     The new position is chosen from domain-compatible candidates.
#     """
#     neighbor = particle.copy()
#     k = random.randint(0, len(particle) - 1)
#     v = vnodes[k]
#     candidates = candidate_lists[v]
    
#     # Try to find a different candidate
#     candidates_diff = [s for s in candidates if s != particle[k]]
#     if candidates_diff:
#         neighbor[k] = random.choice(candidates_diff)
#     else:
#         # No different candidate, keep the same or choose from all candidates
#         neighbor[k] = random.choice(candidates)
    
#     return neighbor


# def hpso_embed(substrate_graph, vnr_graph, particles=30, iterations=50,
#                beta=0.3, gamma=0.4, w_max=0.9, w_min=0.4, cooling_rate=0.9):
#     """
#     Hybrid PSO (PSO + Simulated Annealing) with domain-aware node mapping.
    
#     Args:
#         substrate_graph: Substrate network
#         vnr_graph: VNR to embed
#         particles: Number of particles in swarm
#         iterations: Number of iterations
#         beta: Weight for personal best
#         gamma: Weight for global best
#         w_max: Maximum inertia weight
#         w_min: Minimum inertia weight
#         cooling_rate: Temperature cooling rate for SA
    
#     Returns:
#         (mapping, link_paths) if successful, None if failed
#     """
#     sub_nodes = list(substrate_graph.nodes())
#     vnodes = list(vnr_graph.nodes())
#     num_v = len(vnodes)
    
#     # Create domain-aware candidate lists
#     candidate_lists = create_candidate_lists(substrate_graph, vnr_graph)
    
#     # Check if any virtual node has no candidates (early termination)
#     for v in vnodes:
#         if not candidate_lists[v]:
#             return None
    
#     # Initialize swarm
#     swarm = []
#     velocities = []
#     fitness_values = []
    
#     for _ in range(particles):
#         particle = []
#         for v in vnodes:
#             candidates = candidate_lists[v]
#             particle.append(random.choice(candidates))
        
#         _, _, fitness = try_build_mapping_from_particle(
#             particle, substrate_graph, vnr_graph
#         )
        
#         velocity = [random.randint(0, 1) for _ in range(num_v)]
        
#         swarm.append(particle)
#         velocities.append(velocity)
#         fitness_values.append(fitness)
    
#     # Initialize personal best and global best
#     pbest = [particle.copy() for particle in swarm]
#     pbest_fitness = fitness_values.copy()
    
#     best_idx = max(range(particles), key=lambda i: fitness_values[i])
#     gbest = swarm[best_idx].copy()
#     gbest_fitness = fitness_values[best_idx]
    
#     # Initialize temperature for simulated annealing
#     fmax = max(fitness_values) if max(fitness_values) != INFEASIBLE_FITNESS else 0
#     fmin = min(fitness_values) if min(fitness_values) != INFEASIBLE_FITNESS else 0
#     temperature = abs(fmax - fmin) if fmax != fmin else 100.0
    
#     # HPSO main loop
#     for iteration in range(iterations):
#         alpha = compute_inertia_weight(iteration, iterations, w_max, w_min)
        
#         # Normalize weights
#         total = alpha + beta + gamma
#         alpha_norm = alpha / total
#         beta_norm = beta / total
#         gamma_norm = gamma / total
        
#         for i in range(particles):
#             # PSO velocity and position update
#             diff_pbest = operation_minus(pbest[i], swarm[i])
#             diff_gbest = operation_minus(gbest, swarm[i])
            
#             temp_v = operation_plus(alpha_norm, velocities[i], beta_norm, diff_pbest)
#             new_velocity = operation_plus(1 - gamma_norm, temp_v, gamma_norm, diff_gbest)
#             velocities[i] = new_velocity
            
#             # Update position (respects domain constraints)
#             new_particle = operation_multiply(swarm[i], new_velocity, candidate_lists, vnodes)
            
#             _, _, new_fitness = try_build_mapping_from_particle(
#                 new_particle, substrate_graph, vnr_graph
#             )
            
#             # Update personal best
#             if new_fitness > pbest_fitness[i]:
#                 pbest[i] = new_particle.copy()
#                 pbest_fitness[i] = new_fitness
            
#             # Update global best
#             if new_fitness > gbest_fitness:
#                 gbest = new_particle.copy()
#                 gbest_fitness = new_fitness
            
#             # Simulated Annealing: Generate neighbor
#             neighbor_particle = generate_neighbor_particle(
#                 new_particle, candidate_lists, vnodes
#             )
            
#             _, _, neighbor_fitness = try_build_mapping_from_particle(
#                 neighbor_particle, substrate_graph, vnr_graph
#             )
            
#             delta_fitness = neighbor_fitness - new_fitness
            
#             # SA acceptance criterion
#             if delta_fitness > 0:
#                 # Accept better solution
#                 swarm[i] = neighbor_particle
#                 fitness_values[i] = neighbor_fitness
                
#                 if neighbor_fitness > pbest_fitness[i]:
#                     pbest[i] = neighbor_particle.copy()
#                     pbest_fitness[i] = neighbor_fitness
                
#                 if neighbor_fitness > gbest_fitness:
#                     gbest = neighbor_particle.copy()
#                     gbest_fitness = neighbor_fitness
#             else:
#                 # Accept worse solution with probability
#                 r = random.random()
#                 acceptance_prob = min(1.0, math.exp(delta_fitness / temperature)) if temperature > 0 else 0
                
#                 if r < acceptance_prob:
#                     swarm[i] = neighbor_particle
#                     fitness_values[i] = neighbor_fitness
                    
#                     if neighbor_fitness > pbest_fitness[i]:
#                         pbest[i] = neighbor_particle.copy()
#                         pbest_fitness[i] = neighbor_fitness
                    
#                     if neighbor_fitness > gbest_fitness:
#                         gbest = neighbor_particle.copy()
#                         gbest_fitness = neighbor_fitness
#                 else:
#                     # Reject neighbor, keep new_particle
#                     swarm[i] = new_particle
#                     fitness_values[i] = new_fitness
        
#         # Cool down temperature
#         temperature = temperature * cooling_rate
    
#     # Return best solution found
#     if gbest_fitness == INFEASIBLE_FITNESS:
#         return None
    
#     # Build final mapping from best particle
#     mapping, link_paths, _ = try_build_mapping_from_particle(
#         gbest, substrate_graph, vnr_graph
#     )
    
#     if mapping is None:
#         return None
    
#     # Reserve resources
#     for i, v in enumerate(vnodes):
#         s = gbest[i]
#         reserve_node(substrate_graph, s, vnr_graph.nodes[v]['cpu'])
    
#     for (u, v), path in link_paths.items():
#         reserve_path(substrate_graph, path, vnr_graph.edges[u, v]['bw'])
    
#     return mapping, link_paths

# ============================================================
# v3
# # src/mapping_vne_hpso.py
# import random
# import math
# import numpy as np
# from utils import shortest_path_with_capacity, reserve_node, reserve_path

# INFEASIBLE_PENALTY = 1e9
# INFEASIBLE_FITNESS = -INFEASIBLE_PENALTY
# def fast_fitness(particle, substrate_graph, vnr_graph):
#     """
#     Fast proxy fitness (minimize):
#     - CPU feasibility
#     - Estimated link cost by BW sum
#     """
#     mapping = {}
#     vnodes = list(vnr_graph.nodes())

#     for i, v in enumerate(vnodes):
#         s = particle[i]
#         if substrate_graph.nodes[s]['cpu'] < vnr_graph.nodes[v]['cpu']:
#             return INFEASIBLE_PENALTY
#         mapping[v] = s

#     est_cost = 0
#     for (u, v) in vnr_graph.edges():
#         if mapping[u] != mapping[v]:
#             est_cost += vnr_graph.edges[u, v]['bw']

#     return est_cost

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

# def sa_neighbor(particle, candidate_lists, vnodes):
#     neighbor = particle.copy()
#     k = random.randrange(len(particle))
#     v = vnodes[k]

#     alt = [s for s in candidate_lists[v] if s != particle[k]]
#     if alt:
#         neighbor[k] = random.choice(alt)
#     return neighbor

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

# def hpso_embed(
#     substrate_graph,
#     vnr_graph,
#     particles=30,
#     iterations=50,
#     beta=0.3,
#     gamma=0.4,
#     w_max=0.9,
#     w_min=0.4,
#     cooling_rate=0.9,
#     stall_limit=5,
#     top_k=5
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

#     # SA temperature
#     temperature = 100.0

#     stall = 0
#     prev_best = INFEASIBLE_PENALTY

#     # ----- HPSO loop -----
#     for it in range(iterations):
#         alpha = w_max - (w_max - w_min) * it / iterations
#         total = alpha + beta + gamma
#         a, b, c = alpha / total, beta / total, gamma / total

#         for i in range(particles):
#             cost = fast_fitness(swarm[i], substrate_graph, vnr_graph)

#             # update pbest
#             if cost < pbest_cost[i]:
#                 pbest_cost[i] = cost
#                 pbest[i] = swarm[i].copy()

#             # update gbest
#             if cost < gbest_cost:
#                 gbest_cost = cost
#                 gbest = swarm[i].copy()

#         # early stop
#         if abs(prev_best - gbest_cost) < 1e-6:
#             stall += 1
#             if stall >= stall_limit:
#                 break
#         else:
#             stall = 0
#         prev_best = gbest_cost

#         # PSO update
#         for i in range(particles):
#             dp = operation_minus(pbest[i], swarm[i])
#             dg = operation_minus(gbest, swarm[i])

#             tmp = operation_plus(a, velocities[i], b, dp)
#             velocities[i] = operation_plus(1 - c, tmp, c, dg)

#             new_particle = operation_multiply(
#                 swarm[i], velocities[i], vnr_graph, candidate_lists
#             )

#             new_cost = fast_fitness(new_particle, substrate_graph, vnr_graph)
#             delta = new_cost - fast_fitness(swarm[i], substrate_graph, vnr_graph)

#             # ---- SA acceptance ----
#             if delta < 0:
#                 swarm[i] = new_particle
#             else:
#                 prob = math.exp(-delta / temperature) if temperature > 0 else 0
#                 if random.random() < prob:
#                     swarm[i] = new_particle

#         temperature *= cooling_rate

#     # ----- TOP-K full embedding -----
#     ranked = sorted(
#         zip(pbest_cost, pbest),
#         key=lambda x: x[0]
#     )[:top_k]

#     path_cache = {}

#     for _, particle in ranked:
#         mapping, link_paths = build_full_solution(
#             particle, substrate_graph, vnr_graph, path_cache
#         )
#         if mapping is not None:
#             for v, s in mapping.items():
#                 reserve_node(substrate_graph, s, vnr_graph.nodes[v]['cpu'])
#             for (u, v), path in link_paths.items():
#                 reserve_path(substrate_graph, path, vnr_graph.edges[u, v]['bw'])
#             return mapping, link_paths

#     return None

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
# def hpso_embed(
#     substrate_graph,
#     vnr_graph,
#     particles=20,
#     iterations=30,
#     w_max=0.9,
#     w_min=0.4,
#     beta=0.3,
#     gamma=0.3,
#     T0=100,
#     cooling_rate=0.95
# ):
#     vnodes = list(vnr_graph.nodes())
#     num_v = len(vnodes)

#     swarm = init_particles_hpso(
#         substrate_graph, vnr_graph, particles
#     )

#     if not swarm:
#         return None

#     velocities = [
#         [random.randint(0, 1) for _ in range(num_v)]
#         for _ in range(len(swarm))
#     ]

#     pbest = copy.deepcopy(swarm)
#     pbest_cost = [
#         hpso_fitness(p, substrate_graph, vnr_graph) for p in swarm
#     ]

#     gbest_idx = min(range(len(pbest_cost)), key=lambda i: pbest_cost[i])
#     gbest = pbest[gbest_idx].copy()
#     gbest_cost = pbest_cost[gbest_idx]

#     T = T0

#     # ================= MAIN LOOP =================
#     for it in range(iterations):
#         alpha = w_max - (w_max - w_min) * it / iterations
#         total = alpha + beta + gamma
#         a, b, c = alpha / total, beta / total, gamma / total

#         for i in range(len(swarm)):
#             # ----- PSO update -----
#             dp = operation_minus(pbest[i], swarm[i])
#             dg = operation_minus(gbest, swarm[i])

#             tmp = operation_plus(a, velocities[i], b, dp)
#             velocities[i] = operation_plus(1 - c, tmp, c, dg)

#             candidate = operation_multiply(
#                 swarm[i], velocities[i], vnr_graph, substrate_graph
#             )

#             # ----- SA neighbor -----
#             sa_cand = sa_neighbor(
#                 swarm[i], substrate_graph, vnr_graph
#             )

#             for new_particle in [candidate, sa_cand]:
#                 old_cost = hpso_fitness(
#                     swarm[i], substrate_graph, vnr_graph
#                 )
#                 new_cost = hpso_fitness(
#                     new_particle, substrate_graph, vnr_graph
#                 )

#                 delta = new_cost - old_cost
#                 if delta < 0 or random.random() < math.exp(-delta / T):
#                     swarm[i] = new_particle

#             # ----- update pbest -----
#             cost = hpso_fitness(swarm[i], substrate_graph, vnr_graph)
#             if cost < pbest_cost[i]:
#                 pbest[i] = swarm[i].copy()
#                 pbest_cost[i] = cost

#                 if cost < gbest_cost:
#                     gbest = swarm[i].copy()
#                     gbest_cost = cost

#         T *= cooling_rate

#     # ================= FINAL EMBEDDING =================
#     mapping = {}
#     for i, v in enumerate(vnodes):
#         mapping[v] = gbest[i]
#         reserve_node(
#             substrate_graph,
#             gbest[i],
#             vnr_graph.nodes[v]['cpu']
#         )

#     link_paths = {}
#     for (u, v) in vnr_graph.edges():
#         bw = vnr_graph.edges[u, v]['bw']
#         path = shortest_path_with_capacity(
#             substrate_graph, mapping[u], mapping[v], bw
#         )
#         if path is None:
#             return None
#         reserve_path(substrate_graph, path, bw)
#         link_paths[(u, v)] = path

#     return mapping, link_paths
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
                return build_embedding(
                    swarm[i], substrate_graph, vnr_graph
                )

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
