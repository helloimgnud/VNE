# src/algorithms/hpso_priority.py
import random
import math
import copy
import numpy as np
from src.utils.graph_utils import shortest_path_with_capacity, reserve_node, reserve_path

INFEASIBLE_PENALTY = 1e9

# =========================================================
# 1. Pre-processing: Build Domain & Available Masks
# =========================================================

def build_domain_masks(substrate_graph):
    """
    Xây dựng k vector bitmask cho k domains.
    domain_masks[i] = vector m chiều, 1 tại node thuộc domain i, 0 tại node khác.
    
    Returns:
        domain_masks: dict {domain_id: np.array of shape (m,)}
        node_to_idx: dict {node_id: index in vector}
    """
    nodes = list(substrate_graph.nodes())
    m = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Lấy tất cả các domain
    domains = set()
    for node in nodes:
        if 'domain' in substrate_graph.nodes[node]:
            domains.add(substrate_graph.nodes[node]['domain'])
    
    domain_masks = {}
    for domain in domains:
        mask = np.zeros(m)
        for node in nodes:
            if substrate_graph.nodes[node].get('domain') == domain:
                mask[node_to_idx[node]] = 1
        domain_masks[domain] = mask
    
    return domain_masks, node_to_idx

def build_available_masks(substrate_graph, vnr_graph, node_to_idx):
    """
    Xây dựng n vector available cho n nodes ảo.
    available_masks[i] = vector m chiều, 1 tại node thực có CPU đủ, 0 tại node không đủ.
    
    Returns:
        available_masks: dict {vnode: np.array of shape (m,)}
    """
    m = len(node_to_idx)
    available_masks = {}
    
    for vnode in vnr_graph.nodes():
        v_cpu = vnr_graph.nodes[vnode]['cpu']
        mask = np.zeros(m)
        
        for snode, idx in node_to_idx.items():
            if substrate_graph.nodes[snode]['cpu'] >= v_cpu:
                mask[idx] = 1
        
        available_masks[vnode] = mask
    
    return available_masks

# =========================================================
# 2. Decode Priority Vector to Mapping
# =========================================================

def decode_priority_vector(priority_vec, substrate_graph, vnr_graph, 
                          domain_masks, available_masks, node_to_idx):
    """
    Giải mã priority vector thành mapping cụ thể.
    
    Args:
        priority_vec: np.array of shape (m,) với giá trị trong [0,1]
        
    Returns:
        mapping: dict {vnode: snode} hoặc None nếu không khả thi
    """
    m = len(node_to_idx)
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    # Sắp xếp vnodes theo CPU demand giảm dần
    vnodes_sorted = sorted(
        vnr_graph.nodes(),
        key=lambda v: vnr_graph.nodes[v]['cpu'],
        reverse=True
    )
    
    mapping = {}
    local = np.ones(m)  # Vector ghi nhận vị trí còn khả dụng
    
    for vnode in vnodes_sorted:
        # Lấy domain của vnode (nếu có)
        v_domain = vnr_graph.nodes[vnode].get('domain')
        
        # Xây dựng vector priority cho vnode này
        if v_domain is not None and v_domain in domain_masks:
            domain_mask = domain_masks[v_domain]
        else:
            domain_mask = np.ones(m)  # Không có ràng buộc domain
        
        # Nhân element-wise: priority_vec * domain * available * local
        priority = priority_vec * domain_mask * available_masks[vnode] * local
        
        # Tìm vị trí có priority cao nhất
        max_idx = np.argmax(priority)
        
        # Kiểm tra khả thi
        if priority[max_idx] == 0:
            return None  # Không tìm được node khả thi
        
        # Ánh xạ vnode vào substrate node
        snode = idx_to_node[max_idx]
        mapping[vnode] = snode
        
        # Cập nhật local: đánh dấu vị trí đã dùng
        local[max_idx] = 0
    
    return mapping



# =========================================================
# 3. Fast Fitness with Priority Vector
# =========================================================

def fast_fitness_priority(priority_vec, substrate_graph, vnr_graph,
                         domain_masks, available_masks, node_to_idx):
    """
    Tính fitness nhanh cho priority vector.
    """
    mapping = decode_priority_vector(
        priority_vec, substrate_graph, vnr_graph,
        domain_masks, available_masks, node_to_idx
    )
    
    if mapping is None:
        return INFEASIBLE_PENALTY
    
    # Ước lượng chi phí link (giống như cũ)
    est_link_cost = 0
    for (u, v) in vnr_graph.edges():
        if mapping[u] != mapping[v]:
            est_link_cost += vnr_graph.edges[u, v]['bw']
    
    return est_link_cost

def build_and_reserve_priority(priority_vec, substrate_graph, vnr_graph,
                               domain_masks, available_masks, node_to_idx):
    """
    Giải mã và reserve resources (chạy Dijkstra thực).
    """
    mapping = decode_priority_vector(
        priority_vec, substrate_graph, vnr_graph,
        domain_masks, available_masks, node_to_idx
    )
    
    if mapping is None:
        return None, None
    
    # Kiểm tra CPU constraints (double check)
    for vnode, snode in mapping.items():
        if substrate_graph.nodes[snode]['cpu'] < vnr_graph.nodes[vnode]['cpu']:
            return None, None
    
    # Link Mapping với Dijkstra
    link_paths = {}
    for (u, v) in vnr_graph.edges():
        bw = vnr_graph.edges[u, v]['bw']
        path = shortest_path_with_capacity(
            substrate_graph, mapping[u], mapping[v], bw
        )
        if path is None:
            return None, None
        link_paths[(u, v)] = path
    
    # Reserve Resources
    for vnode, snode in mapping.items():
        reserve_node(substrate_graph, snode, vnr_graph.nodes[vnode]['cpu'])
    for (u, v), path in link_paths.items():
        reserve_path(substrate_graph, path, vnr_graph.edges[u, v]['bw'])
    
    return mapping, link_paths

# =========================================================
# 4. Particle Initialization (Priority Vector)
# =========================================================

def init_particles_priority(m, particles):
    """
    Khởi tạo swarm với priority vectors ngẫu nhiên trong [0,1].
    
    Args:
        m: số lượng substrate nodes
        particles: số lượng particles
        
    Returns:
        swarm: list of np.array, shape (particles, m)
    """
    swarm = []
    for _ in range(particles):
        # Random uniform [0, 1]
        priority_vec = np.random.uniform(0, 1, m)
        swarm.append(priority_vec)
    
    return swarm

# =========================================================
# 5. PSO Operators for Continuous Space
# =========================================================

def update_velocity_priority(velocity, position, pbest, gbest, 
                            w, c1, c2):
    """
    Cập nhật velocity theo công thức PSO chuẩn cho không gian liên tục.
    
    v_new = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
    """
    r1 = np.random.uniform(0, 1, len(velocity))
    r2 = np.random.uniform(0, 1, len(velocity))
    
    v_new = (w * velocity + 
             c1 * r1 * (pbest - position) + 
             c2 * r2 * (gbest - position))
    
    return v_new

def update_position_priority(position, velocity):
    """
    Cập nhật position và clamp về [0,1].
    
    x_new = x + v
    """
    x_new = position + velocity
    
    # Clamp về [0, 1]
    x_new = np.clip(x_new, 0, 1)
    
    return x_new

# =========================================================
# 6. Simulated Annealing Neighbor (Priority Vector)
# =========================================================

def sa_neighbor_priority(priority_vec, mutation_rate=0.1):
    """
    Tạo neighbor bằng cách thêm Gaussian noise vào một số chiều ngẫu nhiên.
    
    Args:
        priority_vec: np.array
        mutation_rate: tỷ lệ số chiều bị mutate
        
    Returns:
        neighbor: np.array
    """
    neighbor = priority_vec.copy()
    m = len(priority_vec)
    
    # Chọn ngẫu nhiên các chiều để mutate
    num_mutations = max(1, int(m * mutation_rate))
    indices = np.random.choice(m, num_mutations, replace=False)
    
    # Thêm Gaussian noise
    noise = np.random.normal(0, 0.1, num_mutations)
    neighbor[indices] += noise
    
    # Clamp về [0, 1]
    neighbor = np.clip(neighbor, 0, 1)
    
    return neighbor

# =========================================================
# 7. HPSO Main Algorithm (Priority Vector Version)
# =========================================================

def hpso_embed(
    substrate_graph,
    vnr_graph,
    particles=20,
    iterations=30,
    w_max=0.9,
    w_min=0.4,
    c1=2.0,  # cognitive parameter
    c2=2.0,  # social parameter
    T0=100,
    cooling_rate=0.95,
    mutation_rate=0.3
):
    """
    HPSO với Priority Vector encoding.
    """
    # Pre-processing
    domain_masks, node_to_idx = build_domain_masks(substrate_graph)
    available_masks = build_available_masks(substrate_graph, vnr_graph, node_to_idx)
    m = len(node_to_idx)
    
    # Initialize swarm
    swarm = init_particles_priority(m, particles)
    velocities = [np.random.uniform(-0.1, 0.1, m) for _ in range(particles)]
    
    # Initialize pbest and gbest
    pbest = copy.deepcopy(swarm)
    pbest_cost = [
        fast_fitness_priority(p, substrate_graph, vnr_graph,
                            domain_masks, available_masks, node_to_idx)
        for p in swarm
    ]

    gbest_idx = int(np.argmin(pbest_cost))
    gbest = pbest[gbest_idx].copy()
    gbest_cost = pbest_cost[gbest_idx]
    
    # gbest = None
    # gbest_cost = INFEASIBLE_PENALTY
    
    # for i in range(particles):
    #     if pbest_cost[i] < gbest_cost:
    #         gbest_cost = pbest_cost[i]
    #         gbest = swarm[i].copy()
    
    T = T0
    
    # ================= MAIN LOOP =================
    for it in range(iterations):
        # Linearly decreasing inertia weight
        w = w_max - (w_max - w_min) * it / iterations
        
        for i in range(particles):
            # [OPTIMIZATION] Early Stop - kiểm tra feasible solution
            mapping, link_paths = build_and_reserve_priority(
                swarm[i], substrate_graph, vnr_graph,
                domain_masks, available_masks, node_to_idx
            )
            if mapping is not None:
                return mapping, link_paths
            
            # --- 1. PSO UPDATE ---
            velocities[i] = update_velocity_priority(
                velocities[i], swarm[i], pbest[i], gbest,
                w, c1, c2
            )
            
            new_pos = update_position_priority(swarm[i], velocities[i])
            
            # Đánh giá fitness
            new_cost = fast_fitness_priority(
                new_pos, substrate_graph, vnr_graph,
                domain_masks, available_masks, node_to_idx
            )
            
            # Update pbest và gbest
            if new_cost < pbest_cost[i]:
                pbest[i] = new_pos.copy()
                pbest_cost[i] = new_cost
                if new_cost < gbest_cost:
                    gbest = new_pos.copy()
                    gbest_cost = new_cost
            
            swarm[i] = new_pos
            
            # --- 2. SA STEP ---
            if T > 0.1:
                sa_cand = sa_neighbor_priority(swarm[i], mutation_rate)
                sa_cost = fast_fitness_priority(
                    sa_cand, substrate_graph, vnr_graph,
                    domain_masks, available_masks, node_to_idx
                )
                
                delta_C = sa_cost - new_cost
                
                accept = False
                if delta_C < 0:
                    accept = True
                else:
                    try:
                        prob = math.exp(-delta_C / T)
                    except OverflowError:
                        prob = 0
                    if random.random() < prob:
                        accept = True
                
                if accept:
                    swarm[i] = sa_cand
                    if sa_cost < pbest_cost[i]:
                        pbest[i] = sa_cand.copy()
                        pbest_cost[i] = sa_cost
                        if sa_cost < gbest_cost:
                            gbest = sa_cand.copy()
                            gbest_cost = sa_cost
        
        # Cooling
        T *= cooling_rate
    
    # Thử build gbest cuối cùng
    if gbest is not None:
        mapping, link_paths = build_and_reserve_priority(
            gbest, substrate_graph, vnr_graph,
            domain_masks, available_masks, node_to_idx
        )
        if mapping is not None:
            return mapping, link_paths
    
    return None