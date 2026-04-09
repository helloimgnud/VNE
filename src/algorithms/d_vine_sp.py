# src/algorithms/d_vine_sp.py
from ortools.linear_solver import pywraplp
import networkx as nx
from src.utils.graph_utils import shortest_path_with_capacity, reserve_node, reserve_path, release_node, release_path


def create_candidates_dict_with_domain(substrate_graph, vnr_graph):
    """
    Create candidates dictionary with domain constraints.
    
    Returns:
        candidates_dict: {vnode_id: [list of valid substrate node ids]}
    """
    candidates_dict = {}
    
    for v in vnr_graph.nodes():
        v_cpu = vnr_graph.nodes[v]['cpu']
        v_domain = vnr_graph.nodes[v].get('domain', None)
        
        candidates = []
        for s in substrate_graph.nodes():
            # Check CPU capacity
            if substrate_graph.nodes[s]['cpu'] < v_cpu:
                continue
            
            # Check domain constraint
            s_domain = substrate_graph.nodes[s].get('domain', None)
            if v_domain is not None and s_domain is not None:
                if v_domain != s_domain:
                    continue
            
            candidates.append(s)
        
        candidates_dict[v] = candidates
    
    return candidates_dict


def build_node_mapping(vnr_node_to_idx, substrate_node_to_idx):
    """Create mappings between node IDs and LP variable indices."""
    idx_to_vnr = {v: k for k, v in vnr_node_to_idx.items()}
    idx_to_substrate = {v: k for k, v in substrate_node_to_idx.items()}
    return idx_to_vnr, idx_to_substrate


def solve_d_vine_lp_ortools(substrate_graph, vnr_graph, candidates_dict, 
                            max_time_seconds=10, meta_bw=9999):
    """
    Solve D-ViNE LP relaxation using Google OR-Tools.
    Adapted from GeminiLight's implementation with domain constraints.
    
    Args:
        substrate_graph: NetworkX substrate graph
        vnr_graph: NetworkX VNR graph
        candidates_dict: {vnode: [list of candidate substrate nodes]}
        max_time_seconds: Max solver time
        meta_bw: Bandwidth for meta-edges
    
    Returns:
        v_p_value_dict: {vnode: {snode: probability}} or None if infeasible
    """
    # Create index mappings
    substrate_nodes = sorted(substrate_graph.nodes())
    vnr_nodes = sorted(vnr_graph.nodes())
    
    num_p_nodes = len(substrate_nodes)
    num_v_nodes = len(vnr_nodes)
    
    # Map node IDs to indices
    substrate_node_to_idx = {node: idx for idx, node in enumerate(substrate_nodes)}
    vnr_node_to_idx = {node: idx for idx, node in enumerate(vnr_nodes)}
    
    # Augmented graph indices
    p_node_list = list(range(num_p_nodes))
    m_node_list = list(range(num_p_nodes, num_p_nodes + num_v_nodes))
    a_node_list = list(range(num_p_nodes + num_v_nodes))
    
    # Build resource dictionaries
    def get_node_resource(n_id, attr):
        if n_id < num_p_nodes:
            # Physical node
            node = substrate_nodes[n_id]
            return substrate_graph.nodes[node][attr]
        else:
            # Virtual node (meta-node)
            node = vnr_nodes[n_id - num_p_nodes]
            return vnr_graph.nodes[node][attr]
    
    def get_edge_resource(u, v, attr):
        # Both physical nodes
        if u < num_p_nodes and v < num_p_nodes:
            u_node = substrate_nodes[u]
            v_node = substrate_nodes[v]
            if substrate_graph.has_edge(u_node, v_node):
                return substrate_graph.edges[u_node, v_node][attr]
            return 0
        # Meta-edge (virtual to physical)
        elif u >= num_p_nodes and v < num_p_nodes:
            v_node = vnr_nodes[u - num_p_nodes]
            s_node = substrate_nodes[v]
            return meta_bw if s_node in candidates_dict[v_node] else 0
        elif u < num_p_nodes and v >= num_p_nodes:
            v_node = vnr_nodes[v - num_p_nodes]
            s_node = substrate_nodes[u]
            return meta_bw if s_node in candidates_dict[v_node] else 0
        else:
            # Both virtual
            return 0
    
    # Create solver
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return None
    
    # Variables: x[(u,v)] for all node pairs in augmented graph
    x = {}
    for u in a_node_list:
        for v in a_node_list:
            x[(u, v)] = solver.NumVar(0, 1, f'x_{u}_{v}')
    
    # Variables: f[(u, v, i)] for flow of virtual link i on edge (u,v)
    f = {}
    vnr_edges = list(vnr_graph.edges())
    for u in a_node_list:
        for v in a_node_list:
            for i, (vu, vv) in enumerate(vnr_edges):
                f[(u, v, i)] = solver.NumVar(0, meta_bw, f'f_{u}_{v}_{i}')
    
    # Objective: Minimize resource cost
    objective = solver.Objective()
    
    # Bandwidth cost: Σ (1/BW_avail) × Σ_i f^i_uv
    for u in p_node_list:
        for v in p_node_list:
            bw_avail = get_edge_resource(u, v, 'bw')
            if bw_avail > 0:
                coeff = 1.0 / (bw_avail + 1e-6)
                for i in range(len(vnr_edges)):
                    objective.SetCoefficient(f[(u, v, i)], coeff)
    
    # Node cost: Σ (1/CPU_avail) × Σ_m x_mw × CPU_req
    for w in p_node_list:
        cpu_avail = get_node_resource(w, 'cpu')
        coeff_base = 1.0 / (cpu_avail + 1e-6)
        for m in m_node_list:
            cpu_req = get_node_resource(m, 'cpu')
            objective.SetCoefficient(x[(m, w)], coeff_base * cpu_req)
    
    objective.SetMinimization()
    
    # Constraint 1: CPU capacity
    for m in m_node_list:
        for w in p_node_list:
            cpu_req = get_node_resource(m, 'cpu')
            cpu_avail = get_node_resource(w, 'cpu')
            solver.Add(x[(m, w)] * cpu_req <= cpu_avail)
    
    # Constraint 2: Bandwidth capacity
    for u in a_node_list:
        for v in a_node_list:
            bw_avail = get_edge_resource(u, v, 'bw')
            sum_flow = solver.Sum([f[(u, v, i)] + f[(v, u, i)] for i in range(len(vnr_edges))])
            solver.Add(sum_flow <= bw_avail * x[(u, v)])
    
    # Constraint 3: Flow conservation
    for i, (v_src, v_dst) in enumerate(vnr_edges):
        bw_req = vnr_graph.edges[v_src, v_dst]['bw']
        src_idx = m_node_list[vnr_node_to_idx[v_src]]
        dst_idx = m_node_list[vnr_node_to_idx[v_dst]]
        
        for n_id in a_node_list:
            outflow = solver.Sum([f[(n_id, w, i)] for w in a_node_list])
            inflow = solver.Sum([f[(w, n_id, i)] for w in a_node_list])
            
            if n_id == src_idx:
                # Source: outflow - inflow = bandwidth
                solver.Add(outflow - inflow == bw_req)
            elif n_id == dst_idx:
                # Sink: outflow - inflow = -bandwidth
                solver.Add(outflow - inflow == -bw_req)
            else:
                # Intermediate: outflow = inflow
                solver.Add(outflow - inflow == 0)
    
    # Constraint 4: Each virtual node assigned to exactly one substrate node
    # IMPORTANT: Sum over ALL physical nodes, not just candidates
    # Domain filtering happens via meta-edge bandwidth (0 for non-candidates)
    # and during deterministic rounding phase
    for m_idx, m in enumerate(m_node_list):
        v_node = vnr_nodes[m_idx]
        # Sum over ALL physical nodes to avoid infeasibility
        solver.Add(solver.Sum([x[(m, w)] for w in p_node_list]) == 1)
    
    # Constraint 5: Each substrate node hosts at most one virtual node
    for w in p_node_list:
        solver.Add(solver.Sum([x[(m, w)] for m in m_node_list]) <= 1)
    
    # Constraint 6: Edge symmetry
    for u in a_node_list:
        for v in a_node_list:
            solver.Add(x[(u, v)] == x[(v, u)])
    
    # Constraint 7: Edge usage upper bound
    for u in a_node_list:
        for v in a_node_list:
            bw_avail = get_edge_resource(u, v, 'bw')
            solver.Add(x[(u, v)] <= bw_avail)
    
    # Set time limit
    solver.SetTimeLimit(max_time_seconds * 1000)
    
    # Solve
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        # Extract solution: compute probabilities as in GeminiLight
        v_p_value_dict = {}
        
        for m_idx, m in enumerate(m_node_list):
            v_node = vnr_nodes[m_idx]
            v_p_value_dict[v_node] = {}
            
            for p_idx, p in enumerate(p_node_list):
                s_node = substrate_nodes[p_idx]
                
                if s_node not in candidates_dict[v_node]:
                    v_p_value_dict[v_node][s_node] = 0.0
                else:
                    # Probability = x_mp × (Σ_i f^i_mp) + (Σ_i f^i_pm)
                    x_val = x[(m, p)].solution_value()
                    flow_mp = sum(f[(m, p, i)].solution_value() for i in range(len(vnr_edges)))
                    flow_pm = sum(f[(p, m, i)].solution_value() for i in range(len(vnr_edges)))
                    
                    v_p_value_dict[v_node][s_node] = x_val * flow_mp + flow_pm
        
        return v_p_value_dict
    else:
        print(f"LP solver failed with status: {status}")
        return None


def d_vine_sp_embed(substrate_graph, vnr_graph):
    """
    D-ViNE-SP embedding using OR-Tools for LP relaxation.
    
    Args:
        substrate_graph: Substrate network
        vnr_graph: VNR to embed
    
    Returns:
        (mapping, link_paths) if successful, None if failed
    """
    # Step 1: Create domain-aware candidates
    candidates_dict = create_candidates_dict_with_domain(substrate_graph, vnr_graph)
    
    # Check feasibility
    for v in vnr_graph.nodes():
        if not candidates_dict[v]:
            return None
    
    # Step 2: Solve LP relaxation
    v_p_value_dict = solve_d_vine_lp_ortools(substrate_graph, vnr_graph, candidates_dict)
    
    if v_p_value_dict is None:
        return None
    
    # Step 3: Deterministic rounding
    mapping = {}
    used_substrate = set()
    
    # Sort virtual nodes by CPU (heuristic ordering)
    vnodes_sorted = sorted(
        vnr_graph.nodes(),
        key=lambda n: vnr_graph.nodes[n]['cpu'],
        reverse=True
    )
    
    for v in vnodes_sorted:
        # Get available candidates
        available = [s for s in candidates_dict[v] if s not in used_substrate]
        
        if not available:
            # Rollback
            for mapped_v, mapped_s in mapping.items():
                release_node(substrate_graph, mapped_s, vnr_graph.nodes[mapped_v]['cpu'])
            return None
        
        # Select substrate node with maximum probability
        # Filter by candidates during selection (domain enforcement)
        selected_s = max(available, key=lambda s: v_p_value_dict[v].get(s, 0))
        
        # Assign
        mapping[v] = selected_s
        used_substrate.add(selected_s)
        reserve_node(substrate_graph, selected_s, vnr_graph.nodes[v]['cpu'])
    
    # Step 4: Link mapping using shortest path
    link_paths = {}
    
    for u, v in vnr_graph.edges():
        bw_req = vnr_graph.edges[u, v]['bw']
        s_src = mapping[u]
        s_dst = mapping[v]
        
        path = shortest_path_with_capacity(substrate_graph, s_src, s_dst, bw_req)
        
        if path is None:
            # Rollback
            for mapped_v, mapped_s in mapping.items():
                release_node(substrate_graph, mapped_s, vnr_graph.nodes[mapped_v]['cpu'])
            
            for (link_u, link_v), link_path in link_paths.items():
                release_path(substrate_graph, link_path, vnr_graph.edges[link_u, link_v]['bw'])
            
            return None
        
        reserve_path(substrate_graph, path, bw_req)
        link_paths[(u, v)] = path
    
    return mapping, link_paths