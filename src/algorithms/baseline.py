# # src/algorithms/baseline.py
# from collections import deque
# from datetime import datetime
# from utils.graph_utils import reserve_node, reserve_path
# from evaluation.eval import revenue_of_vnr
# import networkx as nx


# def calculate_node_resource(substrate_graph, node):
#     """
#     Calculate available resource for a substrate node using the paper's formula:
#     CPU(n_s) * Σ bw(l_s)
#     where l_s are adjacent links of n_s
#     """
#     cpu = substrate_graph.nodes[node]['cpu']
    
#     # Sum bandwidth of all adjacent links
#     total_bw = 0
#     for neighbor in substrate_graph.neighbors(node):
#         edge_data = substrate_graph.edges[node, neighbor]
#         total_bw += edge_data['bw']
    
#     return cpu * total_bw


# def find_k_shortest_paths_with_capacity(substrate_graph, source, target, bw_required, max_k=10):
#     """
#     Find K-shortest paths with sufficient bandwidth capacity.
#     Searches for increasing K until a valid path is found.
    
#     Args:
#         substrate_graph: NetworkX graph
#         source: Source node
#         target: Target node
#         bw_required: Required bandwidth
#         max_k: Maximum number of shortest paths to try
        
#     Returns:
#         List of nodes representing the path, or None if no valid path found
#     """
#     try:
#         # Use NetworkX's k_shortest_paths with length weight
#         # We iterate through paths in order of increasing length
#         k_paths_generator = nx.shortest_simple_paths(
#             substrate_graph, 
#             source, 
#             target, 
#             weight='length'
#         )
        
#         paths_checked = 0
#         for path in k_paths_generator:
#             paths_checked += 1
            
#             # Check if this path has sufficient bandwidth on all edges
#             has_capacity = True
#             for i in range(len(path) - 1):
#                 edge = (path[i], path[i+1])
#                 if substrate_graph.edges[edge]['bw'] < bw_required:
#                     has_capacity = False
#                     break
            
#             if has_capacity:
#                 # Found a valid path
#                 return path
            
#             # Stop after checking K paths
#             if paths_checked >= max_k:
#                 break
        
#         # No valid path found in K-shortest paths
#         return None
        
#     except nx.NetworkXNoPath:
#         # No path exists between source and target
#         return None


# def baseline_embed_batch(substrate_graph, vnr_list):
#     """
#     Embed multiple VNRs in a time window according to the paper's algorithm.
    
#     Algorithm:
#     1. Sort requests by revenue (descending)
#     2. For each request:
#        a. Map nodes using greedy algorithm
#        b. Map links using K-shortest paths
#        c. If success, accept; if fail, reject
    
#     Args:
#         substrate_graph: NetworkX graph (substrate network)
#         vnr_list: List of tuples (vnr_graph, revenue) where vnr_graph is NetworkX
        
#     Returns:
#         accepted_requests: List of tuples (vnr_graph, mapping, link_paths)
#         rejected_requests: List of vnr_graphs that couldn't be embedded
#     """
#     # Step 1: Sort requests by revenue (descending)
#     sorted_vnrs = sorted(vnr_list, key=lambda x: x[1], reverse=True)
    
#     accepted_requests = []
#     rejected_requests = []
    
#     # Step 2: Process each request in order of decreasing revenue
#     for vnr_graph, revenue in sorted_vnrs:
#         result = embed_single_vnr(substrate_graph, vnr_graph)
        
#         if result is not None:
#             mapping, link_paths = result
#             accepted_requests.append((vnr_graph, mapping, link_paths))
#         else:
#             # Request rejected - could be stored in request queue for retry
#             rejected_requests.append(vnr_graph)
    
#     return accepted_requests, rejected_requests


# def embed_single_vnr(substrate_graph, vnr_graph, max_k=10):
#     """
#     Embed a single VNR using the paper's greedy node mapping algorithm
#     and K-shortest paths for link mapping.
    
#     Args:
#         substrate_graph: NetworkX graph
#         vnr_graph: NetworkX graph (VNR to embed)
#         max_k: Maximum K for K-shortest paths search
        
#     Returns:
#         (mapping, link_paths) if successful, None otherwise
#     """
#     # ===== STEP 1: Node Mapping =====
#     # Sort virtual nodes by CPU requirement (descending) for stable ordering
#     vnodes_sorted = sorted(
#         vnr_graph.nodes, 
#         key=lambda n: vnr_graph.nodes[n]['cpu'], 
#         reverse=True
#     )
    
#     mapping = {}
#     used_substrate_nodes = set()
    
#     # Map each virtual node
#     for v in vnodes_sorted:
#         cpu_req = vnr_graph.nodes[v]['cpu']
#         v_domain = vnr_graph.nodes[v]['domain']
        
#         # Find subset S of substrate nodes satisfying restrictions
#         candidate_nodes = []
#         for s in substrate_graph.nodes:
#             # Skip already used nodes
#             if s in used_substrate_nodes:
#                 continue
            
#             # Check CPU capacity
#             if substrate_graph.nodes[s]['cpu'] < cpu_req:
#                 continue
            
#             # Check domain restriction
#             s_domain = substrate_graph.nodes[s]['domain']
#             if v_domain != s_domain:
#                 continue
            
#             candidate_nodes.append(s)
        
#         # Find substrate node with maximum available resource
#         if not candidate_nodes:
#             # Fail - rollback node mappings
#             rollback_node_mapping(substrate_graph, vnr_graph, mapping)
#             return None
        
#         # Select node with maximum available resource using paper's formula
#         best_node = max(
#             candidate_nodes, 
#             key=lambda s: calculate_node_resource(substrate_graph, s)
#         )
        
#         # Reserve the node
#         mapping[v] = best_node
#         used_substrate_nodes.add(best_node)
#         reserve_node(substrate_graph, best_node, cpu_req)
    
#     # ===== STEP 2: Link Mapping with K-Shortest Paths =====
#     link_paths = {}
    
#     for u, v in vnr_graph.edges():
#         bw_req = vnr_graph.edges[u, v]['bw']
#         s_src = mapping[u]
#         s_dst = mapping[v]
        
#         # Step 3: Search K-shortest paths for increasing K
#         # Stop when we find a path with sufficient bandwidth
#         path = find_k_shortest_paths_with_capacity(
#             substrate_graph, 
#             s_src, 
#             s_dst, 
#             bw_req,
#             max_k=max_k
#         )
        
#         # Step 4: If fail for some virtual link, reject this request
#         if path is None:
#             # Fail - rollback everything
#             rollback_node_mapping(substrate_graph, vnr_graph, mapping)
#             rollback_link_mapping(substrate_graph, vnr_graph, link_paths)
#             return None
        
#         # Reserve the path
#         reserve_path(substrate_graph, path, bw_req)
#         link_paths[(u, v)] = path
    
#     return mapping, link_paths


# def rollback_node_mapping(substrate_graph, vnr_graph, mapping):
#     """Release reserved CPU resources for mapped nodes."""
#     for v, s in mapping.items():
#         cpu_to_release = vnr_graph.nodes[v]['cpu']
#         substrate_graph.nodes[s]['cpu'] += cpu_to_release


# def rollback_link_mapping(substrate_graph, vnr_graph, link_paths):
#     """Release reserved bandwidth resources for mapped links."""
#     for (u, v), path in link_paths.items():
#         bw_to_release = vnr_graph.edges[u, v]['bw']
#         for i in range(len(path) - 1):
#             edge = (path[i], path[i+1])
#             substrate_graph.edges[edge]['bw'] += bw_to_release

# src/algorithms/baseline.py
from collections import deque
from datetime import datetime
from utils.graph_utils import reserve_node, reserve_path
from evaluation.eval import revenue_of_vnr
import networkx as nx


def calculate_node_resource(substrate_graph, node):
    """
    Calculate available resource for a substrate node using the paper's formula:
    CPU(n_s) * Σ bw(l_s)
    where l_s are adjacent links of n_s
    """
    cpu = substrate_graph.nodes[node]['cpu']
    
    # Sum bandwidth of all adjacent links
    total_bw = 0
    for neighbor in substrate_graph.neighbors(node):
        edge_data = substrate_graph.edges[node, neighbor]
        total_bw += edge_data['bw']
    
    return cpu * total_bw


def shortest_path_with_capacity(substrate_graph, source, target, bw_required):
    """
    Find shortest path with sufficient bandwidth capacity.
    Uses shortest path algorithm with length as weight.
    
    Args:
        substrate_graph: NetworkX graph
        source: Source node
        target: Target node
        bw_required: Required bandwidth
        
    Returns:
        List of nodes representing the path, or None if no valid path found
    """
    try:
        # Find shortest path based on 'length' weight
        path = nx.shortest_path(substrate_graph, source, target, weight='length')
        
        # Check if this path has sufficient bandwidth on all edges
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            if substrate_graph.edges[edge]['bw'] < bw_required:
                return None
        
        return path
        
    except nx.NetworkXNoPath:
        # No path exists between source and target
        return None


def baseline_embed_batch(substrate_graph, vnr_list):
    """
    Embed multiple VNRs in a time window according to the paper's algorithm.
    
    Algorithm:
    1. Sort requests by revenue (descending)
    2. For each request:
       a. Map nodes using greedy algorithm
       b. Map links using shortest path
       c. If success, accept; if fail, reject
    
    Args:
        substrate_graph: NetworkX graph (substrate network)
        vnr_list: List of tuples (vnr_graph, revenue) where vnr_graph is NetworkX
        
    Returns:
        accepted_requests: List of tuples (vnr_graph, mapping, link_paths)
        rejected_requests: List of vnr_graphs that couldn't be embedded
    """
    # Step 1: Sort requests by revenue (descending)
    sorted_vnrs = sorted(vnr_list, key=lambda x: x[1], reverse=True)
    
    accepted_requests = []
    rejected_requests = []
    
    # Step 2: Process each request in order of decreasing revenue
    for vnr_graph, revenue in sorted_vnrs:
        result = embed_single_vnr(substrate_graph, vnr_graph)
        
        if result is not None:
            mapping, link_paths = result
            accepted_requests.append((vnr_graph, mapping, link_paths))
        else:
            # Request rejected - could be stored in request queue for retry
            rejected_requests.append(vnr_graph)
    
    return accepted_requests, rejected_requests


def embed_single_vnr(substrate_graph, vnr_graph):
    """
    Embed a single VNR using the paper's greedy node mapping algorithm
    and shortest path for link mapping.
    
    Args:
        substrate_graph: NetworkX graph
        vnr_graph: NetworkX graph (VNR to embed)
        
    Returns:
        (mapping, link_paths) if successful, None otherwise
    """
    # ===== STEP 1: Node Mapping =====
    # Sort virtual nodes by CPU requirement (descending) for stable ordering
    vnodes_sorted = sorted(
        vnr_graph.nodes, 
        key=lambda n: vnr_graph.nodes[n]['cpu'], 
        reverse=True
    )
    
    mapping = {}
    used_substrate_nodes = set()
    
    # Map each virtual node
    for v in vnodes_sorted:
        cpu_req = vnr_graph.nodes[v]['cpu']
        v_domain = vnr_graph.nodes[v]['domain']
        
        # Find subset S of substrate nodes satisfying restrictions
        candidate_nodes = []
        for s in substrate_graph.nodes:
            # Skip already used nodes
            if s in used_substrate_nodes:
                continue
            
            # Check CPU capacity
            if substrate_graph.nodes[s]['cpu'] < cpu_req:
                continue
            
            # Check domain restriction
            s_domain = substrate_graph.nodes[s]['domain']
            if v_domain != s_domain:
                continue
            
            candidate_nodes.append(s)
        
        # Find substrate node with maximum available resource
        if not candidate_nodes:
            # Fail - rollback node mappings
            rollback_node_mapping(substrate_graph, vnr_graph, mapping)
            return None
        
        # Select node with maximum available resource using paper's formula
        best_node = max(
            candidate_nodes, 
            key=lambda s: calculate_node_resource(substrate_graph, s)
        )
        
        # Reserve the node
        mapping[v] = best_node
        used_substrate_nodes.add(best_node)
        reserve_node(substrate_graph, best_node, cpu_req)
    
    # ===== STEP 2: Link Mapping with Shortest Path =====
    link_paths = {}
    
    for u, v in vnr_graph.edges():
        bw_req = vnr_graph.edges[u, v]['bw']
        s_src = mapping[u]
        s_dst = mapping[v]
        
        # Find shortest path with sufficient bandwidth
        path = shortest_path_with_capacity(
            substrate_graph, 
            s_src, 
            s_dst, 
            bw_req
        )
        
        # If fail for some virtual link, reject this request
        if path is None:
            # Fail - rollback everything
            rollback_node_mapping(substrate_graph, vnr_graph, mapping)
            rollback_link_mapping(substrate_graph, vnr_graph, link_paths)
            return None
        
        # Reserve the path
        reserve_path(substrate_graph, path, bw_req)
        link_paths[(u, v)] = path
    
    return mapping, link_paths


def rollback_node_mapping(substrate_graph, vnr_graph, mapping):
    """Release reserved CPU resources for mapped nodes."""
    for v, s in mapping.items():
        cpu_to_release = vnr_graph.nodes[v]['cpu']
        substrate_graph.nodes[s]['cpu'] += cpu_to_release


def rollback_link_mapping(substrate_graph, vnr_graph, link_paths):
    """Release reserved bandwidth resources for mapped links."""
    for (u, v), path in link_paths.items():
        bw_to_release = vnr_graph.edges[u, v]['bw']
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            substrate_graph.edges[edge]['bw'] += bw_to_release