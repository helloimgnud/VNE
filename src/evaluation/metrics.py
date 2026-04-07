"""
Evaluation Metrics for VNE
Logic extracted from eval.py - calculates costs, revenues, and ratios.
"""


def revenue_of_vnr(vnr_graph):
    """
    Calculate total revenue of a VNR.
    Revenue = Sum of CPU requirements + Sum of bandwidth requirements
    
    Args:
        vnr_graph: NetworkX VNR graph
        
    Returns:
        float: Total revenue
    """
    cpu = sum(vnr_graph.nodes[n]['cpu'] for n in vnr_graph.nodes())
    bw = sum(vnr_graph.edges[e]['bw'] for e in vnr_graph.edges())
    return cpu + bw


def cost_of_embedding(mapping, link_paths, vnr_graph, substrate_graph=None):
    """
    Calculate total cost of embedding a VNR.
    Cost = Node cost + Link cost
    
    Node cost: CPU_req × node_cost for each mapped node
    Link cost: BW_req × sum(bw_cost) for each edge in path
    
    Args:
        mapping: Dict {vnode: snode}
        link_paths: Dict {(u,v): path}
        vnr_graph: NetworkX VNR graph
        substrate_graph: NetworkX substrate graph (required for costs)
        
    Returns:
        float: Total embedding cost
    """
    if substrate_graph is None:
        raise ValueError("substrate_graph must be provided for cost calculation")

    node_cost_total = 0
    link_cost_total = 0

    # Node costs
    for v_node, p_node in mapping.items():
        v_cpu = vnr_graph.nodes[v_node]['cpu']
        p_cost = substrate_graph.nodes[p_node].get('cost', 1.0)
        node_cost_total += v_cpu * p_cost

    # Link costs
    for (u, v) in vnr_graph.edges():
        bw_req = vnr_graph.edges[u, v]['bw']
        path = link_paths.get((u, v))
        
        if path is None:
            return float('inf')

        path_cost = 0
        for i in range(len(path) - 1):
            u_p, v_p = path[i], path[i + 1]
            edge_data = substrate_graph.edges[u_p, v_p]
            bw_cost = edge_data.get('bw_cost', 1.0)
            path_cost += bw_req * bw_cost

        link_cost_total += path_cost

    return node_cost_total + link_cost_total


def revenue_cost_ratio(total_revenue, total_cost):
    """
    Calculate revenue-to-cost ratio.
    
    Args:
        total_revenue: Total revenue
        total_cost: Total cost
        
    Returns:
        float: Revenue/Cost ratio (inf if cost is 0)
    """
    if total_cost == 0:
        return float('inf')
    return total_revenue / total_cost


def acceptance_ratio(successful_embeddings, total_requests):
    """
    Calculate VNR acceptance ratio.
    
    Args:
        successful_embeddings: Number of successfully embedded VNRs
        total_requests: Total number of VNR requests
        
    Returns:
        float: Acceptance ratio [0, 1]
    """
    if total_requests == 0:
        return 0.0
    return successful_embeddings / total_requests