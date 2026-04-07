# src/eval.py

def revenue_of_vnr(vnr_graph):
    cpu = sum(vnr_graph.nodes[n]['cpu'] for n in vnr_graph.nodes())
    bw = sum(vnr_graph.edges[e]['bw'] for e in vnr_graph.edges())
    return cpu + bw

def cost_of_embedding(mapping, link_paths, vnr_graph, substrate_graph=None):
    if substrate_graph is None:
        raise ValueError("substrate_graph must be provided for cost calculation with costs.")

    node_cost_total = 0
    link_cost_total = 0

    for v_node, p_node in mapping.items():
        v_cpu = vnr_graph.nodes[v_node]['cpu']
        p_cost = substrate_graph.nodes[p_node].get('cost', 1.0)
        node_cost_total += v_cpu * p_cost

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
    if total_cost == 0:
        return float('inf')
    return total_revenue / total_cost