# src/utils/graph_utils.py
"""
Graph Utilities for VNE
Logic extracted from utils.py - graph operations and resource management.
"""

import networkx as nx
import random
import numpy as np


def copy_substrate(sub):
    """
    Create a deep copy of substrate network.
    
    Args:
        sub: NetworkX substrate graph
        
    Returns:
        NetworkX graph: Copy of substrate
    """
    G = nx.Graph()
    for n, d in sub.nodes(data=True):
        G.add_node(n, **d)
    for u, v, d in sub.edges(data=True):
        G.add_edge(u, v, **d)
    return G


def shortest_path_with_capacity(substrate, src, dst, bw_required):
    """
    Find shortest path with sufficient bandwidth capacity.
    
    Args:
        substrate: NetworkX substrate graph
        src: Source node
        dst: Destination node
        bw_required: Required bandwidth
        
    Returns:
        list: Path as list of nodes, or None if no valid path
    """
    G = substrate.copy()
    
    # Remove edges with insufficient bandwidth
    for u, v, data in list(G.edges(data=True)):
        if data.get('bw', 0) < bw_required:
            G.remove_edge(u, v)
    
    try:
        path = nx.shortest_path(G, src, dst)
        return path
    except nx.NetworkXNoPath:
        return None


def can_place_node(substrate, sub_node, cpu_req):
    """
    Check if a substrate node has sufficient CPU capacity.
    
    Args:
        substrate: NetworkX substrate graph
        sub_node: Substrate node ID
        cpu_req: Required CPU
        
    Returns:
        bool: True if node can accommodate the requirement
    """
    return substrate.nodes[sub_node].get('cpu', 0) >= cpu_req


def reserve_node(substrate, sub_node, cpu_amount):
    """
    Reserve CPU resources on a substrate node.
    
    Args:
        substrate: NetworkX substrate graph
        sub_node: Substrate node ID
        cpu_amount: Amount of CPU to reserve
    """
    substrate.nodes[sub_node]['cpu'] -= cpu_amount


def release_node(substrate, sub_node, cpu_amount):
    """
    Release CPU resources on a substrate node.
    
    Args:
        substrate: NetworkX substrate graph
        sub_node: Substrate node ID
        cpu_amount: Amount of CPU to release
    """
    substrate.nodes[sub_node]['cpu'] += cpu_amount


def reserve_path(substrate, path, bw_amount):
    """
    Reserve bandwidth resources on a path.
    
    Args:
        substrate: NetworkX substrate graph
        path: List of nodes representing the path
        bw_amount: Amount of bandwidth to reserve
    """
    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        substrate.edges[a, b]['bw'] -= bw_amount


def release_path(substrate, path, bw_amount):
    """
    Release bandwidth resources on a path.
    
    Args:
        substrate: NetworkX substrate graph
        path: List of nodes representing the path
        bw_amount: Amount of bandwidth to release
    """
    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        substrate.edges[a, b]['bw'] += bw_amount


def cpu_free_list(substrate):
    """
    Get available CPU for all substrate nodes.
    
    Args:
        substrate: NetworkX substrate graph
        
    Returns:
        dict: {node_id: available_cpu}
    """
    return {n: substrate.nodes[n]['cpu'] for n in substrate.nodes}


def random_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

def substrate_utilisation(substrate) -> dict:
    total_cpu = sum(d.get('max_cpu', d.get('cpu', 1e-9)) for n, d in substrate.nodes(data=True))
    avail_cpu = sum(d.get('cpu', 0) for n, d in substrate.nodes(data=True))
    
    total_bw = sum(d.get('max_bw', d.get('bw', 1e-9)) for u, v, d in substrate.edges(data=True))
    avail_bw = sum(d.get('bw', 0) for u, v, d in substrate.edges(data=True))
    
    return {
        'cpu_util': 1.0 - (avail_cpu / max(1e-9, total_cpu)),
        'bw_util': 1.0 - (avail_bw / max(1e-9, total_bw))
    }

def substrate_resource_summary(substrate) -> str:
    u = substrate_utilisation(substrate)
    return f"CPU Util: {u['cpu_util']:.1%} | BW Util: {u['bw_util']:.1%}"

def release_vnr_embedding(substrate, vnr, mapping, link_paths):
    for v_node, s_node in mapping.items():
        if s_node in substrate:
            cpu_req = float(vnr.nodes[v_node].get('cpu', 0.0))
            substrate.nodes[s_node]['cpu'] += cpu_req
            
    for (u, v), path in link_paths.items():
        bw_req = float(vnr.edges[u, v].get('bw', 0.0))
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            if substrate.has_edge(a, b):
                substrate.edges[a, b]['bw'] += bw_req