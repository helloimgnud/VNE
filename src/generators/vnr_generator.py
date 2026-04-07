# src/generators/vnr_generator.py
"""
VNR (Virtual Network Request) Generator
Logic extracted from generator.py - generates VNR streams with timing.
"""

import random
import json
import os
import networkx as nx


def generate_vnr(num_nodes=6,
                 edge_prob=0.5,
                 cpu_range=(1, 10),
                 bw_range=(5, 15),
                 seed=42):
    """
    Generate a single VNR (simple version without domains).
    
    Args:
        num_nodes: Number of virtual nodes
        edge_prob: Edge probability (Erdos-Renyi model)
        cpu_range: (min, max) CPU requirement range
        bw_range: (min, max) bandwidth requirement range
        seed: Random seed
        
    Returns:
        NetworkX graph: Generated VNR
    """
    if seed is not None:
        random.seed(seed)

    G = nx.fast_gnp_random_graph(n=num_nodes, p=edge_prob, seed=seed)
    
    # Ensure connected graph
    while not nx.is_connected(G):
        u, v = random.sample(range(num_nodes), 2)
        G.add_edge(u, v)

    # Assign node CPU requirements
    for n in G.nodes:
        G.nodes[n]['cpu'] = random.randint(*cpu_range)
    
    # Assign link bandwidth requirements
    for u, v in list(G.edges()):
        G.edges[u, v]['bw'] = random.randint(*bw_range)

    return G


def generate_single_vnr(num_nodes=6,
                        edge_prob=0.5,
                        cpu_range=(1, 10),
                        bw_range=(5, 15),
                        max_lifetime=50):
    """
    Generate a single VNR without timing attributes.
    """
    G = nx.fast_gnp_random_graph(n=num_nodes, p=edge_prob)

    # Ensure connected graph
    while not nx.is_connected(G):
        u, v = random.sample(range(num_nodes), 2)
        G.add_edge(u, v)

    # Assign node CPU
    for n in G.nodes:
        G.nodes[n]['cpu'] = random.randint(*cpu_range)

    # Assign link BW
    for u, v in G.edges:
        G.edges[u, v]['bw'] = random.randint(*bw_range)

    return G


def generate_vnr_stream(num_vnrs=10,
                        num_nodes=6,
                        num_domains=3,
                        edge_prob=0.5,
                        cpu_range=(1, 10),
                        bw_range=(5, 15),
                        max_lifetime=50,
                        avg_inter_arrival=5,
                        export_mode="none",
                        export_path="vnr_stream.json",
                        seed=42):
    """
    Generate VNR stream with timing (v1 - balanced domain assignment).
    
    Args:
        num_vnrs: Number of VNRs in stream
        num_nodes: Number of nodes per VNR
        num_domains: Number of domains per VNR
        edge_prob: Edge probability
        cpu_range: (min, max) CPU requirement range
        bw_range: (min, max) bandwidth requirement range
        max_lifetime: Maximum lifetime
        avg_inter_arrival: Average inter-arrival time (exponential)
        export_mode: "none" | "single" | "multiple"
        export_path: Path for export
        seed: Random seed
        
    Returns:
        List of VNR graphs with timing metadata
    """
    random.seed(seed)
    current_time = 0
    vnr_list = []

    # Prepare directory for multiple exports
    if export_mode == "multiple":
        os.makedirs(export_path, exist_ok=True)

    # Precompute domain assignments (balanced)
    domain_assignments = [[] for _ in range(num_domains)]
    for n in range(num_nodes):
        domain_assignments[n % num_domains].append(n)
    
    # Flatten: node_domain[n] = d
    node_domain = {}
    for d, nodes in enumerate(domain_assignments):
        for n in nodes:
            node_domain[n] = d

    for i in range(num_vnrs):
        # --- 1. Generate VNR graph ---
        G = generate_single_vnr(
            num_nodes=num_nodes,
            edge_prob=edge_prob,
            cpu_range=cpu_range,
            bw_range=bw_range,
            max_lifetime=max_lifetime
        )

        # --- 2. Generate arrival time (exponential distribution) ---
        inter_arrival = max(1, int(random.expovariate(1 / avg_inter_arrival)))
        current_time += inter_arrival

        # --- 3. Generate lifetime (uniform) ---
        lifetime = random.randint(1, max_lifetime)

        # --- 4. Assign domains to nodes ---
        for n in G.nodes:
            G.nodes[n]['domain'] = node_domain[n]

        # --- 5. Add timing metadata to graph ---
        G.graph['id'] = i
        G.graph['arrival_time'] = current_time
        G.graph['lifetime'] = lifetime

        vnr_list.append(G)

        # --- 6. Export individual files ---
        if export_mode == "multiple":
            vnr_dict = {
                "id": i,
                "arrival_time": current_time,
                "lifetime": lifetime,
                "nodes": [
                    {"id": int(n), "cpu": int(G.nodes[n]["cpu"]), 
                     "domain": int(G.nodes[n]["domain"])}
                    for n in G.nodes
                ],
                "links": [
                    {"u": int(u), "v": int(v), "bw": int(G.edges[u, v]["bw"])}
                    for u, v in G.edges
                ]
            }
            
            file_path = os.path.join(export_path, f"vnr_{i}.json")
            with open(file_path, "w") as f:
                json.dump(vnr_dict, f, indent=4)

    # --- 7. Export as a single combined file ---
    if export_mode == "single":
        vnr_dicts = []
        for G in vnr_list:
            vnr_dicts.append({
                "id": G.graph['id'],
                "arrival_time": G.graph['arrival_time'],
                "lifetime": G.graph['lifetime'],
                "nodes": [
                    {"id": int(n), "cpu": int(G.nodes[n]["cpu"]),
                     "domain": int(G.nodes[n]["domain"])}
                    for n in G.nodes
                ],
                "links": [
                    {"u": int(u), "v": int(v), "bw": int(G.edges[u, v]["bw"])}
                    for u, v in G.edges
                ]
            })
        
        with open(export_path, "w") as f:
            json.dump(vnr_dicts, f, indent=4)
        
        print(f"✓ Exported VNR stream to {export_path}")

    return vnr_list


# ============================================================
# VNR Stream v2 - More realistic distributions
# ============================================================

def sample_lifetime(max_lifetime=300):
    """Pareto distribution for long-tail lifetime."""
    lifetime = int(random.paretovariate(2.5) * 10)
    return min(max(lifetime, 5), max_lifetime)


def sample_inter_arrival(avg=1.0):
    """Bursty arrival pattern."""
    if random.random() < 0.2:  # 20% burst arrivals
        return random.randint(0, 1)
    return max(1, int(random.expovariate(1 / avg)))


def sample_vnr_size():
    """Variable VNR sizes."""
    if random.random() < 0.3:  # 30% large VNRs
        return random.randint(10, 15)
    return random.randint(6, 9)


def sample_cpu():
    """Variable CPU requirements."""
    if random.random() < 0.2:  # 20% heavy nodes
        return random.randint(20, 30)
    return random.randint(5, 15)


def sample_bw():
    """Variable bandwidth requirements."""
    if random.random() < 0.2:  # 20% high bandwidth
        return random.randint(30, 50)
    return random.randint(10, 25)


def generate_vnr_stream_v2(num_vnrs=500,
                           num_domains=3,
                           edge_prob=0.5,
                           max_lifetime=300,
                           avg_inter_arrival=1.0,
                           export_path="vnr_stream_v2.json",
                           seed=42):
    """
    Generate VNR stream with realistic distributions (v2).
    Features:
    - Variable VNR sizes
    - Heavy-tailed lifetime distribution
    - Bursty arrivals
    - Domain affinity (hot domains)
    
    Args:
        num_vnrs: Number of VNRs in stream
        num_domains: Number of domains
        edge_prob: Edge probability
        max_lifetime: Maximum lifetime
        avg_inter_arrival: Average inter-arrival time
        export_path: Path to export JSON
        seed: Random seed
        
    Returns:
        List of NetworkX graphs
    """
    random.seed(seed)
    current_time = 0
    vnr_list = []

    for i in range(num_vnrs):
        # Variable VNR size
        num_nodes = sample_vnr_size()
        G = nx.fast_gnp_random_graph(num_nodes, edge_prob)

        while not nx.is_connected(G):
            u, v = random.sample(range(num_nodes), 2)
            G.add_edge(u, v)

        # Domain assignment with affinity (hot domain)
        node_domain = {}
        hot_domain = random.randint(0, num_domains - 1)

        for n in G.nodes:
            # 60% chance to be in hot domain
            if random.random() < 0.6:
                node_domain[n] = hot_domain
            else:
                node_domain[n] = random.randint(0, num_domains - 1)

            G.nodes[n]["cpu"] = sample_cpu()
            G.nodes[n]["domain"] = node_domain[n]

        for u, v in G.edges:
            G.edges[u, v]["bw"] = sample_bw()

        # Timing with realistic distributions
        current_time += sample_inter_arrival(avg_inter_arrival)
        lifetime = sample_lifetime(max_lifetime)

        G.graph['id'] = i
        G.graph['arrival_time'] = current_time
        G.graph['lifetime'] = lifetime

        vnr_list.append(G)

    # Export to JSON
    vnr_dicts = []
    for G in vnr_list:
        vnr_dicts.append({
            "id": G.graph['id'],
            "arrival_time": G.graph['arrival_time'],
            "lifetime": G.graph['lifetime'],
            "nodes": [
                {"id": int(n), "cpu": int(G.nodes[n]["cpu"]),
                 "domain": int(G.nodes[n]["domain"])}
                for n in G.nodes
            ],
            "links": [
                {"u": int(u), "v": int(v), "bw": int(G.edges[u, v]["bw"])}
                for u, v in G.edges
            ]
        })

    with open(export_path, "w") as f:
        json.dump(vnr_dicts, f, indent=2)

    print(f"✓ VNR stream v2 exported to {export_path}")
    return vnr_list