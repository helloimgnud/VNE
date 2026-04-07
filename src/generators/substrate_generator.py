# src/generators/substrate_generator.py
"""
Substrate Network Generator
Logic extracted from generator.py - generates substrate topologies.
"""

import random
import json
import networkx as nx


def generate_substrate(num_domains=4,
                       num_nodes_total=50,
                       p_intra=0.5,
                       p_inter=0.05,
                       cpu_range=(100, 300),
                       bw_range=(1000, 3000),
                       node_cost_range=(1, 10),
                       inter_domain_bw_cost=(5, 15),
                       seed=42,
                       export_path=None):
    """
    Generate multi-domain substrate network.
    
    Args:
        num_domains: Number of domains in substrate
        num_nodes_total: Total number of nodes across all domains
        p_intra: Probability of intra-domain edges
        p_inter: Probability of inter-domain edges
        cpu_range: (min, max) CPU capacity range
        bw_range: (min, max) bandwidth capacity range
        node_cost_range: (min, max) node cost range
        inter_domain_bw_cost: (min, max) inter-domain bandwidth cost range
        seed: Random seed for reproducibility
        export_path: Optional path to export substrate to JSON
        
    Returns:
        NetworkX graph: Generated substrate network
    """
    if seed is not None:
        random.seed(seed)

    G = nx.Graph()
    domain_of = {}

    # --- 1. Generate domain sizes (balanced distribution) ---
    remaining = num_nodes_total
    domain_sizes = []
    for i in range(num_domains - 1):
        size = random.randint(1, remaining - (num_domains - i - 1))
        domain_sizes.append(size)
        remaining -= size
    domain_sizes.append(remaining)

    # --- 2. Create substrate nodes ---
    node_id = 0
    for d, size in enumerate(domain_sizes):
        for _ in range(size):
            cpu = random.randint(*cpu_range)
            cost = random.uniform(*node_cost_range)
            G.add_node(node_id, 
                      cpu=cpu, 
                      cpu_total=cpu,  # For tracking original capacity
                      domain=d, 
                      cost=cost)
            domain_of[node_id] = d
            node_id += 1

    nodes = list(G.nodes)

    # --- 3. Create substrate links ---
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            
            if domain_of[u] == domain_of[v]:
                # INTRA-domain edge (cheaper, more likely)
                if random.random() < p_intra:
                    bw = random.randint(*bw_range)
                    bw_cost = 1.0  # Intra-domain edges have unit cost
                    G.add_edge(u, v, 
                             bw=bw, 
                             bw_total=bw,  # For tracking original capacity
                             bw_cost=bw_cost)
            else:
                # INTER-domain edge (expensive, less likely)
                if random.random() < p_inter:
                    bw = random.randint(*bw_range)
                    bw_cost = random.uniform(*inter_domain_bw_cost)
                    G.add_edge(u, v, 
                             bw=bw, 
                             bw_total=bw,
                             bw_cost=bw_cost)

    # --- 4. Ensure global connectivity ---
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for k in range(len(comps) - 1):
            # Connect adjacent components
            a = next(iter(comps[k]))
            b = next(iter(comps[k + 1]))
            bw = random.randint(*bw_range)
            bw_cost = random.uniform(*inter_domain_bw_cost)
            G.add_edge(a, b, bw=bw, bw_total=bw, bw_cost=bw_cost)

    # --- 5. Export to JSON (optional) ---
    if export_path is not None:
        substrate_json = {
            "num_domains": num_domains,
            "nodes": [
                {
                    "id": int(n),
                    "cpu": int(G.nodes[n]["cpu"]),
                    "cpu_total": int(G.nodes[n]["cpu_total"]),
                    "domain": int(G.nodes[n]["domain"]),
                    "cost": float(G.nodes[n]["cost"])
                }
                for n in G.nodes
            ],
            "links": [
                {
                    "u": int(u),
                    "v": int(v),
                    "bw": int(G.edges[u, v]["bw"]),
                    "bw_total": int(G.edges[u, v]["bw_total"]),
                    "bw_cost": float(G.edges[u, v]["bw_cost"])
                }
                for u, v in G.edges
            ]
        }

        with open(export_path, "w") as f:
            json.dump(substrate_json, f, indent=4)

        print(f"✓ Substrate exported to {export_path}")

    return G