# src/generators/generator.py
import random
import json
import os
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
    If export_path is a string, the substrate graph will also be exported to JSON.
    """

    if seed is not None:
        random.seed(seed)

    G = nx.Graph()
    domain_of = {}

    # --- 1. Generate domain sizes ---
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
            G.add_node(node_id, cpu=cpu, cpu_total=cpu, domain=d, cost=cost)
            domain_of[node_id] = d
            node_id += 1

    nodes = list(G.nodes)

    # --- 3. Create substrate links ---
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if domain_of[u] == domain_of[v]:  # INTRA domain
                if random.random() < p_intra:
                    bw = random.randint(*bw_range)
                    bw_cost = 1.0
                    G.add_edge(u, v, bw=bw, bw_total=bw, bw_cost=bw_cost)
            else:  # INTER domain
                if random.random() < p_inter:
                    bw = random.randint(*bw_range)
                    bw_cost = random.uniform(*inter_domain_bw_cost)
                    G.add_edge(u, v, bw=bw, bw_total=bw, bw_cost=bw_cost)

    # --- 4. Ensure global connectivity ---
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for k in range(len(comps) - 1):
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

        print(f"Substrate exported to {export_path}")

    return G
    
def generate_vnr(num_nodes=6,
                     edge_prob=0.5,
                     cpu_range=(1, 10),
                     bw_range=(5, 15),
                     seed=42):

    if seed is not None:
        random.seed(seed)

    G = nx.fast_gnp_random_graph(n=num_nodes, p=edge_prob, seed=seed)
    while not nx.is_connected(G):
        u, v = random.sample(range(num_nodes), 2)
        G.add_edge(u, v)

    for n in G.nodes:
        G.nodes[n]['cpu'] = random.randint(*cpu_range)
    for u, v in list(G.edges()):
        G.edges[u, v]['bw'] = random.randint(*bw_range)

    return G

def generate_single_vnr(num_nodes=6,
                        edge_prob=0.5,
                        cpu_range=(1, 10),
                        bw_range=(5, 15),
                        max_lifetime=50):

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
def generate_vnr_stream(
        num_vnrs=10,
        num_nodes=6,
        num_domains=3,                 # NEW: number of VNR domains
        edge_prob=0.5,
        cpu_range=(1, 10),
        bw_range=(5, 15),
        max_lifetime=50,
        avg_inter_arrival=5,
        export_mode="none",            # none | single | multiple
        export_path="vnr_stream.json",
        seed=42):

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

        # --- 2. Generate arrival time ---
        inter_arrival = max(1, int(random.expovariate(1 / avg_inter_arrival)))
        current_time += inter_arrival

        # --- 3. Generate lifetime ---
        lifetime = random.randint(1, max_lifetime)

        # --- 4. Build VNR structure ---
        vnr = {
            "id": i,
            "arrival_time": current_time,
            "lifetime": lifetime,
            "nodes": [
                {
                    "id": int(n),
                    "cpu": int(G.nodes[n]["cpu"]),
                    "domain": int(node_domain[n])   # <<< assign domain here
                }
                for n in G.nodes
            ],
            "links": [
                {"u": int(u), "v": int(v), "bw": int(G.edges[u, v]["bw"])}
                for u, v in G.edges
            ]
        }

        vnr_list.append(vnr)

        # --- 5. Export individual files ---
        if export_mode == "multiple":
            file_path = os.path.join(export_path, f"vnr_{i}.json")
            with open(file_path, "w") as f:
                json.dump(vnr, f, indent=4)

    # --- 6. Export as a single combined file ---
    if export_mode == "single":
        with open(export_path, "w") as f:
            json.dump(vnr_list, f, indent=4)
        print(f"Exported VNR stream to {export_path}")

    return vnr_list

#--- VNR stream v2 ---
def sample_lifetime(max_lifetime=300):
    # Pareto-like long tail
    lifetime = int(random.paretovariate(2.5) * 10)
    return min(max(lifetime, 5), max_lifetime)

def sample_inter_arrival(avg=1.0):
    if random.random() < 0.2:  # 20% burst
        return random.randint(0, 1)
    return max(1, int(random.expovariate(1 / avg)))


def sample_vnr_size(min_vnodes=6, max_vnodes=15):
    """
    Robust VNR size sampler.
    - Supports fixed-size VNRs (min == max)
    - Supports small ranges safely
    - Uses bimodal distribution when possible
    """
    # --- FIXED SIZE ---
    if min_vnodes >= max_vnodes:
        return min_vnodes

    # --- SMALL RANGE: fallback to uniform ---
    if max_vnodes - min_vnodes < 3:
        return random.randint(min_vnodes, max_vnodes)

    # --- NORMAL BIMODAL ---
    if random.random() < 0.3:  # large VNR
        low = max(min_vnodes, int(0.7 * max_vnodes))
        low = min(low, max_vnodes)
        return random.randint(low, max_vnodes)
    else:                      # small VNR
        high = min(max_vnodes, int(0.6 * max_vnodes))
        high = max(high, min_vnodes)
        return random.randint(min_vnodes, high)

def sample_cpu():
    if random.random() < 0.2:
        return random.randint(20, 30)  # heavy node
    return random.randint(5, 15)

def sample_bw():
    if random.random() < 0.2:
        return random.randint(30, 50)
    return random.randint(10, 25)

def generate_vnr_stream_v2(
        num_vnrs=500,
        num_domains=3,
        edge_prob=0.5,
        min_vnodes=6,              # Min VNR size
        max_vnodes=15,             # Max VNR size
        cpu_range=(5, 30),         # NEW: CPU range for virtual nodes
        bw_range=(10, 50),         # NEW: Bandwidth range for virtual links
        max_lifetime=300,
        avg_inter_arrival=1.0,
        export_path="vnr_stream_v2.json",
        seed=42):
    """
    Generate VNR stream with configurable CPU and bandwidth ranges.
    
    Args:
        num_vnrs: Number of VNR requests to generate
        num_domains: Number of domains for domain constraints
        edge_prob: Edge probability for random graph generation
        min_vnodes: Minimum number of nodes per VNR
        max_vnodes: Maximum number of nodes per VNR
        cpu_range: (min_cpu, max_cpu) range for virtual node CPU requirements
        bw_range: (min_bw, max_bw) range for virtual link bandwidth requirements
        max_lifetime: Maximum lifetime for VNRs
        avg_inter_arrival: Average inter-arrival time
        export_path: Path to export the VNR stream JSON
        seed: Random seed for reproducibility
        
    Returns:
        List of VNR dictionaries
    """
    random.seed(seed)
    current_time = 0
    vnr_list = []
    
    cpu_min, cpu_max = cpu_range
    bw_min, bw_max = bw_range

    for i in range(num_vnrs):

        num_nodes = sample_vnr_size(min_vnodes, max_vnodes)
        G = nx.fast_gnp_random_graph(num_nodes, edge_prob)

        while not nx.is_connected(G):
            u, v = random.sample(range(num_nodes), 2)
            G.add_edge(u, v)

        # node domains (biased)
        node_domain = {}
        hot_domain = random.randint(0, num_domains - 1)

        for n in G.nodes:
            if random.random() < 0.6:
                node_domain[n] = hot_domain
            else:
                node_domain[n] = random.randint(0, num_domains - 1)

            # Use configurable CPU range with bimodal distribution
            if random.random() < 0.2:  # 20% heavy nodes
                heavy_min = max(cpu_min, int(0.7 * cpu_max))
                G.nodes[n]["cpu"] = random.randint(heavy_min, cpu_max)
            else:
                light_max = min(cpu_max, int(0.6 * cpu_max))
                light_max = max(light_max, cpu_min)
                G.nodes[n]["cpu"] = random.randint(cpu_min, light_max)

        for u, v in G.edges:
            # Use configurable BW range with bimodal distribution
            if random.random() < 0.2:  # 20% heavy links
                heavy_min = max(bw_min, int(0.7 * bw_max))
                G.edges[u, v]["bw"] = random.randint(heavy_min, bw_max)
            else:
                light_max = min(bw_max, int(0.6 * bw_max))
                light_max = max(light_max, bw_min)
                G.edges[u, v]["bw"] = random.randint(bw_min, light_max)

        # timing
        current_time += sample_inter_arrival(avg_inter_arrival)
        lifetime = sample_lifetime(max_lifetime)

        vnr = {
            "id": i,
            "arrival_time": current_time,
            "lifetime": lifetime,
            "nodes": [
                {
                    "id": int(n),
                    "cpu": int(G.nodes[n]["cpu"]),
                    "domain": int(node_domain[n])
                }
                for n in G.nodes
            ],
            "links": [
                {"u": int(u), "v": int(v), "bw": int(G.edges[u, v]["bw"])}
                for u, v in G.edges
            ]
        }

        vnr_list.append(vnr)

    with open(export_path, "w") as f:
        json.dump(vnr_list, f, indent=2)

    print(f"VNR stream v2 exported to {export_path}")
    return vnr_list

if __name__ == "__main__":
    generate_vnr_stream(
        num_vnrs=5,
        export_mode="single",
        export_path="vnr_stream.json"
    )

    G = generate_substrate(
        num_domains=6,
        num_nodes_total=80,
        p_intra=0.7,
        p_inter=0.03,
        export_path="substrate_topology.json"
    )