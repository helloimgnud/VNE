# src/utils/io_utils.py
"""
I/O Utilities for VNE
Logic extracted from utils.py - load/save networks from/to JSON.
"""

import json
import networkx as nx


def load_vnr_stream_from_json(json_path):
    """
    Load VNR stream from JSON file.
    Each VNR includes nodes with domains, links, and timing metadata.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        list: List of NetworkX VNR graphs with metadata
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Handle both list format and dict format
    if isinstance(data, list):
        vnr_list = data
    else:
        vnr_list = data.get("vnrs", [])

    vnr_stream = []

    for v in vnr_list:
        G = nx.Graph()

        # --- Add virtual nodes ---
        for n in v["nodes"]:
            G.add_node(
                n["id"],
                cpu=n["cpu"],
                domain=n["domain"]
            )

        # --- Add virtual links ---
        for e in v["links"]:
            G.add_edge(
                e["u"], e["v"],
                bw=e["bw"]
            )

        # --- Attach metadata on the graph ---
        G.graph["id"] = v["id"]
        G.graph["arrival_time"] = v["arrival_time"]
        G.graph["lifetime"] = v["lifetime"]

        vnr_stream.append(G)

    return vnr_stream


def load_substrate_from_json(json_path):
    """
    Load substrate network from JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        NetworkX graph: Substrate network
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    G = nx.Graph()
    
    # Add nodes
    for n in data["nodes"]:
        G.add_node(
            n["id"],
            cpu=n["cpu"],
            cpu_total=n["cpu_total"],
            domain=n["domain"],
            cost=n["cost"],
        )
    
    # Add edges
    for e in data["links"]:
        G.add_edge(
            e["u"], e["v"],
            bw=e["bw"],
            bw_total=e["bw_total"],
            bw_cost=e["bw_cost"]
        )
    
    return G


def save_substrate_to_json(substrate_graph, json_path):
    """
    Save substrate network to JSON file.
    
    Args:
        substrate_graph: NetworkX substrate graph
        json_path: Path to save JSON
    """
    data = {
        "num_domains": len(set(substrate_graph.nodes[n]['domain'] 
                              for n in substrate_graph.nodes)),
        "nodes": [
            {
                "id": int(n),
                "cpu": int(substrate_graph.nodes[n]["cpu"]),
                "cpu_total": int(substrate_graph.nodes[n]["cpu_total"]),
                "domain": int(substrate_graph.nodes[n]["domain"]),
                "cost": float(substrate_graph.nodes[n]["cost"])
            }
            for n in substrate_graph.nodes
        ],
        "links": [
            {
                "u": int(u),
                "v": int(v),
                "bw": int(substrate_graph.edges[u, v]["bw"]),
                "bw_total": int(substrate_graph.edges[u, v]["bw_total"]),
                "bw_cost": float(substrate_graph.edges[u, v]["bw_cost"])
            }
            for u, v in substrate_graph.edges
        ]
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


def save_vnr_stream_to_json(vnr_stream, json_path):
    """
    Save VNR stream to JSON file.
    
    Args:
        vnr_stream: List of NetworkX VNR graphs
        json_path: Path to save JSON
    """
    vnr_dicts = []
    
    for G in vnr_stream:
        vnr_dicts.append({
            "id": G.graph['id'],
            "arrival_time": G.graph['arrival_time'],
            "lifetime": G.graph['lifetime'],
            "nodes": [
                {
                    "id": int(n),
                    "cpu": int(G.nodes[n]["cpu"]),
                    "domain": int(G.nodes[n]["domain"])
                }
                for n in G.nodes
            ],
            "links": [
                {
                    "u": int(u),
                    "v": int(v),
                    "bw": int(G.edges[u, v]["bw"])
                }
                for u, v in G.edges
            ]
        })

    with open(json_path, "w") as f:
        json.dump(vnr_dicts, f, indent=4)