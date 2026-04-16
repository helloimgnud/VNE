# src/generators_v2/substrate_generator.py
"""
Substrate Network Generator — v2

Two public generators:
  1. generate_substrate       — lightweight, returns nx.Graph + optional JSON export
                                100% backward-compatible with src/generators/substrate_generator.py
  2. generate_substrate_virne — virne-native, returns PhysicalNetwork with full attribute system
                                supports richer topology types (waxman, barabasi_albert, …)

Load helper:
  load_substrate_from_json    — read an existing JSON file → nx.Graph
"""

from __future__ import annotations

import json
import random
import os
from typing import Optional, Tuple, Dict, Any

import networkx as nx

from src.generators_v2.topology import TopologyType, build_topology


# ══════════════════════════════════════════════════════════════════════════════
# 1. Lightweight substrate generator (v1 compatible + extensions)
# ══════════════════════════════════════════════════════════════════════════════

def generate_substrate(
    num_domains: int = 4,
    num_nodes_total: int = 50,
    p_intra: float = 0.5,
    p_inter: float = 0.05,
    cpu_range: Tuple[int, int] = (100, 300),
    bw_range: Tuple[int, int] = (1000, 3000),
    node_cost_range: Tuple[float, float] = (1.0, 10.0),
    inter_domain_bw_cost: Tuple[float, float] = (5.0, 15.0),
    # Extensions beyond v1 ─────────────────────────────────────────────────
    memory_range: Optional[Tuple[int, int]] = None,   # NEW: optional memory attr
    gpu_range: Optional[Tuple[int, int]] = None,      # NEW: optional GPU attr
    latency_range: Optional[Tuple[float, float]] = None,  # NEW: link latency
    ensure_domain_connectivity: bool = True,           # NEW: intra-domain Hamiltonian
    # ─────────────────────────────────────────────────────────────────────────
    seed: Optional[int] = 42,
    export_path: Optional[str] = None,
) -> nx.Graph:
    """
    Generate a multi-domain substrate network.

    Backward-compatible with ``src.generators.substrate_generator.generate_substrate``.
    Adds optional memory, GPU, and latency attributes.

    Parameters
    ----------
    num_domains : int
        Number of administrative domains.
    num_nodes_total : int
        Total substrate nodes across all domains.
    p_intra : float
        Probability of adding an intra-domain edge.
    p_inter : float
        Probability of adding an inter-domain edge.
    cpu_range : (int, int)
        Uniform range for node CPU capacity.
    bw_range : (int, int)
        Uniform range for link bandwidth.
    node_cost_range : (float, float)
        Uniform range for per-node embedding cost.
    inter_domain_bw_cost : (float, float)
        Uniform range for inter-domain link cost multiplier.
    memory_range : (int, int) or None
        If set, adds a ``memory`` / ``memory_total`` attribute to every node.
    gpu_range : (int, int) or None
        If set, adds a ``gpu`` / ``gpu_total`` attribute to every node.
    latency_range : (float, float) or None
        If set, adds a ``latency`` attribute to every link.
    ensure_domain_connectivity : bool
        If True, enforces a spanning path within each domain before adding
        random intra-domain edges, so every domain is internally connected.
    seed : int or None
        Random seed for reproducibility.
    export_path : str or None
        If given, serialises the result to JSON at this path.

    Returns
    -------
    nx.Graph
        Substrate graph with node attrs: cpu, cpu_total, domain, cost
        (+ memory/gpu/latency when requested).
    """
    if seed is not None:
        random.seed(seed)

    G = nx.Graph()
    domain_of: Dict[int, int] = {}

    # ── 1. Distribute nodes across domains ───────────────────────────────────
    remaining = num_nodes_total
    domain_sizes = []
    for i in range(num_domains - 1):
        size = random.randint(1, remaining - (num_domains - i - 1))
        domain_sizes.append(size)
        remaining -= size
    domain_sizes.append(remaining)

    # ── 2. Create nodes ──────────────────────────────────────────────────────
    node_id = 0
    domain_node_lists: Dict[int, list] = {d: [] for d in range(num_domains)}
    for d, size in enumerate(domain_sizes):
        for _ in range(size):
            attrs: Dict[str, Any] = {
                "cpu": random.randint(*cpu_range),
                "domain": d,
                "cost": random.uniform(*node_cost_range),
            }
            attrs["cpu_total"] = attrs["cpu"]

            if memory_range is not None:
                mem = random.randint(*memory_range)
                attrs["memory"] = mem
                attrs["memory_total"] = mem

            if gpu_range is not None:
                gpu = random.randint(*gpu_range)
                attrs["gpu"] = gpu
                attrs["gpu_total"] = gpu

            G.add_node(node_id, **attrs)
            domain_of[node_id] = d
            domain_node_lists[d].append(node_id)
            node_id += 1

    nodes = list(G.nodes)

    # ── 3. Build intra-domain topology ───────────────────────────────────────
    for d, d_nodes in domain_node_lists.items():
        if len(d_nodes) < 2:
            continue
        if ensure_domain_connectivity and len(d_nodes) >= 2:
            # Add a spanning path so each domain stays connected
            for k in range(len(d_nodes) - 1):
                u, v = d_nodes[k], d_nodes[k + 1]
                bw = random.randint(*bw_range)
                edge_attrs = {"bw": bw, "bw_total": bw, "bw_cost": 1.0}
                if latency_range:
                    edge_attrs["latency"] = random.uniform(*latency_range)
                G.add_edge(u, v, **edge_attrs)
        # Additional random intra-domain edges
        for i in range(len(d_nodes)):
            for j in range(i + 1, len(d_nodes)):
                u, v = d_nodes[i], d_nodes[j]
                if G.has_edge(u, v):
                    continue
                if random.random() < p_intra:
                    bw = random.randint(*bw_range)
                    edge_attrs = {"bw": bw, "bw_total": bw, "bw_cost": 1.0}
                    if latency_range:
                        edge_attrs["latency"] = random.uniform(*latency_range)
                    G.add_edge(u, v, **edge_attrs)

    # ── 4. Add inter-domain edges ─────────────────────────────────────────────
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if domain_of[u] == domain_of[v]:
                continue
            if G.has_edge(u, v):
                continue
            if random.random() < p_inter:
                bw = random.randint(*bw_range)
                bw_cost = random.uniform(*inter_domain_bw_cost)
                edge_attrs = {"bw": bw, "bw_total": bw, "bw_cost": bw_cost}
                if latency_range:
                    edge_attrs["latency"] = random.uniform(*latency_range)
                G.add_edge(u, v, **edge_attrs)

    # ── 5. Ensure global connectivity ─────────────────────────────────────────
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for k in range(len(comps) - 1):
            a = next(iter(comps[k]))
            b = next(iter(comps[k + 1]))
            bw = random.randint(*bw_range)
            bw_cost = random.uniform(*inter_domain_bw_cost)
            edge_attrs = {"bw": bw, "bw_total": bw, "bw_cost": bw_cost}
            if latency_range:
                edge_attrs["latency"] = random.uniform(*latency_range)
            G.add_edge(a, b, **edge_attrs)

    # ── 6. Optional JSON export ───────────────────────────────────────────────
    if export_path is not None:
        _export_substrate_json(G, num_domains, export_path,
                               has_memory=memory_range is not None,
                               has_gpu=gpu_range is not None,
                               has_latency=latency_range is not None)

    return G


def _export_substrate_json(
    G: nx.Graph,
    num_domains: int,
    export_path: str,
    has_memory: bool = False,
    has_gpu: bool = False,
    has_latency: bool = False,
) -> None:
    """Serialise substrate graph to the project JSON format."""
    nodes = []
    for n in G.nodes:
        node_dict: Dict[str, Any] = {
            "id": int(n),
            "cpu": int(G.nodes[n]["cpu"]),
            "cpu_total": int(G.nodes[n]["cpu_total"]),
            "domain": int(G.nodes[n]["domain"]),
            "cost": float(G.nodes[n]["cost"]),
        }
        if has_memory:
            node_dict["memory"] = int(G.nodes[n]["memory"])
            node_dict["memory_total"] = int(G.nodes[n]["memory_total"])
        if has_gpu:
            node_dict["gpu"] = int(G.nodes[n]["gpu"])
            node_dict["gpu_total"] = int(G.nodes[n]["gpu_total"])
        nodes.append(node_dict)

    links = []
    for u, v in G.edges:
        link_dict: Dict[str, Any] = {
            "u": int(u),
            "v": int(v),
            "bw": int(G.edges[u, v]["bw"]),
            "bw_total": int(G.edges[u, v]["bw_total"]),
            "bw_cost": float(G.edges[u, v]["bw_cost"]),
        }
        if has_latency and "latency" in G.edges[u, v]:
            link_dict["latency"] = float(G.edges[u, v]["latency"])
        links.append(link_dict)

    substrate_json = {
        "num_domains": num_domains,
        "nodes": nodes,
        "links": links,
    }
    os.makedirs(os.path.dirname(export_path) if os.path.dirname(export_path) else ".", exist_ok=True)
    with open(export_path, "w") as f:
        json.dump(substrate_json, f, indent=4)
    print(f"[OK] Substrate exported to {export_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Virne-native substrate generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_substrate_virne(
    config: Dict[str, Any],
    seed: Optional[int] = None,
    save_dir: Optional[str] = None,
):
    """
    Generate a substrate using virne's PhysicalNetwork attribute system.

    This gives access to:
      - Waxman / Barabási–Albert topologies
      - Rich typed attributes (resource, extrema, position, latency)
      - GML persistence (``PhysicalNetwork.save_dataset`` / ``load_dataset``)

    Parameters
    ----------
    config : dict
        virne-style ``p_net_setting`` dict.  Example::

            config = {
                "topology": {
                    "type": "waxman",
                    "num_nodes": 100,
                    "wm_alpha": 0.5,
                    "wm_beta": 0.2,
                },
                "node_attrs_setting": [
                    {"name": "cpu", "owner": "node", "type": "resource",
                     "generative": True, "distribution": "uniform",
                     "dtype": "int", "low": 100, "high": 300},
                ],
                "link_attrs_setting": [
                    {"name": "bw", "owner": "link", "type": "resource",
                     "generative": True, "distribution": "uniform",
                     "dtype": "int", "low": 1000, "high": 3000},
                ],
            }

    seed : int or None
        Random seed.
    save_dir : str or None
        If given, calls ``p_net.save_dataset(save_dir)``.

    Returns
    -------
    PhysicalNetwork
        A virne PhysicalNetwork object (subclass of nx.Graph).

    Notes
    -----
    Requires ``virne`` to be importable.  virne is located at
    ``d:/HUST_file/@research/@working/virne-main`` and may need to be on
    ``sys.path`` (or installed) before calling this function.
    """
    try:
        from virne.network.physical_network import PhysicalNetwork
    except ImportError as exc:
        raise ImportError(
            "virne is not importable. Add virne-main to sys.path:\n"
            "  import sys; sys.path.insert(0, 'd:/HUST_file/@research/@working/virne-main')"
        ) from exc

    p_net = PhysicalNetwork.from_setting(config, seed=seed)

    if save_dir is not None:
        p_net.save_dataset(save_dir)
        print(f"[OK] virne PhysicalNetwork saved to {save_dir}")

    return p_net


# ══════════════════════════════════════════════════════════════════════════════
# 3. Load utility
# ══════════════════════════════════════════════════════════════════════════════

def load_substrate_from_json(json_path: str) -> nx.Graph:
    """
    Load a substrate from the project JSON format into an nx.Graph.

    The JSON must follow the schema produced by ``generate_substrate``::

        {
          "num_domains": 4,
          "nodes": [{"id": 0, "cpu": 150, "cpu_total": 150, "domain": 0, "cost": 3.5}, ...],
          "links": [{"u": 0, "v": 1, "bw": 2000, "bw_total": 2000, "bw_cost": 1.0}, ...]
        }

    Parameters
    ----------
    json_path : str
        Path to the JSON file.

    Returns
    -------
    nx.Graph
        Substrate graph with all node and edge attributes restored.
    """
    with open(json_path) as f:
        data = json.load(f)

    G = nx.Graph()
    G.graph["num_domains"] = data.get("num_domains", 0)

    for node in data["nodes"]:
        n_id = node.pop("id")
        G.add_node(n_id, **node)

    for link in data["links"]:
        u, v = link.pop("u"), link.pop("v")
        G.add_edge(u, v, **link)

    return G
