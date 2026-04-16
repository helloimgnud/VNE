# src/generators_v2/vnr_generator.py
"""
VNR (Virtual Network Request) Generator — v2

Backward-compatible with src/generators/vnr_generator.py, plus:
  - generate_vnr_stream_virne  : returns virne VirtualNetworkRequestSimulator
  - Richer typed attribute support (memory, gpu, latency, max_latency)
  - sample_vnr_size now accepts min/max kwargs (no global state)
  - load_vnr_stream_from_json  : reload saved JSON → list[dict]
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple, Any

import networkx as nx


# ══════════════════════════════════════════════════════════════════════════════
# Helpers  (extended / parameterised vs v1)
# ══════════════════════════════════════════════════════════════════════════════

def sample_lifetime(max_lifetime: int = 300) -> int:
    """Heavy-tailed (Pareto) lifetime distribution."""
    return min(max(int(random.paretovariate(2.5) * 10), 5), max_lifetime)


def sample_inter_arrival(avg: float = 1.0) -> int:
    """Bursty inter-arrival: 20 % burst, 80 % exponential."""
    if random.random() < 0.2:
        return random.randint(0, 1)
    return max(1, int(random.expovariate(1.0 / avg)))


def sample_vnr_size(min_vnodes: int = 6, max_vnodes: int = 15) -> int:
    """
    Bimodal VNR size sampler.

    Parameters
    ----------
    min_vnodes, max_vnodes : int
        Inclusive bounds.

    Returns
    -------
    int  Number of nodes.
    """
    if min_vnodes >= max_vnodes:
        return min_vnodes
    if max_vnodes - min_vnodes < 3:
        return random.randint(min_vnodes, max_vnodes)
    # 30 % large, 70 % small
    if random.random() < 0.3:
        low = max(min_vnodes, int(0.7 * max_vnodes))
        return random.randint(min(low, max_vnodes), max_vnodes)
    else:
        high = min(max_vnodes, int(0.6 * max_vnodes))
        return random.randint(min_vnodes, max(high, min_vnodes))


def _build_vnr_graph(
    num_nodes: int,
    edge_prob: float,
    cpu_range: Tuple[int, int],
    bw_range: Tuple[int, int],
    memory_range: Optional[Tuple[int, int]] = None,
    gpu_range: Optional[Tuple[int, int]] = None,
    latency_range: Optional[Tuple[float, float]] = None,
    bimodal: bool = False,
) -> nx.Graph:
    """
    Internal helper: build a single connected VNR graph.

    When *bimodal* is True, CPU and BW use the heavy-tail bimodal distribution
    used in v2 stream generators. Otherwise simple uniform sampling.
    """
    G = nx.fast_gnp_random_graph(num_nodes, edge_prob)
    while not nx.is_connected(G):
        u, v = random.sample(range(num_nodes), 2)
        G.add_edge(u, v)

    cpu_lo, cpu_hi = cpu_range
    for n in G.nodes:
        if bimodal and random.random() < 0.2:
            heavy_min = max(cpu_lo, int(0.7 * cpu_hi))
            G.nodes[n]["cpu"] = random.randint(heavy_min, cpu_hi)
        else:
            light_max = max(min(int(0.6 * cpu_hi), cpu_hi), cpu_lo)
            G.nodes[n]["cpu"] = random.randint(cpu_lo, light_max if bimodal else cpu_hi)

        if memory_range:
            G.nodes[n]["memory"] = random.randint(*memory_range)
        if gpu_range:
            G.nodes[n]["gpu"] = random.randint(*gpu_range)

    bw_lo, bw_hi = bw_range
    for u, v in G.edges:
        if bimodal and random.random() < 0.2:
            heavy_min = max(bw_lo, int(0.7 * bw_hi))
            G.edges[u, v]["bw"] = random.randint(heavy_min, bw_hi)
        else:
            light_max = max(min(int(0.6 * bw_hi), bw_hi), bw_lo)
            G.edges[u, v]["bw"] = random.randint(bw_lo, light_max if bimodal else bw_hi)

        if latency_range:
            G.edges[u, v]["latency"] = random.uniform(*latency_range)

    return G


# ══════════════════════════════════════════════════════════════════════════════
# 1. Simple single-VNR generators (v1 compatible)
# ══════════════════════════════════════════════════════════════════════════════

def generate_vnr(
    num_nodes: int = 6,
    edge_prob: float = 0.5,
    cpu_range: Tuple[int, int] = (1, 10),
    bw_range: Tuple[int, int] = (5, 15),
    seed: Optional[int] = 42,
) -> nx.Graph:
    """
    Generate a single VNR graph (simple, no timing).

    Backward-compatible with ``src.generators.vnr_generator.generate_vnr``.
    """
    if seed is not None:
        random.seed(seed)
    return _build_vnr_graph(num_nodes, edge_prob, cpu_range, bw_range)


def generate_single_vnr(
    num_nodes: int = 6,
    edge_prob: float = 0.5,
    cpu_range: Tuple[int, int] = (1, 10),
    bw_range: Tuple[int, int] = (5, 15),
    memory_range: Optional[Tuple[int, int]] = None,
    gpu_range: Optional[Tuple[int, int]] = None,
    latency_range: Optional[Tuple[float, float]] = None,
    max_lifetime: int = 50,  # kept for signature compat; unused internally
) -> nx.Graph:
    """
    Generate a single VNR without timing attributes.

    Backward-compatible with ``src.generators.vnr_generator.generate_single_vnr``.
    Extended with optional memory / GPU / latency attributes.
    """
    return _build_vnr_graph(
        num_nodes, edge_prob, cpu_range, bw_range,
        memory_range=memory_range,
        gpu_range=gpu_range,
        latency_range=latency_range,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. VNR stream v1 — fixed size, balanced domains
# ══════════════════════════════════════════════════════════════════════════════

def generate_vnr_stream(
    num_vnrs: int = 10,
    num_nodes: int = 6,
    num_domains: int = 3,
    edge_prob: float = 0.5,
    cpu_range: Tuple[int, int] = (1, 10),
    bw_range: Tuple[int, int] = (5, 15),
    max_lifetime: int = 50,
    avg_inter_arrival: float = 5.0,
    # Extensions ──────────────────────────────────────────────────────────────
    memory_range: Optional[Tuple[int, int]] = None,
    gpu_range: Optional[Tuple[int, int]] = None,
    latency_range: Optional[Tuple[float, float]] = None,
    # ─────────────────────────────────────────────────────────────────────────
    export_mode: str = "none",   # none | single | multiple
    export_path: str = "vnr_stream.json",
    seed: Optional[int] = 42,
) -> List[nx.Graph]:
    """
    Generate a VNR stream with fixed node count and balanced domain assignment.

    Backward-compatible with ``src.generators.vnr_generator.generate_vnr_stream``.

    Parameters
    ----------
    num_vnrs : int
        Number of VNRs in the stream.
    num_nodes : int
        Number of nodes per VNR (fixed).
    num_domains : int
        Number of domains; nodes distributed in round-robin.
    edge_prob : float
        Edge probability.
    cpu_range, bw_range : (int, int)
        Resource requirement ranges.
    max_lifetime : int
        Upper bound for uniform lifetime draw.
    avg_inter_arrival : float
        Mean of the exponential inter-arrival distribution.
    memory_range, gpu_range, latency_range : (int/float, int/float) or None
        Optional extra attributes (NEW in v2).
    export_mode : "none" | "single" | "multiple"
        "none"     – no files written.
        "single"   – write all VNRs to one JSON array at *export_path*.
        "multiple" – write each VNR to a separate file in *export_path* dir.
    export_path : str
        File or directory path for export.
    seed : int or None
        Random seed.

    Returns
    -------
    list[nx.Graph]
        Each graph has graph-level keys: id, arrival_time, lifetime.
        Each node has: cpu, domain (+ memory, gpu if requested).
        Each edge has: bw (+ latency if requested).
    """
    random.seed(seed)
    current_time = 0
    vnr_list: List[nx.Graph] = []

    if export_mode == "multiple":
        os.makedirs(export_path, exist_ok=True)

    # Balanced round-robin domain assignment
    node_domain = {n: n % num_domains for n in range(num_nodes)}

    for i in range(num_vnrs):
        G = generate_single_vnr(
            num_nodes=num_nodes,
            edge_prob=edge_prob,
            cpu_range=cpu_range,
            bw_range=bw_range,
            memory_range=memory_range,
            gpu_range=gpu_range,
            latency_range=latency_range,
        )
        for n in G.nodes:
            G.nodes[n]["domain"] = node_domain[n]

        inter = max(1, int(random.expovariate(1.0 / avg_inter_arrival)))
        current_time += inter
        lifetime = random.randint(1, max_lifetime)

        G.graph["id"] = i
        G.graph["arrival_time"] = current_time
        G.graph["lifetime"] = lifetime
        vnr_list.append(G)

        if export_mode == "multiple":
            _write_vnr_json(G, os.path.join(export_path, f"vnr_{i}.json"))

    if export_mode == "single":
        _write_vnr_stream_json(vnr_list, export_path)
        print(f"[OK] VNR stream (v1) exported to {export_path}")

    return vnr_list


# ══════════════════════════════════════════════════════════════════════════════
# 3. VNR stream v2 — variable size, bursty arrivals, hot-domain affinity
# ══════════════════════════════════════════════════════════════════════════════

def generate_vnr_stream_v2(
    num_vnrs: int = 500,
    num_domains: int = 3,
    edge_prob: float = 0.5,
    min_vnodes: int = 6,
    max_vnodes: int = 15,
    cpu_range: Tuple[int, int] = (5, 30),
    bw_range: Tuple[int, int] = (10, 50),
    max_lifetime: int = 300,
    avg_inter_arrival: float = 1.0,
    # Extensions ──────────────────────────────────────────────────────────────
    memory_range: Optional[Tuple[int, int]] = None,
    gpu_range: Optional[Tuple[int, int]] = None,
    latency_range: Optional[Tuple[float, float]] = None,
    max_latency_range: Optional[Tuple[float, float]] = None,  # per-VNR SLA
    hot_domain_prob: float = 0.6,    # probability node is in hot-domain
    # ─────────────────────────────────────────────────────────────────────────
    export_path: str = "vnr_stream_v2.json",
    seed: Optional[int] = 42,
) -> List[nx.Graph]:
    """
    Generate a realistic VNR stream with variable sizes and bursty arrivals.

    Backward-compatible with ``src.generators.vnr_generator.generate_vnr_stream_v2``,
    plus new parameters for extra attributes and hot-domain probability.

    Parameters
    ----------
    num_vnrs : int
        Number of VNRs.
    num_domains : int
        Number of domains.
    edge_prob : float
        Edge probability.
    min_vnodes, max_vnodes : int
        Bimodal VNR size range (see *sample_vnr_size*).
    cpu_range, bw_range : (int, int)
        Resource ranges (bimodal distribution applied internally).
    max_lifetime : int
        Upper bound for Pareto lifetime.
    avg_inter_arrival : float
        Mean inter-arrival (bursty exponential).
    memory_range, gpu_range, latency_range : optional
        Extra node/link attributes.
    max_latency_range : (float, float) or None
        If set, each VNR gets a ``max_latency`` graph attribute (end-to-end SLA).
    hot_domain_prob : float
        Probability that a node is assigned to the VNR's "hot" domain.
    export_path : str
        Output JSON file path.
    seed : int or None
        Random seed.

    Returns
    -------
    list[nx.Graph]
        Graphs with variable node count, bimodal resources, bursty timings.
    """
    random.seed(seed)
    current_time = 0
    vnr_list: List[nx.Graph] = []

    cpu_lo, cpu_hi = cpu_range
    bw_lo, bw_hi = bw_range

    for i in range(num_vnrs):
        num_nodes = sample_vnr_size(min_vnodes, max_vnodes)
        G = nx.fast_gnp_random_graph(num_nodes, edge_prob)
        while not nx.is_connected(G):
            u, v = random.sample(range(num_nodes), 2)
            G.add_edge(u, v)

        hot_domain = random.randint(0, num_domains - 1)

        for n in G.nodes:
            # domain affinity
            G.nodes[n]["domain"] = (
                hot_domain if random.random() < hot_domain_prob
                else random.randint(0, num_domains - 1)
            )
            # bimodal CPU
            if random.random() < 0.2:
                G.nodes[n]["cpu"] = random.randint(max(cpu_lo, int(0.7 * cpu_hi)), cpu_hi)
            else:
                light_max = max(min(int(0.6 * cpu_hi), cpu_hi), cpu_lo)
                G.nodes[n]["cpu"] = random.randint(cpu_lo, light_max)
            # optional attrs
            if memory_range:
                G.nodes[n]["memory"] = random.randint(*memory_range)
            if gpu_range:
                G.nodes[n]["gpu"] = random.randint(*gpu_range)

        for u, v in G.edges:
            # bimodal BW
            if random.random() < 0.2:
                G.edges[u, v]["bw"] = random.randint(max(bw_lo, int(0.7 * bw_hi)), bw_hi)
            else:
                light_max = max(min(int(0.6 * bw_hi), bw_hi), bw_lo)
                G.edges[u, v]["bw"] = random.randint(bw_lo, light_max)
            if latency_range:
                G.edges[u, v]["latency"] = random.uniform(*latency_range)

        current_time += sample_inter_arrival(avg_inter_arrival)
        lifetime = sample_lifetime(max_lifetime)

        G.graph["id"] = i
        G.graph["arrival_time"] = current_time
        G.graph["lifetime"] = lifetime
        if max_latency_range:
            G.graph["max_latency"] = random.uniform(*max_latency_range)

        vnr_list.append(G)

    # Export
    _write_vnr_stream_json(vnr_list, export_path)
    print(f"[OK] VNR stream v2 exported to {export_path}")
    return vnr_list


# ══════════════════════════════════════════════════════════════════════════════
# 4. Virne-native VNR stream generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_vnr_stream_virne(
    config: Dict[str, Any],
    seed: Optional[int] = None,
    save_dir: Optional[str] = None,
    changeable: bool = False,
):
    """
    Generate a VNR stream using virne's VirtualNetworkRequestSimulator.

    Provides access to:
      - Distribution-based attribute generation (uniform / normal / exponential …)
      - Full event timeline (arrival + departure)
      - GML persistence (``save_dataset`` / ``load_dataset``)
      - Changeable mode: 4-stage progressive resource/size escalation

    Parameters
    ----------
    config : dict
        Must contain ``"v_sim_setting"`` key.  Example::

            config = {
                "v_sim_setting": {
                    "num_v_nets": 500,
                    "topology": {"type": "random", "random_prob": 0.5},
                    "v_net_size": {"distribution": "uniform", "dtype": "int",
                                   "low": 2, "high": 10},
                    "lifetime": {"distribution": "exponential", "scale": 500,
                                 "dtype": "int"},
                    "arrival_rate": {"distribution": "exponential", "scale": 100,
                                     "dtype": "int"},
                    "node_attrs_setting": [
                        {"name": "cpu", "owner": "node", "type": "resource",
                         "generative": True, "distribution": "uniform",
                         "dtype": "int", "low": 1, "high": 30},
                    ],
                    "link_attrs_setting": [
                        {"name": "bw", "owner": "link", "type": "resource",
                         "generative": True, "distribution": "uniform",
                         "dtype": "int", "low": 1, "high": 30},
                    ],
                    "output": {
                        "events_file_name": "events.yaml",
                        "setting_file_name": "v_sim_setting.yaml",
                    },
                }
            }

    seed : int or None
        Random seed.
    save_dir : str or None
        If provided, persist the dataset (GML + YAML).
    changeable : bool
        If True, use the 4-stage dynamic resource/size escalation generator.

    Returns
    -------
    VirtualNetworkRequestSimulator
        virne simulator with ``.v_nets`` (list of VirtualNetwork) and
        ``.events`` (list of VirtualNetworkEvent).

    Notes
    -----
    Requires ``virne`` to be importable.
    """
    try:
        from virne.network.dataset_generator import Generator
    except ImportError as exc:
        raise ImportError(
            "virne is not importable. Add virne-main to sys.path:\n"
            "  import sys; sys.path.insert(0, 'd:/HUST_file/@research/@working/virne-main')"
        ) from exc

    if changeable:
        v_net_sim = Generator.generate_changeable_v_nets_dataset_from_config(
            config, save=save_dir is not None
        )
    else:
        v_net_sim = Generator.generate_v_nets_dataset_from_config(
            config, save=save_dir is not None
        )

    if save_dir and not changeable:
        v_net_sim.save_dataset(save_dir)
        print(f"[OK] virne VNR stream saved to {save_dir}")

    return v_net_sim


# ══════════════════════════════════════════════════════════════════════════════
# 5. JSON utilities
# ══════════════════════════════════════════════════════════════════════════════

def _graph_to_vnr_dict(G: nx.Graph) -> Dict[str, Any]:
    """Serialise a VNR nx.Graph to the project JSON dict format."""
    vnr: Dict[str, Any] = {
        "id": G.graph.get("id", 0),
        "arrival_time": G.graph.get("arrival_time", 0),
        "lifetime": G.graph.get("lifetime", 0),
        "nodes": [],
        "links": [],
    }
    if "max_latency" in G.graph:
        vnr["max_latency"] = G.graph["max_latency"]

    for n in G.nodes:
        node_dict: Dict[str, Any] = {"id": int(n)}
        for attr in ("cpu", "domain", "memory", "gpu"):
            if attr in G.nodes[n]:
                node_dict[attr] = int(G.nodes[n][attr])
        vnr["nodes"].append(node_dict)

    for u, v in G.edges:
        link_dict: Dict[str, Any] = {"u": int(u), "v": int(v)}
        for attr in ("bw",):
            if attr in G.edges[u, v]:
                link_dict[attr] = int(G.edges[u, v][attr])
        for attr in ("latency",):
            if attr in G.edges[u, v]:
                link_dict[attr] = float(G.edges[u, v][attr])
        vnr["links"].append(link_dict)

    return vnr


def _write_vnr_json(G: nx.Graph, path: str) -> None:
    with open(path, "w") as f:
        json.dump(_graph_to_vnr_dict(G), f, indent=4)


def _write_vnr_stream_json(vnr_list: List[nx.Graph], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    dicts = [_graph_to_vnr_dict(G) for G in vnr_list]
    with open(path, "w") as f:
        json.dump(dicts, f, indent=2)


def load_vnr_stream_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Load a VNR stream from the project JSON format.

    Parameters
    ----------
    json_path : str
        Path to a JSON file produced by generate_vnr_stream / generate_vnr_stream_v2.

    Returns
    -------
    list[dict]
        Raw list of VNR dicts; each has keys:
        id, arrival_time, lifetime, nodes (list), links (list).
    """
    with open(json_path) as f:
        return json.load(f)
