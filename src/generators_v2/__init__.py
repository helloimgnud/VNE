# src/generators_v2/__init__.py
"""
generators_v2 — Enhanced network generator package.

Integrates the rich attribute/topology/event-simulator machinery from
virne's network layer with the lightweight JSON-first substrate/VNR
generators already used in src/generators.

Public API
----------
Substrate generation
    generate_substrate          – fast, dict/NetworkX output (v1 compatible)
    generate_substrate_virne    – virne PhysicalNetwork (full attribute system)

VNR / Stream generation
    generate_vnr                – single VNR graph (simple)
    generate_single_vnr         – single VNR graph without timing
    generate_vnr_stream         – v1 stream (balanced domains, fixed size)
    generate_vnr_stream_v2      – v2 stream (bursty/pareto, variable size)
    generate_vnr_stream_virne   – virne VirtualNetworkRequestSimulator

Dataset orchestration
    DatasetGeneratorV2          – extended DatasetGenerator class
    VirneDatasetGenerator       – virne-native config-driven generator

Topology
    TopologyType                – supported topology names enum-like class

Utilities
    load_substrate_from_json    – load JSON → nx.Graph (existing format)
    load_vnr_stream_from_json   – load JSON → list of dicts
"""

from src.generators_v2.substrate_generator import (
    generate_substrate,
    generate_substrate_virne,
    load_substrate_from_json,
)

from src.generators_v2.vnr_generator import (
    generate_vnr,
    generate_single_vnr,
    generate_vnr_stream,
    generate_vnr_stream_v2,
    generate_vnr_stream_virne,
    load_vnr_stream_from_json,
    sample_lifetime,
    sample_inter_arrival,
    sample_vnr_size,
)

from src.generators_v2.dataset_generator import (
    DatasetGeneratorV2,
    VirneDatasetGenerator,
)

from src.generators_v2.topology import TopologyType

__all__ = [
    # Substrate
    "generate_substrate",
    "generate_substrate_virne",
    "load_substrate_from_json",
    # VNR
    "generate_vnr",
    "generate_single_vnr",
    "generate_vnr_stream",
    "generate_vnr_stream_v2",
    "generate_vnr_stream_virne",
    "load_vnr_stream_from_json",
    "sample_lifetime",
    "sample_inter_arrival",
    "sample_vnr_size",
    # Dataset
    "DatasetGeneratorV2",
    "VirneDatasetGenerator",
    # Topology
    "TopologyType",
]
