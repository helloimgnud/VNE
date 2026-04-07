# scripts/example_parser.py
"""
Example Dataset Parser/Generator

This script demonstrates how to:
1. Generate custom datasets with specific configurations
2. Parse existing datasets from files
3. Create datasets programmatically with custom parameters

Run examples:
  # Generate a simple dataset
  python scripts/example_parser.py --generate simple
  
  # Generate with custom parameters
  python scripts/example_parser.py --generate custom \
      --num-domains 6 --num-vnodes 10 --num-vnrs 500 --output-dir my_dataset
  
  # Parse and display info about existing dataset
  python scripts/example_parser.py --parse dataset/fig6
"""

import os
import sys
import argparse
import json
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from generators.generator import generate_substrate, generate_vnr_stream_v2
from utils.io_utils import load_substrate_from_json, load_vnr_stream_from_json


# ============================================================
# 1. SIMPLE DATASET GENERATION EXAMPLE
# ============================================================

def generate_simple_dataset(output_dir="dataset/example_simple", seed=42):
    """
    Generate a simple dataset with fixed parameters.
    Good for quick testing and demos.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("GENERATING SIMPLE DATASET")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}")
    print(f"  Seed: {seed}")
    
    # Generate substrate network
    substrate_path = os.path.join(output_dir, "substrate.json")
    print("\n[1/2] Generating substrate network...")
    
    generate_substrate(
        num_domains=4,           # 4 domains
        num_nodes_total=50,      # 50 nodes total (12-13 per domain)
        p_intra=0.5,             # 50% intra-domain edge probability
        p_inter=0.1,             # 10% inter-domain edge probability
        cpu_range=(100, 300),    # CPU capacity: 100-300
        bw_range=(1000, 3000),   # Bandwidth: 1000-3000
        node_cost_range=(1, 10), # Node cost: 1-10
        inter_domain_bw_cost=(5, 15),
        seed=seed,
        export_path=substrate_path
    )
    print(f"  ✓ Saved: {substrate_path}")
    
    # Generate VNR stream
    vnr_path = os.path.join(output_dir, "vnr_stream.json")
    print("\n[2/2] Generating VNR stream...")
    
    generate_vnr_stream_v2(
        num_vnrs=100,            # 100 VNR requests
        num_domains=4,           # 4 domains
        min_vnodes=2,            # Min 2 nodes per VNR
        max_vnodes=6,            # Max 6 nodes per VNR
        cpu_range=(50, 200),     # CPU demand: 50-200
        bw_range=(500, 2000),    # Bandwidth demand: 500-2000
        max_lifetime=500,        # Max VNR lifetime
        avg_inter_arrival=1.0,   # Average 1 time unit between arrivals
        seed=seed,
        export_path=vnr_path
    )
    print(f"  ✓ Saved: {vnr_path}")
    
    # Save metadata
    metadata = {
        "name": "simple_example",
        "description": "Simple example dataset for testing",
        "num_domains": 4,
        "num_nodes": 50,
        "num_vnrs": 100,
        "seed": seed,
        "substrate_path": substrate_path,
        "vnr_path": vnr_path
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  ✓ Metadata: {metadata_path}")
    
    print(f"\n{'='*60}")
    print("✓ SIMPLE DATASET GENERATED")
    print(f"{'='*60}\n")
    
    return metadata


# ============================================================
# 2. CUSTOM DATASET GENERATION EXAMPLE
# ============================================================

def generate_custom_dataset(
    output_dir="dataset/example_custom",
    num_domains=4,
    num_nodes=100,
    num_vnrs=200,
    min_vnodes=2,
    max_vnodes=8,
    num_replicas=1,
    seed=42
):
    """
    Generate a custom dataset with user-specified parameters.
    Supports multiple replicas for statistical significance.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("GENERATING CUSTOM DATASET")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}")
    print(f"  Domains: {num_domains}")
    print(f"  Nodes: {num_nodes}")
    print(f"  VNRs: {num_vnrs}")
    print(f"  VNodes: {min_vnodes}-{max_vnodes}")
    print(f"  Replicas: {num_replicas}")
    print(f"  Base seed: {seed}")
    
    replicas = []
    
    for replica_id in range(num_replicas):
        replica_seed = seed + replica_id * 1000
        
        if num_replicas > 1:
            replica_dir = os.path.join(output_dir, f"replica_{replica_id}")
            os.makedirs(replica_dir, exist_ok=True)
            print(f"\n--- Replica {replica_id + 1}/{num_replicas} (seed={replica_seed}) ---")
        else:
            replica_dir = output_dir
        
        # Substrate
        substrate_path = os.path.join(replica_dir, "substrate.json")
        print(f"\n  Generating substrate ({num_domains} domains, {num_nodes} nodes)...")
        
        generate_substrate(
            num_domains=num_domains,
            num_nodes_total=num_nodes,
            p_intra=0.5,
            p_inter=0.1,
            cpu_range=(100, 300),
            bw_range=(1000, 3000),
            node_cost_range=(1, 10),
            inter_domain_bw_cost=(5, 15),
            seed=replica_seed,
            export_path=substrate_path
        )
        
        # VNR stream
        vnr_path = os.path.join(replica_dir, "vnr_stream.json")
        print(f"  Generating VNR stream ({num_vnrs} VNRs)...")
        
        generate_vnr_stream_v2(
            num_vnrs=num_vnrs,
            num_domains=num_domains,
            min_vnodes=min_vnodes,
            max_vnodes=max_vnodes + 1,
            cpu_range=(50, 200),
            bw_range=(500, 2000),
            max_lifetime=1000,
            avg_inter_arrival=1.0,
            seed=replica_seed + 100,
            export_path=vnr_path
        )
        
        replicas.append({
            "replica_id": replica_id,
            "seed": replica_seed,
            "substrate_path": substrate_path,
            "vnr_path": vnr_path
        })
    
    # Metadata
    metadata = {
        "name": "custom_example",
        "description": "Custom dataset with user-specified parameters",
        "num_domains": num_domains,
        "num_nodes": num_nodes,
        "num_vnrs": num_vnrs,
        "vnode_range": [min_vnodes, max_vnodes],
        "num_replicas": num_replicas,
        "base_seed": seed,
        "replicas": replicas
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ CUSTOM DATASET GENERATED ({num_replicas} replica(s))")
    print(f"{'='*60}\n")
    
    return metadata


# ============================================================
# 3. PROGRAMMATIC GENERATION EXAMPLE
# ============================================================

def generate_parametric_sweep(
    output_dir="dataset/sweep",
    vnode_values=[2, 4, 6, 8, 10],
    domain_values=[4, 6, 8],
    seed=42
):
    """
    Generate datasets for a parametric sweep over vnodes and domains.
    Useful for comprehensive experiments.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("GENERATING PARAMETRIC SWEEP DATASET")
    print(f"{'='*60}")
    print(f"  VNode values: {vnode_values}")
    print(f"  Domain values: {domain_values}")
    
    configs = []
    
    for num_domains in domain_values:
        domain_dir = os.path.join(output_dir, f"domains_{num_domains}")
        os.makedirs(domain_dir, exist_ok=True)
        
        # One substrate per domain configuration
        substrate_path = os.path.join(domain_dir, "substrate.json")
        
        print(f"\n[Domains={num_domains}] Generating substrate...")
        generate_substrate(
            num_domains=num_domains,
            num_nodes_total=100,
            seed=seed + num_domains * 100,
            export_path=substrate_path
        )
        
        # Multiple VNR streams for different vnode counts
        for num_vnodes in vnode_values:
            vnr_path = os.path.join(domain_dir, f"vnr_{num_vnodes}nodes.json")
            
            print(f"  → VNR stream: {num_vnodes} nodes")
            generate_vnr_stream_v2(
                num_vnrs=200,
                num_domains=num_domains,
                min_vnodes=num_vnodes,
                max_vnodes=num_vnodes + 1,
                seed=seed + num_domains * 100 + num_vnodes,
                export_path=vnr_path
            )
            
            configs.append({
                "num_domains": num_domains,
                "num_vnodes": num_vnodes,
                "substrate_path": substrate_path,
                "vnr_path": vnr_path
            })
    
    # Metadata
    metadata = {
        "name": "parametric_sweep",
        "vnode_values": vnode_values,
        "domain_values": domain_values,
        "configs": configs,
        "seed": seed
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ PARAMETRIC SWEEP GENERATED ({len(configs)} configurations)")
    print(f"{'='*60}\n")
    
    return metadata


# ============================================================
# 4. DATASET PARSER / LOADER
# ============================================================

def parse_dataset(dataset_dir):
    """
    Parse and display information about an existing dataset.
    """
    print(f"\n{'='*60}")
    print("PARSING DATASET")
    print(f"{'='*60}")
    print(f"  Directory: {dataset_dir}")
    
    # Check for metadata
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    
    if not os.path.exists(metadata_path):
        print(f"\n  ✗ No metadata.json found in {dataset_dir}")
        
        # Try to find substrate/vnr files directly
        files = os.listdir(dataset_dir)
        substrate_files = [f for f in files if 'substrate' in f.lower() and f.endswith('.json')]
        vnr_files = [f for f in files if 'vnr' in f.lower() and f.endswith('.json')]
        
        print(f"\n  Found files:")
        print(f"    Substrate files: {substrate_files}")
        print(f"    VNR files: {vnr_files}")
        
        if substrate_files:
            substrate_path = os.path.join(dataset_dir, substrate_files[0])
            _display_substrate_info(substrate_path)
        
        if vnr_files:
            vnr_path = os.path.join(dataset_dir, vnr_files[0])
            _display_vnr_info(vnr_path)
        
        return None
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n  Metadata loaded successfully!")
    print(f"\n  Experiment: {metadata.get('experiment', metadata.get('name', 'Unknown'))}")
    print(f"  Description: {metadata.get('description', 'N/A')}")
    
    # Display experiment-specific info
    if 'replicas' in metadata:
        print(f"\n  Replicas: {len(metadata['replicas'])}")
        for replica in metadata['replicas'][:3]:  # Show first 3
            print(f"    - Replica {replica['replica_id']}: seed={replica['seed']}")
        if len(metadata['replicas']) > 3:
            print(f"    ... and {len(metadata['replicas']) - 3} more")
    
    if 'vnode_range' in metadata:
        print(f"  VNode range: {metadata['vnode_range']}")
    
    if 'domain_range' in metadata:
        print(f"  Domain range: {metadata['domain_range']}")
    
    # Try to load and display info about first substrate
    if 'substrate_path' in metadata:
        _display_substrate_info(metadata['substrate_path'])
    elif 'replicas' in metadata and metadata['replicas']:
        _display_substrate_info(metadata['replicas'][0]['substrate_path'])
    
    print(f"\n{'='*60}")
    print("✓ PARSING COMPLETE")
    print(f"{'='*60}\n")
    
    return metadata


def _display_substrate_info(substrate_path):
    """Helper to display substrate network info."""
    if not os.path.exists(substrate_path):
        print(f"\n  Substrate file not found: {substrate_path}")
        return
    
    try:
        substrate = load_substrate_from_json(substrate_path)
        
        print(f"\n  Substrate Network:")
        print(f"    Nodes: {substrate.number_of_nodes()}")
        print(f"    Edges: {substrate.number_of_edges()}")
        
        # Count domains
        domains = set()
        for node in substrate.nodes():
            domains.add(substrate.nodes[node].get('domain', 'unknown'))
        print(f"    Domains: {len(domains)}")
        
        # CPU stats
        cpus = [substrate.nodes[n]['cpu'] for n in substrate.nodes()]
        print(f"    CPU range: {min(cpus)}-{max(cpus)} (avg: {sum(cpus)/len(cpus):.1f})")
        
    except Exception as e:
        print(f"    Error loading substrate: {e}")


def _display_vnr_info(vnr_path):
    """Helper to display VNR stream info."""
    if not os.path.exists(vnr_path):
        print(f"\n  VNR file not found: {vnr_path}")
        return
    
    try:
        vnr_stream = load_vnr_stream_from_json(vnr_path)
        
        print(f"\n  VNR Stream:")
        print(f"    Total VNRs: {len(vnr_stream)}")
        
        # Node count stats
        node_counts = [vnr.number_of_nodes() for vnr, _ in vnr_stream]
        print(f"    Nodes/VNR: {min(node_counts)}-{max(node_counts)} (avg: {sum(node_counts)/len(node_counts):.1f})")
        
    except Exception as e:
        print(f"    Error loading VNRs: {e}")


# ============================================================
# MAIN CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Example Dataset Generator and Parser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate simple dataset
  python scripts/example_parser.py --generate simple
  
  # Generate custom dataset
  python scripts/example_parser.py --generate custom --num-domains 6 --num-vnrs 300
  
  # Generate parametric sweep
  python scripts/example_parser.py --generate sweep
  
  # Parse existing dataset
  python scripts/example_parser.py --parse dataset/fig6
        """
    )
    
    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--generate', choices=['simple', 'custom', 'sweep'],
                      help='Generate a new dataset')
    mode.add_argument('--parse', type=str, metavar='DIR',
                      help='Parse an existing dataset directory')
    
    # Generation parameters
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for generated dataset')
    parser.add_argument('--num-domains', type=int, default=4,
                        help='Number of domains (default: 4)')
    parser.add_argument('--num-nodes', type=int, default=100,
                        help='Number of substrate nodes (default: 100)')
    parser.add_argument('--num-vnrs', type=int, default=200,
                        help='Number of VNRs (default: 200)')
    parser.add_argument('--min-vnodes', type=int, default=2,
                        help='Min virtual nodes per VNR (default: 2)')
    parser.add_argument('--max-vnodes', type=int, default=8,
                        help='Max virtual nodes per VNR (default: 8)')
    parser.add_argument('--num-replicas', type=int, default=1,
                        help='Number of replicas (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    if args.parse:
        parse_dataset(args.parse)
    
    elif args.generate == 'simple':
        output_dir = args.output_dir or "dataset/example_simple"
        generate_simple_dataset(output_dir=output_dir, seed=args.seed)
    
    elif args.generate == 'custom':
        output_dir = args.output_dir or "dataset/example_custom"
        generate_custom_dataset(
            output_dir=output_dir,
            num_domains=args.num_domains,
            num_nodes=args.num_nodes,
            num_vnrs=args.num_vnrs,
            min_vnodes=args.min_vnodes,
            max_vnodes=args.max_vnodes,
            num_replicas=args.num_replicas,
            seed=args.seed
        )
    
    elif args.generate == 'sweep':
        output_dir = args.output_dir or "dataset/example_sweep"
        generate_parametric_sweep(output_dir=output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
