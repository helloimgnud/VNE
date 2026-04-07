# scripts/generate_datasets.py
"""
Dataset Generation Script
Generates all datasets needed for VNE experiments.
Run this BEFORE running experiments.
"""

import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from generators.dataset_generator import DatasetGenerator


def parse_range(value, default):
    """Parse a range string like '100,200' into a tuple (100, 200).
    If single value, returns (value, value)."""
    if value is None:
        return default
    parts = [int(x.strip()) for x in value.split(',')]
    if len(parts) == 1:
        return (parts[0], parts[0])
    return (parts[0], parts[1])


def main():
    parser = argparse.ArgumentParser(
        description="Generate datasets for VNE experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all datasets with default settings
  python scripts/generate_datasets.py --experiments all
  
  # Generate with 10 replicas and custom ranges
  python scripts/generate_datasets.py --experiments fig6 --num-replicas 10 --substrate-nodes 50,100 --num-vnrs 150,250 --force
  
  # Generate specific dataset
  python scripts/generate_datasets.py --experiments fig6
  
  # Force regeneration (overwrite existing)
  python scripts/generate_datasets.py --force --experiments fig6
        """
    )
    
    parser.add_argument(
        '--experiments',
        nargs='+',
        choices=['fig6', 'fig7', 'fig8', 'all'],
        default=['all'],
        help='Which experiment datasets to generate'
    )
    
    parser.add_argument(
        '--output-dir',
        default='dataset',
        help='Output directory for datasets (default: dataset/)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration even if dataset exists'
    )
    
    # Multi-replica and range parameters
    parser.add_argument(
        '--num-replicas',
        type=int,
        default=10,
        help='Number of dataset replicas to generate (default: 10)'
    )
    
    parser.add_argument(
        '--substrate-nodes',
        type=str,
        default=None,
        help='Substrate nodes range, e.g., "80,120" (default: 100,100)'
    )
    
    parser.add_argument(
        '--num-vnrs',
        type=str,
        default=None,
        help='Number of VNRs range, e.g., "150,250" (default: 200,200)'
    )
    
    parser.add_argument(
        '--vnode-range',
        type=str,
        default=None,
        help='Virtual node counts, e.g., "2,4,6,8" (default: 2,4,6,8)'
    )
    
    parser.add_argument(
        '--base-seed',
        type=int,
        default=42,
        help='Base random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Parse range arguments
    substrate_nodes_range = parse_range(args.substrate_nodes, (100, 100))
    num_vnrs_range = parse_range(args.num_vnrs, (200, 200))
    
    # Parse vnode_range
    if args.vnode_range:
        vnode_range = [int(x.strip()) for x in args.vnode_range.split(',')]
    else:
        vnode_range = [2, 4, 6, 8]
    
    # Expand 'all' to all experiments
    if 'all' in args.experiments:
        args.experiments = ['fig6', 'fig7', 'fig8']
    
    print("\n" + "="*70)
    print(" VNE DATASET GENERATOR ".center(70, "="))
    print("="*70)
    print(f"\n Configuration:")
    print(f"   • Experiments: {', '.join(args.experiments)}")
    print(f"   • Output directory: {args.output_dir}")
    print(f"   • Num replicas: {args.num_replicas}")
    print(f"   • Substrate nodes range: {substrate_nodes_range}")
    print(f"   • Num VNRs range: {num_vnrs_range}")
    print(f"   • VNode range: {vnode_range}")
    print(f"   • Base seed: {args.base_seed}")
    print(f"   • Force regeneration: {args.force}")
    print()
    
    # Check if datasets already exist (unless force)
    if not args.force:
        existing = []
        for exp_name in args.experiments:
            exp_dir = os.path.join(args.output_dir, exp_name)
            meta_path = os.path.join(exp_dir, 'metadata.json')
            if os.path.exists(meta_path):
                existing.append(exp_name)
        
        if existing:
            print(f" The following datasets already exist:")
            for exp_name in existing:
                print(f"   • {exp_name}")
            print()
            
            response = input("Overwrite existing datasets? [y/N]: ").strip().lower()
            if response not in ['y', 'yes']:
                print("\n Generation cancelled.")
                print("   Use --force to overwrite without prompting.")
                return
    
    # Create generator
    generator = DatasetGenerator(base_dir=args.output_dir)
    
    # Generate datasets
    generated = {}
    
    for exp_name in args.experiments:
        try:
            if exp_name == 'fig6':
                metadata = generator.generate_fig6_dataset(
                    vnode_range=vnode_range,
                    num_replicas=args.num_replicas,
                    substrate_nodes_range=substrate_nodes_range,
                    num_vnrs_range=num_vnrs_range,
                    base_seed=args.base_seed
                )
                generated[exp_name] = metadata
            
            elif exp_name == 'fig7':
                metadata = generator.generate_fig7_dataset()
                generated[exp_name] = metadata
            
            elif exp_name == 'fig8':
                metadata = generator.generate_fig8_dataset()
                generated[exp_name] = metadata
        
        except Exception as e:
            print(f"\n ERROR generating {exp_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print(" GENERATION SUMMARY ".center(70, "="))
    print("="*70)
    
    if generated:
        print(f"\n Successfully generated {len(generated)} dataset(s):")
        for exp_name in generated:
            exp_dir = os.path.join(args.output_dir, exp_name)
            print(f"\n    {exp_name.upper()}")
            print(f"      Location: {exp_dir}/")
            
            # Show replica info for fig6
            if exp_name == 'fig6' and 'replicas' in generated[exp_name]:
                print(f"      Replicas: {len(generated[exp_name]['replicas'])}")
        
        print("\n" + "="*70)
        print("\n Datasets ready! You can now run experiments:")
        print(f"   python scripts/run_experiments.py --experiments {' '.join(generated.keys())}")
        print()
    else:
        print("\n No datasets were generated.")
        print()


if __name__ == "__main__":
    main()