# src/generators/dataset_generator.py
"""
Dataset Generator Module
Orchestrates generation of all datasets for experiments.
Separated from experiment logic for better modularity.
"""

import os
import json
from generators.generator import generate_substrate, generate_vnr_stream_v2


class DatasetGenerator:
    """
    Centralized dataset generation for all experiments.
    Each experiment type has its own generation method.
    """
    
    def __init__(self, base_dir="dataset"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def _save_metadata(self, experiment_name, metadata):
        """Save dataset metadata for reproducibility."""
        meta_path = os.path.join(self.base_dir, experiment_name, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved: {meta_path}")
    
    # ============================================================
    # Fig6: Impact of Virtual Nodes (varying vnodes)
    # ============================================================
    
    def generate_fig6_dataset(self, 
                             vnode_range=[2, 4, 6, 8],
                             num_domains=4,
                             # Randomizable ranges (can be single value or tuple)
                             num_vnrs_range=(200, 200),
                             substrate_nodes_range=(100, 100),
                             # Substrate parameters
                             p_intra=0.5,
                             p_inter=0.1,
                             substrate_cpu_range=(100, 300),
                             substrate_bw_range=(1000, 3000),
                             node_cost_range=(1, 10),
                             inter_domain_bw_cost=(5, 15),
                             # VNR parameters
                             vnr_cpu_range=(50, 200),
                             vnr_bw_range=(500, 2000),
                             max_lifetime=1000,
                             avg_inter_arrival=1.0,
                             # Replica settings
                             num_replicas=10,
                             base_seed=42):
        """
        Generate dataset for Fig6 experiment with multiple replicas.
        
        Args:
            vnode_range: List of virtual node counts to test
            num_domains: Number of domains (fixed)
            num_vnrs_range: (min, max) range for number of VNRs (random per replica)
            substrate_nodes_range: (min, max) range for substrate nodes (random per replica)
            
            # Substrate parameters
            p_intra: Probability of intra-domain links
            p_inter: Probability of inter-domain links
            substrate_cpu_range: (min, max) CPU for substrate nodes
            substrate_bw_range: (min, max) bandwidth for substrate links
            node_cost_range: (min, max) cost per substrate node
            inter_domain_bw_cost: (min, max) cost for inter-domain bandwidth
            
            # VNR parameters
            vnr_cpu_range: (min, max) CPU range for virtual nodes
            vnr_bw_range: (min, max) bandwidth range for virtual links
            max_lifetime: Maximum VNR lifetime
            avg_inter_arrival: Average inter-arrival time between VNRs
            
            # Replica settings
            num_replicas: Number of dataset replicas to generate
            base_seed: Base random seed (each replica uses base_seed + i * 1000)
        """
        import random
        
        exp_dir = os.path.join(self.base_dir, "fig6")
        os.makedirs(exp_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print(f"GENERATING FIG6 DATASET ({num_replicas} replicas)")
        print("="*60)
        
        replicas = []
        
        for replica_id in range(num_replicas):
            replica_seed = base_seed + replica_id * 1000
            rng = random.Random(replica_seed)
            
            # Sample random values from ranges
            num_nodes_total = rng.randint(substrate_nodes_range[0], substrate_nodes_range[1])
            num_vnrs = rng.randint(num_vnrs_range[0], num_vnrs_range[1])
            
            replica_dir = os.path.join(exp_dir, f"replica_{replica_id}")
            os.makedirs(replica_dir, exist_ok=True)
            
            print(f"\n[Replica {replica_id + 1}/{num_replicas}] "
                  f"nodes={num_nodes_total}, vnrs={num_vnrs}, seed={replica_seed}")
            
            # Generate substrate
            substrate_path = os.path.join(replica_dir, "substrate.json")
            generate_substrate(
                num_domains=num_domains,
                num_nodes_total=num_nodes_total,
                p_intra=p_intra,
                p_inter=p_inter,
                cpu_range=substrate_cpu_range,
                bw_range=substrate_bw_range,
                node_cost_range=node_cost_range,
                inter_domain_bw_cost=inter_domain_bw_cost,
                seed=replica_seed,
                export_path=substrate_path
            )
            
            # Generate VNR streams for each vnode configuration
            vnr_configs = {}
            for num_vnodes in vnode_range:
                vnr_path = os.path.join(replica_dir, f"vnr_{num_vnodes}nodes.json")
                
                generate_vnr_stream_v2(
                    num_vnrs=num_vnrs,
                    num_domains=num_domains,
                    min_vnodes=num_vnodes,
                    max_vnodes=num_vnodes+1,
                    cpu_range=vnr_cpu_range,
                    bw_range=vnr_bw_range,
                    max_lifetime=max_lifetime,
                    avg_inter_arrival=avg_inter_arrival,
                    export_path=vnr_path,
                    seed=replica_seed + num_vnodes
                )
                
                vnr_configs[f"{num_vnodes}nodes"] = vnr_path
            
            replica_info = {
                "replica_id": replica_id,
                "seed": replica_seed,
                "substrate_nodes": num_nodes_total,
                "num_vnrs": num_vnrs,
                "substrate_path": substrate_path,
                "vnr_configs": vnr_configs
            }
            replicas.append(replica_info)
        
        # Save global metadata
        metadata = {
            "experiment": "fig6",
            "description": "Impact of Virtual Nodes",
            "vnode_range": vnode_range,
            "num_domains": num_domains,
            "num_replicas": num_replicas,
            "substrate_nodes_range": list(substrate_nodes_range),
            "num_vnrs_range": list(num_vnrs_range),
            "base_seed": base_seed,
            "replicas": replicas
        }
        
        self._save_metadata("fig6", metadata)
        
        print("\n" + "="*60)
        print(f"✓ FIG6 DATASET GENERATION COMPLETE ({num_replicas} replicas)")
        print(f"✓ Location: {exp_dir}")
        print("="*60 + "\n")
        
        return metadata
    
    # ============================================================
    # Fig7: Embedding Overhead vs Domains (varying domains)
    # ============================================================
    
    def generate_fig7_dataset(self,
                             domain_range=[4, 6, 8],
                             num_vnodes=4,
                             num_vnrs=300,
                             substrate_seed_base=200,
                             vnr_seed_base=300):
        """
        Generate dataset for Fig7 experiment.
        
        Args:
            domain_range: List of domain counts to test
            num_vnodes: Number of virtual nodes (fixed)
            num_vnrs: Number of VNRs per configuration
            substrate_seed_base: Base seed for substrates
            vnr_seed_base: Base seed for VNR streams
        """
        exp_dir = os.path.join(self.base_dir, "fig7")
        os.makedirs(exp_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("GENERATING FIG7 DATASET")
        print("="*60)
        
        substrate_configs = {}
        vnr_configs = {}
        
        for num_domains in domain_range:
            print(f"\n[{domain_range.index(num_domains)+1}/{len(domain_range)}] Generating config: {num_domains} domains")
            
            # Generate substrate for this domain configuration
            substrate_path = os.path.join(exp_dir, f"substrate_{num_domains}domains.json")
            
            print(f"  → Substrate ({num_domains} domains)")
            generate_substrate(
                num_domains=num_domains,
                num_nodes_total=100,
                p_intra=0.5,
                p_inter=0.1,
                cpu_range=(100, 300),
                bw_range=(1000, 3000),
                node_cost_range=(1, 10),
                inter_domain_bw_cost=(5, 15),
                seed=substrate_seed_base + num_domains,
                export_path=substrate_path
            )
            
            substrate_configs[f"{num_domains}domains"] = substrate_path
            
            # Generate VNR stream for this configuration
            vnr_path = os.path.join(exp_dir, f"vnr_{num_domains}domains.json")
            
            print(f"  → VNR stream ({num_vnodes} nodes, {num_domains} domains)")
            generate_vnr_stream_v2(
                num_vnrs=num_vnrs,
                num_domains=num_domains,
                max_lifetime=300,
                avg_inter_arrival=1.0,
                export_path=vnr_path,
                seed=vnr_seed_base + num_domains
            )
            
            vnr_configs[f"{num_domains}domains"] = vnr_path
        
        # Save metadata
        metadata = {
            "experiment": "fig7",
            "description": "Embedding Overhead vs Number of Domains",
            "domain_range": domain_range,
            "num_vnodes": num_vnodes,
            "num_vnrs": num_vnrs,
            "substrate_configs": substrate_configs,
            "vnr_configs": vnr_configs,
            "seeds": {
                "substrate_base": substrate_seed_base,
                "vnr_base": vnr_seed_base
            }
        }
        
        self._save_metadata("fig7", metadata)
        
        print("\n" + "="*60)
        print("✓ FIG7 DATASET GENERATION COMPLETE")
        print(f"✓ Location: {exp_dir}")
        print("="*60 + "\n")
        
        return metadata
    
    # ============================================================
    # Fig8: VNR Acceptance Rate (single configuration, online)
    # ============================================================
    
    def generate_fig8_dataset(self,
                             num_vnodes=4,
                             num_domains=4,
                             num_vnrs=300,
                             substrate_seed=400,
                             vnr_seed=500):
        """
        Generate dataset for Fig8 experiment.
        
        Args:
            num_vnodes: Number of virtual nodes
            num_domains: Number of domains
            num_vnrs: Number of VNRs in stream
            substrate_seed: Random seed for substrate
            vnr_seed: Random seed for VNR stream
        """
        exp_dir = os.path.join(self.base_dir, "fig8")
        os.makedirs(exp_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("GENERATING FIG8 DATASET")
        print("="*60)
        
        # Generate substrate
        substrate_path = os.path.join(exp_dir, "substrate.json")
        print("\n[1/2] Generating substrate network...")
        
        generate_substrate(
            num_domains=num_domains,
            num_nodes_total=100,
            p_intra=0.5,
            p_inter=0.1,
            cpu_range=(100, 300),
            bw_range=(1000, 3000),
            node_cost_range=(1, 10),
            inter_domain_bw_cost=(5, 15),
            seed=substrate_seed,
            export_path=substrate_path
        )
        
        # Generate VNR stream
        vnr_path = os.path.join(exp_dir, "vnr_stream.json")
        print("\n[2/2] Generating VNR stream...")
        
        generate_vnr_stream_v2(
            num_vnrs=num_vnrs,
            num_domains=num_domains,
            max_lifetime=300,
            avg_inter_arrival=1.0,
            export_path=vnr_path,
            seed=vnr_seed
        )
        
        # Save metadata
        metadata = {
            "experiment": "fig8",
            "description": "VNR Acceptance Rate Over Time",
            "num_vnodes": num_vnodes,
            "num_domains": num_domains,
            "num_vnrs": num_vnrs,
            "substrate_path": substrate_path,
            "vnr_path": vnr_path,
            "seeds": {
                "substrate": substrate_seed,
                "vnr": vnr_seed
            }
        }
        
        self._save_metadata("fig8", metadata)
        
        print("\n" + "="*60)
        print("✓ FIG8 DATASET GENERATION COMPLETE")
        print(f"✓ Location: {exp_dir}")
        print("="*60 + "\n")
        
        return metadata
    
    # ============================================================
    # Generate All Datasets
    # ============================================================
    
    def generate_all(self):
        """Generate all experiment datasets."""
        print("\n" + "="*70)
        print(" GENERATING ALL EXPERIMENT DATASETS ".center(70, "="))
        print("="*70)
        
        results = {}
        
        # Fig6
        results['fig6'] = self.generate_fig6_dataset()
        
        # Fig7
        results['fig7'] = self.generate_fig7_dataset()
        
        # Fig8
        results['fig8'] = self.generate_fig8_dataset()
        
        print("\n" + "="*70)
        print(" ALL DATASETS GENERATED SUCCESSFULLY ".center(70, "="))
        print("="*70)
        print(f"\n Datasets location: {self.base_dir}/")
        print(" Ready for experiments!\n")
        
        return results


def main():
    """Command-line interface for dataset generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate datasets for VNE experiments"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["fig6", "fig7", "fig8", "all"],
        default=["all"],
        help="Which experiments to generate data for"
    )
    parser.add_argument(
        "--output-dir",
        default="dataset",
        help="Output directory for datasets"
    )
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(base_dir=args.output_dir)
    
    if "all" in args.experiments:
        generator.generate_all()
    else:
        for exp in args.experiments:
            if exp == "fig6":
                generator.generate_fig6_dataset()
            elif exp == "fig7":
                generator.generate_fig7_dataset()
            elif exp == "fig8":
                generator.generate_fig8_dataset()


if __name__ == "__main__":
    main()