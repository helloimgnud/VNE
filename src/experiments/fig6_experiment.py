# src/experiments/fig6_experiment.py
"""
Fig6 Experiment: Impact of Virtual Nodes
Varies the number of virtual nodes while keeping domains fixed.
Supports multiple dataset replicas for statistical significance.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from experiments.base_experiment import BaseExperiment
from simulation.simulator import VNRSimulator, BatchedVNRSimulator
from datetime import datetime, timezone, timedelta
from utils.io_utils import load_substrate_from_json, load_vnr_stream_from_json

class Fig6Experiment(BaseExperiment):
    """
    Experiment that varies number of virtual nodes.
    Loads pre-generated datasets (with replicas) and runs algorithms.
    """
    
    def __init__(self, dataset_dir="dataset/fig6", results_dir="results/fig6", run_id=None):
        if run_id is None:
            utc_now = datetime.now(timezone.utc)
            vn_time = utc_now + timedelta(hours=7)
            run_id = vn_time.strftime("%Y%m%d_%H%M%S")
        super().__init__(
            experiment_name="fig6_experiment",
            dataset_dir=dataset_dir,
            results_dir=results_dir,
            run_id=run_id
        )

        self.results_file = os.path.join(
            self.results_dir, 
            f"results_{self.experiment_name}_{self.run_id}.csv"
        )
        
        self.plot_file = os.path.join(
            self.results_dir, 
            f"{self.experiment_name}_{self.run_id}.png"
        )
        
        # Extract experiment parameters from metadata
        self.vnode_range = self.metadata.get('vnode_range', [2, 4, 6, 8])
        self.domain_fixed = self.metadata.get('num_domains', 4)
        self.num_replicas = self.metadata.get('num_replicas', 1)
        self.replicas = self.metadata.get('replicas', [])
        
        print(f" Fig6 Experiment Configuration:")
        print(f"   • Virtual nodes: {self.vnode_range}")
        print(f"   • Fixed domains: {self.domain_fixed}")
        print(f"   • Replicas: {self.num_replicas}")
    
    def run(self, algorithms=None, verbose=True):
        """
        Run experiment for all vnode configurations across all replicas.
        
        Args:
            algorithms: List of algorithm names to test
            verbose: If True, print detailed progress
            
        Returns:
            List of result records
        """
        if algorithms is None:
            algorithms = ['baseline','proposed', 'pso', 'hpso', 'hpso_priority']
        
        print(f"\n{'='*60}")
        print(f" Running {self.experiment_name} (run_id: {self.run_id})")
        print(f" Replicas: {self.num_replicas}")
        print(f"{'='*60}")
        
        all_records = []
        
        # Check if using new replica format or legacy format
        if self.replicas:
            # New format with replicas
            for replica_info in self.replicas:
                replica_id = replica_info['replica_id']
                
                print(f"\n{'='*60}")
                print(f" REPLICA {replica_id + 1}/{self.num_replicas}")
                print(f"   Substrate nodes: {replica_info.get('substrate_nodes', 'N/A')}")
                print(f"   Num VNRs: {replica_info.get('num_vnrs', 'N/A')}")
                print(f"{'='*60}")
                
                # Load substrate for this replica
                substrate = load_substrate_from_json(replica_info['substrate_path'])
                vnr_configs = replica_info['vnr_configs']
                
                # Test each vnode configuration
                for num_vnodes in self.vnode_range:
                    print(f"\n{'-'*60}")
                    print(f" Testing with {num_vnodes} virtual nodes")
                    print(f"{'-'*60}")
                    
                    vnr_path = vnr_configs.get(f"{num_vnodes}nodes")
                    if not vnr_path or not os.path.exists(vnr_path):
                        print(f"   WARNING: VNR stream not found for {num_vnodes} nodes")
                        continue
                    
                    vnr_stream = load_vnr_stream_from_json(vnr_path)
                    
                    # Test each algorithm
                    for algo_name in algorithms:
                        print(f"\n Running {algo_name}...")
                        
                        metrics = self._run_algorithm(substrate, vnr_stream, algo_name)
                        
                        record = {
                            'replica_id': replica_id,
                            'algorithm': algo_name,
                            'num_vnodes': num_vnodes,
                            'num_domains': self.domain_fixed,
                            **metrics
                        }
                        
                        all_records.append(record)
                        
                        if verbose:
                            print(f"   ✓ Acceptance Ratio: {metrics['acceptance_ratio']:.2%}")
                            print(f"   ✓ Avg Cost: {metrics['avg_cost']:.2f}")
                            print(f"   ✓ Avg Revenue: {metrics['avg_revenue']:.2f}")
                            print(f"   ✓ Avg Time: {metrics['avg_execution_time']:.4f}s")
        else:
            # Legacy format (single dataset, no replicas)
            substrate = self.load_substrate()
            
            for num_vnodes in self.vnode_range:
                print(f"\n{'-'*60}")
                print(f" Testing with {num_vnodes} virtual nodes")
                print(f"{'-'*60}")
                
                vnr_path = os.path.join(self.dataset_dir, f"vnr_{num_vnodes}nodes.json")
                vnr_stream = self.load_vnr_stream(vnr_path)
                
                for algo_name in algorithms:
                    print(f"\n Running {algo_name}...")
                    
                    metrics = self._run_algorithm(substrate, vnr_stream, algo_name)
                    
                    record = {
                        'replica_id': 0,
                        'algorithm': algo_name,
                        'num_vnodes': num_vnodes,
                        'num_domains': self.domain_fixed,
                        **metrics
                    }
                    
                    all_records.append(record)
                    
                    if verbose:
                        print(f"   ✓ Acceptance Ratio: {metrics['acceptance_ratio']:.2%}")
                        print(f"   ✓ Avg Cost: {metrics['avg_cost']:.2f}")
                        print(f"   ✓ Avg Revenue: {metrics['avg_revenue']:.2f}")
                        print(f"   ✓ Avg Time: {metrics['avg_execution_time']:.4f}s")
        
        # Save results
        self.save_results(all_records)
        
        print(f"\n{'='*60}")
        print(f" Experiment completed!")
        print(f" Total records: {len(all_records)}")
        print(f"{'='*60}\n")
        
        return all_records
    
    def _run_algorithm(self, substrate, vnr_stream, algo_name):
        """Run a single algorithm on a single VNR stream."""
        if algo_name in ['baseline', 'proposed', 'proposed_KL', 'batch_hpso', 'parallel_hpso_priority']:
            simulator = BatchedVNRSimulator(
                substrate, 
                window_size=10, 
                max_queue_delay=50
            )
            batch_algo = self.get_algorithm_runner(algo_name)
            metrics = simulator.simulate_batched_stream(
                vnr_stream, 
                batch_algo,
                verbose=False
            )
        else:
            simulator = VNRSimulator(substrate)
            algo = self.get_algorithm_runner(algo_name)
            metrics = simulator.simulate_stream(
                vnr_stream,
                algo,
                verbose=False
            )
        return metrics
    
    def plot(self, run_id=None, compare_runs=False):
        """
        Generate plots comparing algorithms across vnode configurations.
        Aggregates across replicas (mean ± std).
        """
        df = self.load_results(run_id if not compare_runs else 'all')
        
        if df is None or df.empty:
            print(" No data to plot.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'Fig6: Impact of Virtual Nodes\n(Experiment: {self.experiment_name}, Run: {self.run_id})',
            fontsize=14, fontweight='bold'
        )
        
        algorithms = df['algorithm'].unique()
        colors = plt.cm.Set2(range(len(algorithms)))
        
        # Aggregate across replicas if multiple exist
        has_replicas = 'replica_id' in df.columns and df['replica_id'].nunique() > 1
        
        if has_replicas:
            self._plot_metrics_with_errorbars(axes, df, algorithms, colors)
        else:
            self._plot_metrics(axes, df, algorithms, colors)
        
        plt.tight_layout()
        
        if run_id and run_id != self.run_id:
            current_plot_file = os.path.join(self.plots_dir, f"{self.experiment_name}_{run_id}.png")
        else:
            current_plot_file = self.plot_file
        
        if compare_runs:
            current_plot_file = current_plot_file.replace('.png', '_compare.png')
        os.makedirs(os.path.dirname(current_plot_file), exist_ok=True)
        plt.savefig(current_plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {current_plot_file}")
        plt.show()
    
    def _plot_metrics_with_errorbars(self, axes, df, algorithms, colors):
        """Plot metrics as vertical bar charts (mean only, no std)."""
        
        metrics_config = [
            ('acceptance_ratio', 'Acceptance Ratio', axes[0, 0]),
            ('avg_cost', 'Average Cost', axes[0, 1]),
            ('cost_revenue_ratio', 'Cost/Revenue Ratio', axes[1, 0]),
            ('avg_execution_time', 'Average Execution Time (s)', axes[1, 1])
        ]
        
        vnode_values = sorted(df['num_vnodes'].unique())
        n_algorithms = len(algorithms)
        bar_width = 0.8 / n_algorithms
        
        for metric_name, ylabel, ax in metrics_config:
            x = np.arange(len(vnode_values))
            
            for idx, algo in enumerate(algorithms):
                algo_df = df[df['algorithm'] == algo]
                
                # Group by num_vnodes and compute mean only
                grouped = algo_df.groupby('num_vnodes')[metric_name].mean()
                means = [grouped.get(v, 0) for v in vnode_values]
                
                # Calculate bar position offset
                offset = (idx - n_algorithms / 2 + 0.5) * bar_width
                
                bars = ax.bar(
                    x + offset, 
                    means,
                    width=bar_width,
                    color=colors[idx],
                    label=algo
                )
                
                # Add value labels on top of bars
                for bar, val in zip(bars, means):
                    if val > 0:
                        if metric_name == 'acceptance_ratio':
                            label_text = f'{val:.1%}'
                        elif metric_name == 'avg_execution_time':
                            label_text = f'{val:.4f}'
                        else:
                            label_text = f'{val:.2f}'
                        ax.annotate(
                            label_text,
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3),
                            textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=6, rotation=90
                        )
            
            ax.set_xlabel('Number of Virtual Nodes')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{ylabel} vs Virtual Nodes')
            ax.set_xticks(x)
            ax.set_xticklabels(vnode_values)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

    def _plot_metrics(self, axes, df, algorithms, colors, 
                     linestyle='-', label_suffix=''):
        """Helper to plot metrics as vertical bar charts (no error bars)."""
        
        metrics_config = [
            ('acceptance_ratio', 'Acceptance Ratio', axes[0, 0]),
            ('avg_cost', 'Average Cost', axes[0, 1]),
            ('cost_revenue_ratio', 'Cost/Revenue Ratio', axes[1, 0]),
            ('avg_execution_time', 'Average Execution Time (s)', axes[1, 1])
        ]
        
        vnode_values = sorted(df['num_vnodes'].unique())
        n_algorithms = len(algorithms)
        bar_width = 0.8 / n_algorithms
        
        for metric_name, ylabel, ax in metrics_config:
            x = np.arange(len(vnode_values))
            
            for idx, algo in enumerate(algorithms):
                algo_df = df[df['algorithm'] == algo].sort_values('num_vnodes')
                means = [algo_df[algo_df['num_vnodes'] == v][metric_name].values[0] 
                        if len(algo_df[algo_df['num_vnodes'] == v]) > 0 else 0 
                        for v in vnode_values]
                
                # Calculate bar position offset
                offset = (idx - n_algorithms / 2 + 0.5) * bar_width
                
                bars = ax.bar(
                    x + offset, 
                    means,
                    width=bar_width,
                    color=colors[idx],
                    label=f'{algo}{label_suffix}'
                )
                
                # Add value labels on top of bars
                for bar, val in zip(bars, means):
                    if val > 0:
                        if metric_name == 'acceptance_ratio':
                            label_text = f'{val:.1%}'
                        elif metric_name == 'avg_execution_time':
                            label_text = f'{val:.4f}'
                        else:
                            label_text = f'{val:.2f}'
                        ax.annotate(
                            label_text,
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3),
                            textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=6, rotation=90
                        )
            
            ax.set_xlabel('Number of Virtual Nodes')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{ylabel} vs Virtual Nodes')
            ax.set_xticks(x)
            ax.set_xticklabels(vnode_values)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')