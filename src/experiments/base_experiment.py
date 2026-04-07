# src/experiments/base_experiments.py
"""
Base Experiment Class
Provides common functionality for all experiments.
Focuses on loading data and running experiments - not generating data.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from datetime import datetime

RESULTS_DIR = "results"


class BaseExperiment(ABC):
    """
    Base class for VNE experiments.
    
    Experiments should:
    1. Load pre-generated datasets
    2. Run algorithms on datasets
    3. Collect and save metrics
    4. Generate plots
    """
    
    def __init__(self, experiment_name, dataset_dir, results_dir, run_id=None):
        """
        Initialize experiment.
        
        Args:
            experiment_name: Name of experiment (e.g., "fig6")
            dataset_dir: Directory containing pre-generated dataset
            run_id: Optional run ID for tracking multiple runs
        """
        self.experiment_name = experiment_name
        self.dataset_dir = dataset_dir
        self.results_dir = results_dir
        
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id
        
        # Result file paths
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.csv_file = os.path.join(RESULTS_DIR, f"{experiment_name}.csv")
        self.plot_file = os.path.join(RESULTS_DIR, f"{experiment_name}.png")
        
        # Load dataset metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """Load dataset metadata."""
        meta_path = os.path.join(self.dataset_dir, "metadata.json")
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Dataset metadata not found: {meta_path}\n"
                f"Please generate dataset first using: python scripts/generate_datasets.py"
            )
        
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        
        print(f"✓ Loaded metadata: {self.experiment_name}")
        return metadata
    
    def load_substrate(self, path=None):
        """
        Load substrate network from JSON.
        
        Args:
            path: Optional custom path. If None, uses metadata path.
        """
        from utils.io_utils import load_substrate_from_json
        
        if path is None:
            path = self.metadata.get('substrate_path')
        
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"Substrate file not found: {path}")
        
        print(f"✓ Loading substrate from {path}")
        return load_substrate_from_json(path)
    
    def load_vnr_stream(self, path=None):
        """
        Load VNR stream from JSON.
        
        Args:
            path: Optional custom path. If None, uses metadata path.
        """
        from utils.io_utils import load_vnr_stream_from_json
        
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"VNR stream file not found: {path}")
        
        print(f"✓ Loading VNR stream from {path}")
        return load_vnr_stream_from_json(path)
    
    @abstractmethod
    def run(self):
        """
        Run the experiment.
        Must be implemented by subclasses.
        
        Returns:
            List of result records
        """
        pass
    
    @abstractmethod
    def plot(self, run_id=None, compare_runs=False):
        """
        Generate plots from results.
        Must be implemented by subclasses.
        
        Args:
            run_id: Specific run ID to plot, or None for latest
            compare_runs: If True, plot multiple runs for comparison
        """
        pass
    
    def save_results(self, records, append=True):
        """
        Save experiment results to CSV.
        
        Args:
            records: List of result dictionaries
            append: If True, append to existing file
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for record in records:
            record['run_id'] = self.run_id
            record['timestamp'] = timestamp
        
        df_new = pd.DataFrame(records)
        
        if append and os.path.exists(self.csv_file):
            df_old = pd.read_csv(self.csv_file)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined.to_csv(self.csv_file, index=False)
            print(f"✓ Appended {len(records)} records to {self.csv_file} "
                  f"(total: {len(df_combined)} records)")
        else:
            df_new.to_csv(self.csv_file, index=False)
            print(f"✓ Saved {len(records)} records to {self.csv_file}")
    
    def load_results(self, run_id=None):
        """
        Load experiment results from CSV.
        
        Args:
            run_id: Specific run ID to load, 'all' for all runs, or None for latest
            
        Returns:
            DataFrame with results
        """
        if not os.path.exists(self.csv_file):
            print(f" CSV file '{self.csv_file}' not found.")
            return None
        
        df = pd.read_csv(self.csv_file)
        
        if df.empty:
            return df
        
        if run_id == 'all':
            return df
        elif run_id is None:
            if 'run_id' in df.columns:
                latest_run = df['run_id'].max()
                return df[df['run_id'] == latest_run]
            else:
                return df
        else:
            if 'run_id' in df.columns:
                return df[df['run_id'] == run_id]
            else:
                return df
    
    def list_runs(self):
        """List all available runs for this experiment."""
        if not os.path.exists(self.csv_file):
            print(f" No data found for {self.experiment_name}")
            return []
        
        df = pd.read_csv(self.csv_file)
        if 'run_id' not in df.columns:
            return []
        
        runs = df.groupby('run_id').agg({
            'timestamp': 'first'
        }).reset_index()
        
        print(f"\n Available runs for {self.experiment_name}:")
        for _, row in runs.iterrows():
            print(f"   • {row['run_id']} (created: {row['timestamp']})")
        
        return runs['run_id'].tolist()
    
    def get_algorithm_runner(self, algo_name):
        """
        Get algorithm runner function.
        
        Args:
            algo_name: Name of algorithm ('baseline', 'd_vine_sp', 'pso', 'hpso')
            
        Returns:
            Function(substrate, vnr) -> (mapping, link_paths) or None
        """
        from algorithms.baseline import baseline_embed_batch
        from algorithms.d_vine_sp import d_vine_sp_embed
        from algorithms.pso import pso_embed
        # from algorithms.hpso import hpso_embed
        from algorithms.fast_hpso import hpso_embed
        from algorithms.parallel_mt_vne import embed_batch
        from algorithms.batch_hpso import batch_hpso_embed
        from algorithms.discrete_pso import pso_embed as discrete_pso_embed
        from algorithms.discrete_hpso import hpso_embed as discrete_hpso_embed
        from algorithms.hpso_priority import hpso_embed as hpso_priority_embed
        from algorithms.parallel_hpso_priority import embed_batch as parallel_hpso_priority_embed
        algorithm_map = {
            'baseline': baseline_embed_batch,
            'd_vine_sp': d_vine_sp_embed,
            'pso': pso_embed,
            'hpso': hpso_embed,
            "proposed": embed_batch,
            "batch_hpso": batch_hpso_embed,
            "discrete_pso": discrete_pso_embed,
            "discrete_hpso": discrete_hpso_embed,
            "hpso_priority": hpso_priority_embed,
            "parallel_hpso_priority": parallel_hpso_priority_embed
        }
        
        if algo_name not in algorithm_map:
            raise ValueError(f"Unknown algorithm: {algo_name}")
        
        return algorithm_map[algo_name]