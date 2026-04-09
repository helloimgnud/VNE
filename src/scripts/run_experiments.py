# scripts/run_experiments.py
"""
Experiment Runner Script
Runs VNE experiments using pre-generated datasets.
Separated from dataset generation for clean workflow.
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.experiments.fig6_experiment import Fig6Experiment
# Import other experiments as they are created
# from experiments.fig7_experiment import Fig7Experiment
# from experiments.fig8_experiment import Fig8Experiment


def run_fig6(dataset_dir="dataset/fig6", run_id=None, algorithms=None, num_runs=3, plot=True):
    """Run Fig6 experiment."""
    print("\n" + "="*70)
    print(" RUNNING FIG6 EXPERIMENT ".center(70, "="))
    print("="*70)
    
    experiment = Fig6Experiment(dataset_dir=dataset_dir, run_id=run_id)
    results = experiment.run(algorithms=algorithms, num_runs=num_runs)
    
    if plot:
        experiment.plot()
    
    return experiment, results


def run_fig7(dataset_dir="dataset/fig7", run_id=None, algorithms=None, plot=True):
    """Run Fig7 experiment."""
    print("\n" + "="*70)
    print(" RUNNING FIG7 EXPERIMENT ".center(70, "="))
    print("="*70)
    
    print(" Fig7 experiment not yet implemented in new architecture")
    # experiment = Fig7Experiment(dataset_dir=dataset_dir, run_id=run_id)
    # results = experiment.run(algorithms=algorithms)
    # if plot:
    #     experiment.plot()
    # return experiment, results


def run_fig8(dataset_dir="dataset/fig8", run_id=None, algorithms=None, plot=True):
    """Run Fig8 experiment."""
    print("\n" + "="*70)
    print(" RUNNING FIG8 EXPERIMENT ".center(70, "="))
    print("="*70)
    
    print(" Fig8 experiment not yet implemented in new architecture")
    # experiment = Fig8Experiment(dataset_dir=dataset_dir, run_id=run_id)
    # results = experiment.run(algorithms=algorithms)
    # if plot:
    #     experiment.plot()
    # return experiment, results


def list_runs(experiment_name):
    """List all runs for an experiment."""
    dataset_dirs = {
        'fig6': 'dataset/fig6',
        'fig7': 'dataset/fig7',
        'fig8': 'dataset/fig8'
    }
    
    if experiment_name == 'fig6':
        exp = Fig6Experiment(dataset_dir=dataset_dirs['fig6'])
        exp.list_runs()
    elif experiment_name == 'fig7':
        print(" Fig7 not yet implemented")
    elif experiment_name == 'fig8':
        print(" Fig8 not yet implemented")


def plot_only(experiment_name, run_id=None, compare_runs=False):
    """Plot results without running experiments."""
    dataset_dirs = {
        'fig6': 'dataset/fig6',
        'fig7': 'dataset/fig7',
        'fig8': 'dataset/fig8'
    }
    
    if experiment_name == 'fig6':
        exp = Fig6Experiment(dataset_dir=dataset_dirs['fig6'])
        exp.plot(run_id=run_id, compare_runs=compare_runs)
    elif experiment_name == 'fig7':
        print(" Fig7 not yet implemented")
    elif experiment_name == 'fig8':
        print(" Fig8 not yet implemented")


def main():
    parser = argparse.ArgumentParser(
        description="Run VNE experiments with pre-generated datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python scripts/run_experiments.py --experiments all
  
  # Run specific experiment
  python scripts/run_experiments.py --experiments fig6
  
  # Run with custom algorithms
  python scripts/run_experiments.py --experiments fig6 --algorithms baseline pso
  
  # Plot only (no execution)
  python scripts/run_experiments.py --plot-only --experiments fig6
  
  # List available runs
  python scripts/run_experiments.py --list-runs --experiments fig6
  
  # Compare multiple runs
  python scripts/run_experiments.py --plot-only --compare-runs --experiments fig6
        """
    )
    
    parser.add_argument(
        '--experiments',
        nargs='+',
        choices=['fig6', 'fig7', 'fig8', 'all'],
        default=['fig6'],
        help='Which experiments to run'
    )
    
    parser.add_argument(
        '--algorithms',
        nargs='+',
        choices=['baseline', 'd_vine_sp', 'pso', 'hpso', 'mp-pva', 'proposed', 'proposed_KL'],
        default=None,
        help='Algorithms to test (default: baseline d_vine_sp pso hpso)'
    )
    
    parser.add_argument(
        '--dataset-dir',
        default='dataset',
        help='Base directory for datasets'
    )
    
    parser.add_argument(
        '--num-runs',
        type=int,
        default=3,
        help='Number of runs per algorithm on the same dataset to average out randomness (default: 3)'
    )
    
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Custom run ID (default: auto-generated timestamp)'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip plotting after running'
    )
    
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='Only generate plots from existing results'
    )
    
    parser.add_argument(
        '--list-runs',
        action='store_true',
        help='List available runs for experiments'
    )
    
    parser.add_argument(
        '--compare-runs',
        action='store_true',
        help='Create comparison plots across multiple runs (use with --plot-only)'
    )
    
    args = parser.parse_args()
    
    # Expand 'all' to all experiments
    if 'all' in args.experiments:
        args.experiments = ['fig6', 'fig7', 'fig8']
    
    # List runs mode
    if args.list_runs:
        for exp_name in args.experiments:
            print(f"\n{'='*60}")
            print(f" Runs for {exp_name.upper()} ".center(60, "="))
            print(f"{'='*60}")
            list_runs(exp_name)
        return
    
    # Plot-only mode
    if args.plot_only:
        for exp_name in args.experiments:
            plot_only(exp_name, run_id=args.run_id, compare_runs=args.compare_runs)
        return
    
    # Check if datasets exist
    for exp_name in args.experiments:
        dataset_path = os.path.join(args.dataset_dir, exp_name)
        if not os.path.exists(dataset_path):
            print(f"\n ERROR: Dataset not found for {exp_name}")
            print(f"   Expected location: {dataset_path}")
            print(f"   Please generate datasets first:")
            print(f"   python scripts/generate_datasets.py --experiments {exp_name}")
            return
    
    # Run experiments
    print("\n" + "="*70)
    print(" VNE EXPERIMENT RUNNER ".center(70, "="))
    print("="*70)
    print(f"\n Configuration:")
    print(f"   • Experiments: {', '.join(args.experiments)}")
    print(f"   • Algorithms: {args.algorithms or 'default'}")
    print(f"   • Dataset dir: {args.dataset_dir}")
    print(f"   • Num runs: {args.num_runs}")
    print(f"   • Run ID: {args.run_id or 'auto-generated'}")
    print(f"   • Plot: {not args.no_plot}")
    
    results = {}
    
    for exp_name in args.experiments:
        dataset_path = os.path.join(args.dataset_dir, exp_name)
        
        if exp_name == 'fig6':
            exp, res = run_fig6(
                dataset_dir=dataset_path,
                run_id=args.run_id,
                algorithms=args.algorithms,
                num_runs=args.num_runs,
                plot=not args.no_plot
            )
            results[exp_name] = (exp, res)
        
        elif exp_name == 'fig7':
            run_fig7(
                dataset_dir=dataset_path,
                run_id=args.run_id,
                algorithms=args.algorithms,
                plot=not args.no_plot
            )
        
        elif exp_name == 'fig8':
            run_fig8(
                dataset_dir=dataset_path,
                run_id=args.run_id,
                algorithms=args.algorithms,
                plot=not args.no_plot
            )
    
    print("\n" + "="*70)
    print(" ALL EXPERIMENTS COMPLETED ".center(70, "="))
    print("="*70)
    print(f"\n Results saved in: results/")
    print(f" Plots saved in: results/")
    print()


if __name__ == "__main__":
    main()