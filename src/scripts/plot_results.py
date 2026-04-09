# scripts/plot_results.py
"""
Standalone script to plot experiment results from CSV files.
Allows plotting specific results based on run_id by filtering the consolidated CSV.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_csv_path(experiment_name, results_dir="results"):
    """Get the consolidated CSV path for an experiment."""
    csv_path = Path(results_dir) / f"{experiment_name}_experiment.csv"
    if csv_path.exists():
        return str(csv_path)
    
    # Try alternative path formats
    alt_path = Path(results_dir) / experiment_name / f"{experiment_name}_experiment.csv"
    if alt_path.exists():
        return str(alt_path)
    
    return None


def list_available_runs(results_dir="results", experiment_name=None):
    """List all available run IDs in the results directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory '{results_dir}' not found.")
        return {}
    
    runs = {}
    
    # Find all experiment CSV files
    for csv_file in results_path.glob("*_experiment.csv"):
        exp_name = csv_file.stem.replace("_experiment", "")
        if experiment_name and exp_name != experiment_name:
            continue
        
        try:
            df = pd.read_csv(csv_file)
            if 'run_id' in df.columns:
                unique_runs = df['run_id'].dropna().unique()
                runs[exp_name] = {
                    'path': str(csv_file),
                    'run_ids': sorted(unique_runs, reverse=True)
                }
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    return runs


def load_results(csv_path, run_id=None):
    """
    Load results from a CSV file, optionally filtering by run_id.
    
    Args:
        csv_path: Path to the CSV file
        run_id: Optional run_id to filter by
        
    Returns:
        DataFrame with results (filtered if run_id provided)
    """
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    if run_id and 'run_id' in df.columns:
        # Filter by run_id
        df = df[df['run_id'] == run_id]
        if df.empty:
            print(f"No data found for run_id: {run_id}")
            return None
    
    return df


def plot_bar_chart(df, metric, title=None, output_path=None, show_values=True):
    """
    Plot a bar chart for a specific metric.
    
    Args:
        df: DataFrame with results
        metric: Column name to plot
        title: Optional title for the plot
        output_path: Optional path to save the figure
        show_values: Whether to show value labels on bars
    """
    # Determine x-axis variable (num_vnodes or num_domains)
    if 'num_vnodes' in df.columns:
        x_var = 'num_vnodes'
        x_label = 'Number of Virtual Nodes'
    elif 'num_domains' in df.columns:
        x_var = 'num_domains'
        x_label = 'Number of Domains'
    else:
        print("Cannot determine x-axis variable.")
        return
    
    algorithms = df['algorithm'].unique()
    x_values = sorted(df[x_var].unique())
    n_algorithms = len(algorithms)
    bar_width = 0.8 / n_algorithms
    colors = plt.cm.Set2(range(n_algorithms))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(x_values))
    
    for idx, algo in enumerate(algorithms):
        algo_df = df[df['algorithm'] == algo]
        
        # Aggregate if multiple replicas
        if 'replica_id' in df.columns and df['replica_id'].nunique() > 1:
            grouped = algo_df.groupby(x_var)[metric].mean()
        else:
            grouped = algo_df.set_index(x_var)[metric]
        
        means = [grouped.get(v, 0) for v in x_values]
        offset = (idx - n_algorithms / 2 + 0.5) * bar_width
        
        bars = ax.bar(
            x + offset,
            means,
            width=bar_width,
            color=colors[idx],
            label=algo
        )
        
        # Add value labels on top of bars
        if show_values:
            for bar, val in zip(bars, means):
                if val > 0:
                    if metric == 'acceptance_ratio':
                        label_text = f'{val:.1%}'
                    elif metric == 'avg_execution_time':
                        label_text = f'{val:.4f}'
                    else:
                        label_text = f'{val:.2f}'
                    ax.annotate(
                        label_text,
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=7, rotation=90
                    )
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title or f'{metric.replace("_", " ").title()} Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(x_values)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_path}")
    
    plt.show()


def plot_all_metrics(df, title_prefix="", output_dir=None, run_id=None):
    """
    Plot all standard metrics in a 2x2 grid.
    
    Args:
        df: DataFrame with results
        title_prefix: Prefix for the plot title
        output_dir: Optional directory to save the figure
        run_id: Run ID for naming the output file
    """
    # Determine x-axis variable
    if 'num_vnodes' in df.columns:
        x_var = 'num_vnodes'
        x_label = 'Number of Virtual Nodes'
    elif 'num_domains' in df.columns:
        x_var = 'num_domains'
        x_label = 'Number of Domains'
    else:
        print("Cannot determine x-axis variable.")
        return
    
    metrics_config = [
        ('acceptance_ratio', 'Acceptance Ratio'),
        ('avg_cost', 'Average Cost'),
        ('cost_revenue_ratio', 'Cost/Revenue Ratio'),
        ('avg_execution_time', 'Average Execution Time (s)')
    ]
    
    algorithms = df['algorithm'].unique()
    x_values = sorted(df[x_var].unique())
    n_algorithms = len(algorithms)
    bar_width = 0.8 / n_algorithms
    colors = plt.cm.Set2(range(n_algorithms))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'{title_prefix}Experiment Results' + (f' (Run: {run_id})' if run_id else ''),
        fontsize=14, fontweight='bold'
    )
    
    for (metric_name, ylabel), ax in zip(metrics_config, axes.flatten()):
        x = np.arange(len(x_values))
        
        for idx, algo in enumerate(algorithms):
            algo_df = df[df['algorithm'] == algo]
            
            # Aggregate if multiple replicas
            if 'replica_id' in df.columns and df['replica_id'].nunique() > 1:
                grouped = algo_df.groupby(x_var)[metric_name].mean()
            else:
                grouped = algo_df.set_index(x_var)[metric_name]
            
            means = [grouped.get(v, 0) for v in x_values]
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
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} vs {x_label.split()[-1]}')
        ax.set_xticks(x)
        ax.set_xticklabels(x_values)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = os.path.join(output_dir, f"plot_{run_id or 'results'}.png")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_path}")
    
    plt.show()


def compare_runs(csv_path, run_ids, labels=None, metric='acceptance_ratio', output_path=None):
    """
    Compare multiple runs from the same CSV file on a single chart.
    
    Args:
        csv_path: Path to CSV file
        run_ids: List of run_ids to compare
        labels: Optional labels for each run
        metric: Metric to compare
        output_path: Optional path to save the figure
    """
    if labels is None:
        labels = run_ids
    
    df_full = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    all_data = []
    for run_id, label in zip(run_ids, labels):
        df = df_full[df_full['run_id'] == run_id].copy()
        if not df.empty:
            df['run_label'] = label
            all_data.append(df)
    
    if not all_data:
        print("No data to compare.")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Determine x-axis variable
    if 'num_vnodes' in combined_df.columns:
        x_var = 'num_vnodes'
        x_label = 'Number of Virtual Nodes'
    elif 'num_domains' in combined_df.columns:
        x_var = 'num_domains'
        x_label = 'Number of Domains'
    else:
        print("Cannot determine x-axis variable.")
        return
    
    algorithms = combined_df['algorithm'].unique()
    runs = combined_df['run_label'].unique()
    x_values = sorted(combined_df[x_var].unique())
    
    n_groups = len(algorithms) * len(runs)
    bar_width = 0.8 / n_groups
    colors = plt.cm.tab20(np.linspace(0, 1, n_groups))
    
    x = np.arange(len(x_values))
    
    group_idx = 0
    for algo in algorithms:
        for run_label in runs:
            subset = combined_df[(combined_df['algorithm'] == algo) & 
                                 (combined_df['run_label'] == run_label)]
            
            if 'replica_id' in subset.columns and subset['replica_id'].nunique() > 1:
                grouped = subset.groupby(x_var)[metric].mean()
            else:
                grouped = subset.set_index(x_var)[metric]
            
            means = [grouped.get(v, 0) for v in x_values]
            offset = (group_idx - n_groups / 2 + 0.5) * bar_width
            
            bars = ax.bar(
                x + offset,
                means,
                width=bar_width,
                color=colors[group_idx],
                label=f'{algo} ({run_label})'
            )
            
            group_idx += 1
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} Comparison Across Runs')
    ax.set_xticks(x)
    ax.set_xticklabels(x_values)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot experiment results from CSV files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available runs
  python scripts/plot_results.py --list
  
  # List runs for a specific experiment
  python scripts/plot_results.py --list --experiment fig6
  
  # Plot results for a specific run_id
  python scripts/plot_results.py --run-id 20260129_014823
  
  # Plot results for a specific run_id and experiment
  python scripts/plot_results.py --run-id 20260129_014823 --experiment fig6
  
  # Plot from a specific CSV file with run_id filter
  python scripts/plot_results.py --csv results/fig6_experiment.csv --run-id 20260129_014823
  
  # Plot a specific metric
  python scripts/plot_results.py --run-id 20260129_014823 --metric acceptance_ratio
  
  # Compare multiple runs
  python scripts/plot_results.py --compare 20260129_014823 20260129_031757 --experiment fig6
  
  # Save plot to file
  python scripts/plot_results.py --run-id 20260129_014823 --output my_plot.png
        """
    )
    
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all available runs in results directory')
    parser.add_argument('--results-dir', '-d', default='results',
                        help='Results directory (default: results)')
    parser.add_argument('--run-id', '-r',
                        help='Run ID to plot (YYYYMMDD_HHMMSS format)')
    parser.add_argument('--experiment', '-e', default='fig6',
                        help='Experiment name (default: fig6)')
    parser.add_argument('--csv', '-c',
                        help='Path to specific CSV file to plot')
    parser.add_argument('--metric', '-m',
                        choices=['acceptance_ratio', 'avg_cost', 'avg_revenue',
                                 'cost_revenue_ratio', 'avg_execution_time', 'all'],
                        default='all',
                        help='Metric to plot (default: all)')
    parser.add_argument('--output', '-o',
                        help='Output path for saving the plot')
    parser.add_argument('--compare', nargs='+',
                        help='Compare multiple run_ids')
    parser.add_argument('--labels', nargs='+',
                        help='Labels for compared runs')
    parser.add_argument('--no-values', action='store_true',
                        help='Hide value labels on bars')
    
    args = parser.parse_args()
    
    # List available runs
    if args.list:
        print("\n" + "="*60)
        print(" Available Experiment Runs")
        print("="*60)
        runs = list_available_runs(args.results_dir, 
                                   args.experiment if args.experiment != 'fig6' else None)
        if not runs:
            print("No results found.")
            return
        
        for experiment, run_info in sorted(runs.items()):
            print(f"\n{experiment}:")
            print(f"  CSV: {run_info['path']}")
            print(f"  Run IDs ({len(run_info['run_ids'])} total):")
            for run_id in run_info['run_ids'][:10]:  # Show first 10
                print(f"    • {run_id}")
            if len(run_info['run_ids']) > 10:
                print(f"    ... and {len(run_info['run_ids']) - 10} more")
        print()
        return
    
    # Determine CSV path
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = get_csv_path(args.experiment, args.results_dir)
        if not csv_path:
            print(f"CSV file not found for experiment: {args.experiment}")
            print(f"Expected: {args.results_dir}/{args.experiment}_experiment.csv")
            return
    
    # Compare multiple runs
    if args.compare:
        compare_runs(
            csv_path,
            args.compare,
            labels=args.labels,
            metric=args.metric if args.metric != 'all' else 'acceptance_ratio',
            output_path=args.output
        )
        return
    
    # Load data
    run_id = args.run_id
    df = load_results(csv_path, run_id)
    
    if df is None or df.empty:
        if run_id:
            # Show available run_ids
            print(f"\nNo data found for run_id: {run_id}")
            print("\nAvailable run_ids in this file:")
            all_df = pd.read_csv(csv_path)
            if 'run_id' in all_df.columns:
                unique_runs = sorted(all_df['run_id'].dropna().unique(), reverse=True)
                for rid in unique_runs[:10]:
                    print(f"  • {rid}")
                if len(unique_runs) > 10:
                    print(f"  ... and {len(unique_runs) - 10} more")
        else:
            print("No data to plot. Please specify a --run-id")
        return
    
    print(f"\nLoaded {len(df)} records from {csv_path}")
    if run_id:
        print(f"Filtered by run_id: {run_id}")
    print(f"Algorithms: {df['algorithm'].unique().tolist()}")
    if 'replica_id' in df.columns:
        print(f"Replicas: {df['replica_id'].nunique()}")
    
    # Plot
    if args.metric == 'all':
        output_dir = os.path.dirname(args.output) if args.output else None
        plot_all_metrics(df, output_dir=output_dir, run_id=run_id)
    else:
        plot_bar_chart(
            df,
            args.metric,
            output_path=args.output,
            show_values=not args.no_values
        )


if __name__ == '__main__':
    main()
