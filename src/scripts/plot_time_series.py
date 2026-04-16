import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize time series metrics from VNE experiments")
    parser.add_argument(
        "--csv_file", 
        type=str, 
        required=True, 
        help="Path to the time-series CSV file (e.g., results/fig6/time_series_fig6_experiment_xxx.csv)"
    )
    parser.add_argument(
        "--metrics", 
        type=str, 
        nargs="+", 
        default=["acceptance_ratio", "avg_cost", "avg_revenue"],
        help="Metrics to plot (defaults to Acceptance Rate, Cost, and Revenue)"
    )
    parser.add_argument(
        "--replica", 
        type=int, 
        default=None, 
        help="Filter the analysis by a specific dataset replica ID"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        default=None, 
        help="Custom path to save the generated plot. If not provided, it saves next to the CSV."
    )
    parser.add_argument(
        "--x_axis", 
        type=str, 
        default="window_idx", 
        choices=["window_idx", "time"], 
        help="What to use for the X-axis (either window_idx or actual physical time in simulation)"
    )
    parser.add_argument(
        "--dpi", 
        type=int, 
        default=300, 
        help="DPI of the output image"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found.")
        return
        
    print(f"Loading time-series data from {args.csv_file}...")
    df = pd.read_csv(args.csv_file)
    
    # Show user what columns actually exist
    print(f"Available columns: {', '.join(df.columns.tolist())}")
    
    # Filter dataset based on the requested replica
    if args.replica is not None:
        if 'replica_id' in df.columns:
            df = df[df['replica_id'] == args.replica]
            print(f"Filtered dataset for replica_id = {args.replica}")
            if df.empty:
                print("No data left after filtering! Please check if your replica param exists in the CSV.")
                return
        else:
            print("Warning: '--replica' given but 'replica_id' column is absent in dataset.")
            
    # Set default metrics and their display names
    metric_display_names = {
        "acceptance_ratio": "Acceptance Rate",
        "avg_cost": "Cost",
        "avg_revenue": "Revenue"
    }
    
    # If using defaults, ensure we have the requested three
    if args.metrics == ["avg_revenue", "avg_cost", "acceptance_ratio"]:
        args.metrics = ["acceptance_ratio", "avg_cost", "avg_revenue"]

    # Calculate average of all records for the same key (algorithm, replica_id, domains, time point)
    # This ensures we get 1 line per (algorithm, dataset/replica) pair
    print(f"Averaging records grouped by algorithm/replica_id/domains and '{args.x_axis}'...")
    
    # We group by these columns to ensure they are preserved in the averaged dataframe
    group_cols = ['algorithm', 'replica_id', 'num_domains', args.x_axis]
    df_avg = df.groupby(group_cols).mean(numeric_only=True).reset_index()

    # Retain only requested metrics that actually exist in the dataframe
    available_metrics = [m for m in args.metrics if m in df_avg.columns]
    if not available_metrics:
        print(f"Error: None of the requested metrics ({args.metrics}) were found in the CSV.")
        return
        
    num_metrics = len(available_metrics)
    # Give a dynamic height depending on how many metrics we want to plot simultaneously
    fig, axes = plt.subplots(num_metrics, 1, figsize=(13, 5 * num_metrics), sharex=True)
    if num_metrics == 1:
        axes = [axes]
        
    # Set beautiful aesthetics
    sns.set_theme(style="whitegrid")
    
    # Map algorithm to colors and replica_id to line styles/markers
    for ax, metric in zip(axes, available_metrics):
        display_name = metric_display_names.get(metric, metric.replace("_", " ").title())
        print(f"Plotting averaged {metric} ({display_name}) over time...")
        
        # Ensure replica_id is continuous or categorical as needed for style
        if 'replica_id' in df_avg.columns:
            # We want replica_id to be treated as a category for styling
            df_avg['replica_id'] = df_avg['replica_id'].astype(str)
            style_col = "replica_id"
        else:
            style_col = None

        sns.lineplot(
            data=df_avg, 
            x=args.x_axis, 
            y=metric, 
            hue="algorithm", 
            style=style_col,
            markers=True,      # Use different markers for different configs
            dashes=True,       # Also use different line styles (solid, dashed, etc.)
            markersize=8,
            linewidth=2.5,
            ax=ax
        )
        
        ax.set_title(f"Simulation Progression of {display_name}", fontsize=16, fontweight="bold")
        ax.set_ylabel(display_name, fontsize=13)
        ax.set_xlabel(args.x_axis.replace("_", " ").title(), fontsize=13)
        
        # Legend formatting - move legend to the side to avoid overlapping lines
        ax.legend(title="Algorithm | Replica (Dataset)", loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
        
    plt.tight_layout()
    
    # Save the output
    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        plt.savefig(args.save_path, dpi=args.dpi, bbox_inches="tight")
        print(f"\nSaved plot to {args.save_path}")
    else:
        # Save near the source CSV file by default
        base_name = args.csv_file.replace('.csv', '')
        replica_suffix = f"_replica{args.replica}" if args.replica is not None else ""
        default_save = f"{base_name}_plotted_{args.x_axis}{replica_suffix}.png"
        
        plt.savefig(default_save, dpi=args.dpi, bbox_inches="tight")
        print(f"\nSaved plotted sequence to {default_save}")
        
    try:
        # Attempt to show plot if running in an interactive session
        plt.show()
    except Exception as e:
        print(f"Notice: Could not display plot interactively ({e}). File was successfully saved though.")

if __name__ == "__main__":
    main()
