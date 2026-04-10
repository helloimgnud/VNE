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
        default=["acceptance_ratio", "avg_cost", "avg_revenue", "window_accepted", "window_expired"],
        help="Metrics to plot (can specify multiple, separated by spaces)"
    )
    parser.add_argument(
        "--vnodes", 
        type=int, 
        default=None, 
        help="Filter the analysis by a specific number of virtual nodes (num_vnodes)"
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
    
    # Filter dataset based on the requested vnodes
    if args.vnodes is not None:
        if 'num_vnodes' in df.columns:
            df = df[df['num_vnodes'] == args.vnodes]
            print(f"Filtered dataset for num_vnodes = {args.vnodes}")
            if df.empty:
                print("No data left after filtering! Please check if your vnodes param exists in the CSV.")
                return
        else:
            print("Warning: '--vnodes' given but 'num_vnodes' column is absent in dataset.")
            
    # Retain only requested metrics that actually exist in the dataframe
    available_metrics = [m for m in args.metrics if m in df.columns]
    if not available_metrics:
        print(f"Error: None of the requested metrics ({args.metrics}) were found in the CSV.")
        return
        
    num_metrics = len(available_metrics)
    # Give a dynamic height depending on how many metrics we want to plot simultaneously
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics), sharex=True)
    if num_metrics == 1:
        axes = [axes]
        
    # Set beautiful aesthetics
    sns.set_theme(style="whitegrid")
    
    # Since there are multiple runs/replicas, Seaborn's lineplot will automatically 
    # group by the x_axis + hue (algorithm), calculate the mean, and draw 95% Confidence Intervals
    for ax, metric in zip(axes, available_metrics):
        print(f"Plotting {metric} over time...")
        sns.lineplot(
            data=df, 
            x=args.x_axis, 
            y=metric, 
            hue="algorithm", 
            marker="o",
            markersize=6,
            ax=ax,
            err_style="band",   # draws the confidence band combining runs
            errorbar=("ci", 95) # 95% confidence interval
        )
        
        filter_str = f"num_vnodes={args.vnodes}" if args.vnodes else "All configurations"
        ax.set_title(f"Simulation Progression of '{metric}' [{filter_str}]", fontsize=12, fontweight="bold")
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel(args.x_axis.replace("_", " ").title(), fontsize=11)
        
        # Legend formatting
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, title="Algorithm")
        
    plt.tight_layout()
    
    # Save the output
    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        plt.savefig(args.save_path, dpi=args.dpi, bbox_inches="tight")
        print(f"\n✓ Saved plot to {args.save_path}")
    else:
        # Save near the source CSV file by default
        base_name = args.csv_file.replace('.csv', '')
        vnode_suffix = f"_vnodes{args.vnodes}" if args.vnodes else ""
        default_save = f"{base_name}_plotted_{args.x_axis}{vnode_suffix}.png"
        
        plt.savefig(default_save, dpi=args.dpi, bbox_inches="tight")
        print(f"\n✓ Saved plotted sequence to {default_save}")
        
    try:
        # Attempt to show plot if running in an interactive session
        plt.show()
    except Exception as e:
        print(f"Notice: Could not display plot interactively ({e}). File was successfully saved though.")

if __name__ == "__main__":
    main()
