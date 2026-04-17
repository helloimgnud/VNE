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
        "--dataset_col",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Column(s) that together identify a dataset configuration. "
            "Runs sharing the same values in these columns are averaged. "
            "Leave blank to auto-detect (prefers dataset_id)."
        )
    )
    parser.add_argument(
        "--dataset_filter",
        type=str,
        default=None,
        help=(
            "Filter to a specific dataset label (the value that appears after grouping). "
            "Format matches the auto-generated label, e.g. 'vnodes=2-8'. "
            "Leave blank to show all datasets."
        )
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


def _make_dataset_label(row: pd.Series, dataset_cols: list) -> str:
    """Build a human-readable dataset label from the identifying columns."""
    if "dataset_id" in dataset_cols:
        return str(row["dataset_id"])
    if dataset_cols == ["vnr_min_nodes", "vnr_max_nodes"]:
        return f"vnodes={int(row['vnr_min_nodes'])}-{int(row['vnr_max_nodes'])}"
    if dataset_cols == ["vnr_min_nodes", "vnr_max_nodes", "replica_id"]:
        return f"vnodes={int(row['vnr_min_nodes'])}-{int(row['vnr_max_nodes'])}_rep={row['replica_id']}"
    # Generic: col=value pairs joined by "/"
    return " / ".join(f"{c}={row[c]}" for c in dataset_cols)


def main():
    args = parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found.")
        return

    print(f"Loading time-series data from {args.csv_file}...")
    df = pd.read_csv(args.csv_file)

    # Show user what columns actually exist
    print(f"Available columns: {', '.join(df.columns.tolist())}")

    # -----------------------------------------------------------------------
    # Build a 'dataset' label column from the identifying columns
    # -----------------------------------------------------------------------
    if args.dataset_col is None:
        # Auto-detect datasets
        if "dataset_id" in df.columns:
            dataset_cols = ["dataset_id"]
        elif "vnr_min_nodes" in df.columns and "vnr_max_nodes" in df.columns:
            dataset_cols = ["vnr_min_nodes", "vnr_max_nodes"]
            if "replica_id" in df.columns:
                dataset_cols.append("replica_id")
        else:
            dataset_cols = []
    else:
        dataset_cols = [c for c in args.dataset_col if c in df.columns]

    if not dataset_cols:
        print(
            f"Warning: could not automatically detect dataset columns and no valid --dataset_col was provided. "
            f"Falling back to treating all rows as one dataset."
        )
        df["dataset"] = "all"
    else:
        df["dataset"] = df.apply(lambda row: _make_dataset_label(row, dataset_cols), axis=1)

    print(f"Datasets found: {sorted(df['dataset'].unique())}")

    # Optional: filter to a specific dataset
    if args.dataset_filter is not None:
        df = df[df["dataset"] == args.dataset_filter]
        print(f"Filtered to dataset = '{args.dataset_filter}'")
        if df.empty:
            print("No data left after filtering! Check --dataset_filter value.")
            return

    # -----------------------------------------------------------------------
    # Average replicas within each (algorithm, dataset, time-point) cell.
    # replica_id is intentionally EXCLUDED from the grouping key so that
    # multiple replicas of the same dataset configuration are collapsed.
    # -----------------------------------------------------------------------
    group_cols = ["algorithm", "dataset", args.x_axis]
    # Add num_domains if present (it characterises the substrate, not the replica)
    if "num_domains" in df.columns:
        group_cols.insert(2, "num_domains")

    print(f"Averaging replicas — grouping by: {group_cols}")
    df_avg = df.groupby(group_cols).mean(numeric_only=True).reset_index()

    # -----------------------------------------------------------------------
    # Validate requested metrics
    # -----------------------------------------------------------------------
    # Normalise metric order
    if set(args.metrics) == {"avg_revenue", "avg_cost", "acceptance_ratio"}:
        args.metrics = ["acceptance_ratio", "avg_cost", "avg_revenue"]

    metric_display_names = {
        "acceptance_ratio": "Acceptance Rate",
        "avg_cost":         "Average Cost",
        "avg_revenue":      "Average Revenue",
    }

    available_metrics = [m for m in args.metrics if m in df_avg.columns]
    if not available_metrics:
        print(f"Error: None of the requested metrics ({args.metrics}) were found in the CSV.")
        return

    # -----------------------------------------------------------------------
    # Plot — hue = algorithm (colour), style = dataset (line style + marker)
    # -----------------------------------------------------------------------
    num_metrics = len(available_metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(13, 5 * num_metrics), sharex=True)
    if num_metrics == 1:
        axes = [axes]

    sns.set_theme(style="whitegrid")

    n_datasets = df_avg["dataset"].nunique()

    for ax, metric in zip(axes, available_metrics):
        display_name = metric_display_names.get(metric, metric.replace("_", " ").title())
        print(f"Plotting {metric} ({display_name}) — grouped by dataset...")

        sns.lineplot(
            data=df_avg,
            x=args.x_axis,
            y=metric,
            hue="algorithm",   # colour  = algorithm
            style="dataset",   # pattern = dataset configuration
            markers=True,
            dashes=True,
            markersize=8,
            linewidth=2.5,
            ax=ax,
        )

        ax.set_title(f"Simulation Progression of {display_name}", fontsize=16, fontweight="bold")
        ax.set_ylabel(display_name, fontsize=13)
        ax.set_xlabel(args.x_axis.replace("_", " ").title(), fontsize=13)

        # Subtitle showing exactly what was averaged
        n_eval_runs = df["eval_run"].nunique() if "eval_run" in df.columns else 1
        ax.set_title(
            f"Simulation Progression of {display_name}\n"
            f"({n_datasets} dataset(s) plotted separately, runs averaged per dataset: {n_eval_runs})",
            fontsize=14, fontweight="bold"
        )

        ax.legend(
            title="Algorithm (colour) / Dataset (style)",
            loc="upper left",
            bbox_to_anchor=(1, 1),
            fontsize=9,
        )

    plt.tight_layout()

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        plt.savefig(args.save_path, dpi=args.dpi, bbox_inches="tight")
        print(f"\nSaved plot to {args.save_path}")
    else:
        base_name = args.csv_file.replace(".csv", "")
        ds_suffix = f"_ds{args.dataset_filter.replace('=','-').replace('/','-')}" if args.dataset_filter else ""
        default_save = f"{base_name}_plotted_{args.x_axis}{ds_suffix}.png"
        plt.savefig(default_save, dpi=args.dpi, bbox_inches="tight")
        print(f"\nSaved plot to {default_save}")

    try:
        plt.show()
    except Exception as e:
        print(f"Notice: Could not display plot interactively ({e}). File was saved successfully.")


if __name__ == "__main__":
    main()
