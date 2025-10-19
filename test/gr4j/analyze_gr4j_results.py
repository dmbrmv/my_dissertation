"""Quick analysis of GR4J optimization results."""

import json
from pathlib import Path

import pandas as pd


def analyze_results() -> None:
    """Analyze GR4J optimization results."""
    base_path = Path("data/optimization/gr4j_optuna")

    all_metrics = []

    # Iterate through all gauge directories
    for gauge_dir in base_path.iterdir():
        if not gauge_dir.is_dir():
            continue

        gauge_id = gauge_dir.name

        # Check each dataset
        for dataset in ["e5l", "gpcp", "e5", "mswep"]:
            metric_file = gauge_dir / f"{gauge_id}_{dataset}" / "metrics.json"

            if metric_file.exists():
                with open(metric_file) as f:
                    metrics = json.load(f)
                    metrics["gauge_id"] = gauge_id
                    metrics["dataset"] = dataset
                    all_metrics.append(metrics)

    # Create DataFrame
    df = pd.DataFrame(all_metrics)

    # Print summary statistics
    print("=" * 80)
    print("GR4J OPTIMIZATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total results: {len(df)}")
    print(f"Unique gauges: {df['gauge_id'].nunique()}")
    print(f"Datasets: {df['dataset'].unique()}")
    print()

    # Overall metrics
    print("OVERALL METRICS (all datasets):")
    print("-" * 80)
    for metric in ["NSE", "KGE", "PBIAS", "RMSE", "MAE", "logNSE", "PFE"]:
        if metric in df.columns:
            print(
                f"{metric:10s} - Median: {df[metric].median():.3f}, "
                f"Mean: {df[metric].mean():.3f}, "
                f"Std: {df[metric].std():.3f}"
            )
    print()

    # Per-dataset metrics
    print("METRICS BY DATASET:")
    print("-" * 80)
    for dataset in df["dataset"].unique():
        subset = df[df["dataset"] == dataset]
        print(f"\n{dataset.upper()} (n={len(subset)}):")
        nse_med, nse_mean = subset["NSE"].median(), subset["NSE"].mean()
        kge_med, kge_mean = subset["KGE"].median(), subset["KGE"].mean()
        bias_med = subset["PBIAS"].median()
        bias_mean = subset["PBIAS"].mean()
        print(f"  NSE:  Median={nse_med:.3f}, Mean={nse_mean:.3f}")
        print(f"  KGE:  Median={kge_med:.3f}, Mean={kge_mean:.3f}")
        print(f"  PBIAS: Median={bias_med:.3f}, Mean={bias_mean:.3f}")

    # Flow regime breakdown
    print("\n" + "=" * 80)
    print("FLOW REGIME PERFORMANCE:")
    print("-" * 80)
    for flow_type in ["very_low", "low", "medium", "high", "very_high"]:
        nse_col = f"{flow_type}_nse"
        bias_col = f"{flow_type}_pbias"
        if nse_col in df.columns:
            print(
                f"{flow_type.upper():12s} - NSE: {df[nse_col].median():.3f}, "
                f"PBIAS: {df[bias_col].median():.3f}"
            )

    # Save summary
    output_file = "data/optimization/gr4j_optuna_summary.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Full results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    analyze_results()
