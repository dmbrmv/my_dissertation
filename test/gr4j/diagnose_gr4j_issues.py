"""Diagnostic script to understand GR4J optimization issues."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def diagnose_gr4j() -> None:
    """Diagnose GR4J optimization results and identify issues."""
    base_path = Path("data/optimization/gr4j_optuna")
    all_results = []

    # Collect all results
    for gauge_dir in base_path.iterdir():
        if not gauge_dir.is_dir():
            continue

        gauge_id = gauge_dir.name

        for dataset in ["e5l", "gpcp", "e5", "mswep"]:
            metric_file = gauge_dir / f"{gauge_id}_{dataset}" / "metrics.json"

            if metric_file.exists():
                with open(metric_file) as f:
                    metrics = json.load(f)
                    metrics["gauge_id"] = gauge_id
                    metrics["dataset"] = dataset
                    all_results.append(metrics)

    df = pd.DataFrame(all_results)

    print("=" * 80)
    print("GR4J OPTIMIZATION DIAGNOSTIC")
    print("=" * 80)

    # 1. Performance categories
    print("\n## Performance Distribution (NSE)")
    print("-" * 80)
    nse_bins = [
        (-np.inf, 0, "Negative (worse than mean)"),
        (0, 0.5, "Poor"),
        (0.5, 0.7, "Acceptable"),
        (0.7, 0.8, "Good"),
        (0.8, np.inf, "Excellent"),
    ]

    for low, high, label in nse_bins:
        count = ((df["NSE"] > low) & (df["NSE"] <= high)).sum()
        pct = 100 * count / len(df)
        print(f"{label:30s}: {count:3d} ({pct:5.1f}%)")

    # 2. Check for problematic flow regimes
    print("\n## Flow Regime Issues")
    print("-" * 80)
    print(
        f"Very low flows (NSE < -50): {(df['very_low_nse'] < -50).sum()} "
        f"({100 * (df['very_low_nse'] < -50).sum() / len(df):.1f}%)"
    )
    print(
        f"Low flows (NSE < 0):        {(df['low_nse'] < 0).sum()} "
        f"({100 * (df['low_nse'] < 0).sum() / len(df):.1f}%)"
    )
    print(
        f"High flows (NSE < 0):       {(df['high_nse'] < 0).sum()} "
        f"({100 * (df['high_nse'] < 0).sum() / len(df):.1f}%)"
    )

    # 3. Dataset comparison
    print("\n## Dataset Performance (Median NSE)")
    print("-" * 80)
    for dataset in ["e5l", "e5", "gpcp", "mswep"]:
        subset = df[df["dataset"] == dataset]
        med_nse = subset["NSE"].median()
        med_kge = subset["KGE"].median()
        n_good = (subset["NSE"] > 0.5).sum()
        print(
            f"{dataset:8s}: NSE={med_nse:.3f}, KGE={med_kge:.3f}, "
            f"Good (NSE>0.5)={n_good}/{len(subset)}"
        )

    # 4. Identify worst performers
    print("\n## Worst 10 Gauges (by NSE)")
    print("-" * 80)
    worst = df.nsmallest(10, "NSE")[["gauge_id", "dataset", "NSE", "KGE", "PBIAS"]]
    print(worst.to_string(index=False))

    # 5. Identify best performers
    print("\n## Best 10 Gauges (by NSE)")
    print("-" * 80)
    best = df.nlargest(10, "NSE")[["gauge_id", "dataset", "NSE", "KGE", "PBIAS"]]
    print(best.to_string(index=False))

    # 6. Check for convergence issues (high PBIAS)
    print("\n## Volume Balance Issues")
    print("-" * 80)
    high_bias = df[abs(df["PBIAS"]) > 20]
    print(
        f"Gauges with |PBIAS| > 20%: {len(high_bias)} ({100 * len(high_bias) / len(df):.1f}%)"
    )  # noqa: E501
    print(f"Median PBIAS: {df['PBIAS'].median():.1f}%")
    print(f"Mean PBIAS: {df['PBIAS'].mean():.1f}%")

    # 7. Correlation between metrics
    print("\n## Metric Correlations")
    print("-" * 80)
    print(f"NSE vs KGE:   r={df['NSE'].corr(df['KGE']):.3f}")
    print(f"NSE vs logNSE: r={df['NSE'].corr(df['logNSE']):.3f}")
    print(f"NSE vs PFE:   r={df['NSE'].corr(df['PFE']):.3f}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    diagnose_gr4j()
