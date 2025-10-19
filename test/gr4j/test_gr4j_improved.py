#!/usr/bin/env python
"""Quick test script for improved GR4J calibration - single gauge.

Run this before the full batch to verify improvements work correctly.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append("./")
from src.models.gr4j import model as gr4j
from src.models.gr4j.gr4j_optuna import run_optimization
from src.models.gr4j.pareto import analyze_pareto_front, select_best_trial_weighted
from src.models.gr4j.pet import pet_oudin
from src.readers.geom_reader import load_geodata
from src.utils.logger import setup_logger
from src.utils.metrics import evaluate_model
from src.utils.metrics_enhanced import analyze_flow_regimes

# Setup
Path("logs").mkdir(exist_ok=True)
logger = setup_logger("test_improved", log_file="logs/test_improved.log", level="INFO")


def main() -> None:
    """Test improved calibration on a single gauge."""
    # Configuration
    gauge_id = "75257"  # Change to any gauge you have
    dataset = "e5l"
    calibration_period = ("2010-01-01", "2018-12-31")
    validation_period = ("2019-01-01", "2020-12-31")
    warmup_years = 2
    n_trials = 1000  # Quick test (use 1000+ for real calibration)

    logger.info("=" * 80)
    logger.info(f"TESTING IMPROVED CALIBRATION ON GAUGE {gauge_id}")
    logger.info("=" * 80)

    # Load gauge data
    _, gauges = load_geodata(folder_depth=".")
    if gauge_id not in gauges.index:
        logger.error(f"Gauge {gauge_id} not found!")
        return

    latitude = float(gauges.loc[gauge_id, "geometry"].y)
    logger.info(f"Gauge latitude: {latitude:.4f}°N")

    # Load hydro-meteo data
    logger.info(f"Loading data for gauge {gauge_id}...")
    try:
        with xr.open_dataset(f"data/nc_all_q/{gauge_id}.nc") as ds:
            data = ds.to_dataframe()
            data["t_mean_e5l"] = (data["t_max_e5l"] + data["t_min_e5l"]) / 2

        # Prepare data with extended range for warm-up
        calib_start_year = pd.to_datetime(calibration_period[0]).year
        data_start_year = calib_start_year - warmup_years

        gr4j_data = data.loc[
            f"{data_start_year}" : "2020",
            ["q_mm_day", "t_min_e5l", "t_mean_e5l", "t_max_e5l", f"prcp_{dataset}"],
        ].copy()

        gr4j_data.rename(
            columns={f"prcp_{dataset}": "prcp", "t_mean_e5l": "t_mean"}, inplace=True
        )

        # Add day_of_year and PET
        gr4j_data["day_of_year"] = gr4j_data.index.dayofyear  # type: ignore[attr-defined]
        t_mean_list = gr4j_data["t_mean"].tolist()
        day_of_year_list = gr4j_data["day_of_year"].tolist()
        gr4j_data["pet_mm_day"] = pet_oudin(t_mean_list, day_of_year_list, latitude)

        logger.info(f"Data range: {gr4j_data.index[0]} to {gr4j_data.index[-1]}")
        logger.info(f"Data shape: {gr4j_data.shape}")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Run improved optimization
    logger.info(f"\n{'=' * 80}")
    logger.info(f"RUNNING OPTIMIZATION ({n_trials} trials, {warmup_years}-year warm-up)")
    logger.info(f"{'=' * 80}\n")

    study = run_optimization(
        gr4j_data,
        calibration_period=calibration_period,
        study_name=f"test_improved_{gauge_id}_{dataset}",
        n_trials=n_trials,
        timeout=600,  # 10 minutes
        verbose=True,
        warmup_years=warmup_years,
        use_detailed=False,
    )

    if not study.best_trials:
        logger.error("No valid trials found!")
        return

    # Analyze Pareto front
    pareto_df = analyze_pareto_front(study.best_trials)
    logger.info(f"\n{'=' * 80}")
    logger.info("OPTIMIZATION RESULTS")
    logger.info(f"{'=' * 80}")
    logger.info(f"Found {len(study.best_trials)} Pareto-optimal solutions")
    logger.info("\nTop 5 by KGE:")
    print(
        pareto_df[
            [
                "KGE",
                "low_flow_composite",
                "high_flow_composite",
                "x1",
                "x2",
                "x3",
                "x4",
                "ctg",
                "kf",
                "tt",
            ]
        ].head()
    )

    # Select best parameter set
    weights = {"KGE": 0.25, "low_flow": 0.35, "high_flow": 0.30, "PBIAS": 0.10}
    best_trial = select_best_trial_weighted(study.best_trials, weights, "weighted_sum")

    logger.info(f"\n{'=' * 80}")
    logger.info("BEST PARAMETERS (weighted selection):")
    logger.info(f"{'=' * 80}")
    for param, value in best_trial.params.items():
        logger.info(f"  {param:4s}: {value:8.3f}")

    # Check parameter realism
    logger.info("\nParameter realism check:")
    ctg = best_trial.params["ctg"]
    kf = best_trial.params["kf"]
    tt = best_trial.params["tt"]
    logger.info(
        f"  CTG: {ctg:.3f} {'✓ OK' if 0.2 <= ctg <= 0.8 else '⚠ Outside typical range (0.2-0.8)'}"
    )
    logger.info(
        f"  Kf:  {kf:.3f} {'✓ OK' if 2.0 <= kf <= 6.0 else '⚠ Outside typical range (2-6)'}"
    )
    logger.info(
        f"  TT:  {tt:.3f} {'✓ OK' if -1.0 <= tt <= 1.0 else '⚠ Outside typical range (-1 to +1)'}"
    )

    # Validate on validation period
    logger.info(f"\n{'=' * 80}")
    logger.info("VALIDATION RESULTS")
    logger.info(f"{'=' * 80}")

    val_start = pd.to_datetime(validation_period[0])
    val_warmup_start = val_start - pd.DateOffset(years=warmup_years)

    if val_warmup_start < gr4j_data.index[0]:
        val_warmup_start = gr4j_data.index[0]

    # Run simulation with warm-up
    gr4j_validation_full = gr4j_data.loc[val_warmup_start : validation_period[1], :]
    q_sim_full = gr4j.simulation(gr4j_validation_full, list(best_trial.params.values()))

    # Extract validation period
    n_val_warmup = len(gr4j_data[val_warmup_start:val_start]) - 1
    q_sim = q_sim_full[n_val_warmup:]

    # Get observed values
    gr4j_validation = gr4j_data.loc[validation_period[0] : validation_period[1], :]
    observed_values = np.array(gr4j_validation["q_mm_day"].values, dtype=float)
    q_sim_np = np.array(q_sim[: len(observed_values)], dtype=float)

    # Calculate metrics
    metrics = evaluate_model(observed_values, q_sim_np)
    regime_metrics = analyze_flow_regimes(observed_values, q_sim_np)

    logger.info("Overall metrics:")
    logger.info(f"  KGE:    {metrics['KGE']:.3f}")
    logger.info(f"  NSE:    {metrics['NSE']:.3f}")
    logger.info(f"  logNSE: {metrics['logNSE']:.3f}")
    logger.info(f"  PBIAS:  {metrics['PBIAS']:+.2f}%")
    logger.info(f"  RMSE:   {metrics['RMSE']:.3f} mm/day")

    logger.info("\nFlow regime performance (NSE by quantile):")
    logger.info(f"  Very low (Q<Q10):  {regime_metrics.get('very_low_nse', np.nan):.3f}")
    logger.info(f"  Low (Q10-Q30):     {regime_metrics.get('low_nse', np.nan):.3f}")
    logger.info(f"  Medium (Q30-Q70):  {regime_metrics.get('medium_nse', np.nan):.3f}")
    logger.info(f"  High (Q70-Q90):    {regime_metrics.get('high_nse', np.nan):.3f}")
    logger.info(f"  Very high (Q>Q90): {regime_metrics.get('very_high_nse', np.nan):.3f}")

    # Success criteria
    logger.info(f"\n{'=' * 80}")
    logger.info("SUCCESS CRITERIA:")
    logger.info(f"{'=' * 80}")

    checks = [
        ("KGE > 0.6", metrics["KGE"] > 0.6),
        ("NSE > 0.5", metrics["NSE"] > 0.5),
        ("logNSE > 0.4", metrics["logNSE"] > 0.4),
        ("|PBIAS| < 15%", abs(metrics["PBIAS"]) < 15),
        (
            "All regime NSE > 0.3",
            all(
                regime_metrics.get(k, 0) > 0.3
                for k in [
                    "very_low_nse",
                    "low_nse",
                    "medium_nse",
                    "high_nse",
                    "very_high_nse",
                ]
            ),
        ),
    ]

    for criterion, passed in checks:
        logger.info(f"  {'✓' if passed else '✗'} {criterion}")

    all_passed = all(c[1] for c in checks)
    logger.info(
        f"\n{'✓ ALL CHECKS PASSED!' if all_passed else '⚠ Some checks failed (may be normal for difficult basins)'}"
    )

    logger.info(f"\n{'=' * 80}")
    logger.info("✅ TEST COMPLETED SUCCESSFULLY!")
    logger.info(f"{'=' * 80}")
    logger.info("\nIf results look good, run full calibration with:")
    logger.info("  python scripts/gr4j_optuna.py")


if __name__ == "__main__":
    main()
