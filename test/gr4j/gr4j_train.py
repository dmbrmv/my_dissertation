"""Main script to run GR4J Optuna optimization for multiple gauges.

This script uses multi-objective optimization with:
- 2-year warm-up period for model initialization
- Corrected CemaNeige parameter bounds (per airGR standards)
- Composite flow regime metrics for balanced low/high flow performance
- Enhanced metric weighting emphasizing low flows
"""

from functools import partial
import multiprocessing as mp
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append("./")


from src.models.gr4j import model as gr4j
from src.models.gr4j.gr4j_optuna import run_optimization
from src.models.gr4j.parallel import run_parallel_optimization
from src.models.gr4j.pareto import (
    save_optimization_results,
    select_best_trial_weighted,
)
from src.models.gr4j.pet import pet_oudin
from src.readers.geom_reader import load_geodata
from src.utils.logger import setup_logger
from timeseries_stats.metrics import evaluate_model
from timeseries_stats.metrics_enhanced import analyze_flow_regimes

# Create log directories before importing modules that might use them
Path("logs").mkdir(exist_ok=True)

# Setup logger
logger = setup_logger("main_gr4j_optuna", log_file="logs/gr4j_optuna.log", level="INFO")


def process_gr4j_gauge(
    gauge_id: str,
    datasets: list[str],
    calibration_period: tuple[str, str],
    validation_period: tuple[str, str],
    save_storage: Path,
    e_obs_gauge: gpd.GeoDataFrame | None = None,
    n_trials: int = 100,
    timeout: int = 3600,
    overwrite_results: bool = False,
    warmup_years: int = 2,
) -> None:
    """Process a single gauge with GR4J+CemaNeige calibration.

    Args:
        gauge_id: Gauge identifier
        datasets: List of meteorological dataset names
        calibration_period: (start_date, end_date) for calibration
        validation_period: (start_date, end_date) for validation
        save_storage: Root directory to save results
        e_obs_gauge: GeoDataFrame with gauge information
        n_trials: Number of optimization trials
        timeout: Optimization timeout in seconds
        overwrite_results: Whether to overwrite existing results
        warmup_years: Years for warm-up period (default 2)
    """
    result_path = save_storage / gauge_id
    result_path.mkdir(parents=True, exist_ok=True)

    # Extract latitude for PET calculation
    if e_obs_gauge is None:
        raise ValueError("e_obs_gauge GeoDataFrame is required")

    point_geom = e_obs_gauge.loc[gauge_id, "geometry"]
    latitude = float(point_geom.y)  # type: ignore[union-attr]

    # Improved weights: Better balance of low/high flows
    hydro_weights: dict[str, float] = {
        "KGE": 0.25,  # Overall performance
        "low_flow": 0.35,  # Composite low-flow metric (up from implicit 0.5)
        "high_flow": 0.30,  # Composite high-flow metric (up from 0.5)
        "PBIAS": 0.10,  # Volume conservation (up from 0.03)
    }

    for dataset in datasets:
        # Check if results exist
        if (result_path / f"{gauge_id}_{dataset}").exists() and not overwrite_results:
            continue

        try:
            # Load data - extend backwards to include warm-up period
            with xr.open_dataset(f"data/nc_all_q/{gauge_id}.nc") as ds:
                exmp = ds.to_dataframe()
                exmp["t_mean_e5l"] = (exmp["t_max_e5l"] + exmp["t_min_e5l"]) / 2

            # Calculate required start year for warm-up
            calib_start_year = pd.to_datetime(calibration_period[0]).year
            data_start_year = calib_start_year - warmup_years

            # Load extended data range (e.g., 2005-2020 instead of 2008-2020)
            gr4j_data = exmp.loc[
                f"{data_start_year}" : "2020",
                [
                    "q_mm_day",
                    "t_min_e5l",
                    "t_mean_e5l",
                    "t_max_e5l",
                    f"prcp_{dataset}",
                ],
            ].copy()

            # Rename columns for consistency
            gr4j_data.rename(
                columns={f"prcp_{dataset}": "prcp", "t_mean_e5l": "t_mean"},
                inplace=True,
            )

            # Ensure day_of_year is available
            if "day_of_year" not in gr4j_data.columns:
                gr4j_data["day_of_year"] = gr4j_data.index.dayofyear  # type: ignore[attr-defined]
            else:
                missing_doy = gr4j_data["day_of_year"].isna()
                if missing_doy.any():
                    gr4j_data.loc[missing_doy, "day_of_year"] = gr4j_data.index[
                        missing_doy
                    ].dayofyear  # type: ignore[attr-defined]

            # Calculate PET using Oudin formula
            t_mean_list = gr4j_data["t_mean"].tolist()
            day_of_year_list = gr4j_data["day_of_year"].tolist()
            gr4j_data["pet_mm_day"] = pet_oudin(t_mean_list, day_of_year_list, latitude)

            # Run IMPROVED optimization with warm-up
            study = run_optimization(
                gr4j_data,
                calibration_period=calibration_period,
                study_name=f"GR4J_improved_{gauge_id}_{dataset}",
                n_trials=n_trials,
                timeout=timeout,
                verbose=False,
                warmup_years=warmup_years,
                use_detailed=False,  # Use 4-objective composite approach
            )

            # Check for valid trials
            if not study.best_trials:
                logger.warning(f"No valid trials found for {gauge_id} with {dataset}")
                continue

            # Select best parameter set using improved weights
            pareto_trials = study.best_trials
            best_hydro = select_best_trial_weighted(
                pareto_trials, hydro_weights, "weighted_sum"
            )
            best_params = dict(best_hydro.params)

            # Validate model with best parameters (include warm-up for consistency)
            val_start = pd.to_datetime(validation_period[0])
            val_warmup_start = val_start - pd.DateOffset(years=warmup_years)

            # Ensure warm-up data is available for validation
            if val_warmup_start < gr4j_data.index[0]:
                logger.warning(
                    f"Insufficient warm-up data for validation. "
                    f"Using available from {gr4j_data.index[0]}"
                )
                val_warmup_start = gr4j_data.index[0]

            gr4j_validation_full = gr4j_data.loc[
                val_warmup_start : validation_period[1], :
            ]
            q_sim_full = gr4j.simulation(gr4j_validation_full, list(best_params.values()))

            # Extract validation period (exclude warm-up)
            n_val_warmup = len(gr4j_data[val_warmup_start:val_start]) - 1
            q_sim = q_sim_full[n_val_warmup:]

            # Get observed values for validation period
            gr4j_validation = gr4j_data.loc[
                validation_period[0] : validation_period[1], :
            ]
            observed_values = np.array(gr4j_validation["q_mm_day"].values, dtype=float)

            # Ensure arrays match in length
            min_len = min(len(observed_values), len(q_sim))
            observed_values = observed_values[:min_len]
            q_sim_np = np.array(q_sim[:min_len], dtype=float)

            # Calculate standard metrics
            metrics = evaluate_model(observed_values, q_sim_np)

            # Add flow regime analysis (NEW)
            regime_metrics = analyze_flow_regimes(observed_values, q_sim_np)
            metrics.update(regime_metrics)

            # Save optimization results
            save_optimization_results(
                study=study,
                dataset_name=dataset,
                gauge_id=gauge_id,
                best_parameters=best_params,
                metrics=metrics,
                output_dir=str(result_path),
            )
            logger.info(f"✓ {gauge_id}/{dataset}: KGE={metrics.get('KGE', np.nan):.3f}")

        except Exception as e:
            logger.error(f"✗ {gauge_id}/{dataset}: {str(e)}")


def main() -> None:
    """Run improved GR4J+CemaNeige calibration for all gauges.

    Key improvements:
    - 2-year warm-up period (requires data from 2006 onwards)
    - Corrected CemaNeige parameter bounds
    - Composite flow regime objectives
    - Enhanced low-flow emphasis
    - Flow regime diagnostics in output
    """
    # Load gauge and watershed data
    ws, gauges = load_geodata(folder_depth=".")

    # Find valid gauges (no missing data)
    logger.info("Finding gauges with valid data...")
    full_gauges = [
        i.stem for i in Path("data/nc_all_q").glob("*.nc") if i.stem in gauges.index
    ]
    logger.info(f"Found {len(full_gauges)} valid gauges")

    # Set up optimization parameters
    calibration_period = ("2010-01-01", "2018-12-31")
    validation_period = ("2019-01-01", "2020-12-31")

    # Save to results directory
    save_storage = Path("data/optimization/gr4j_optuna/")
    save_storage.mkdir(parents=True, exist_ok=True)  # Optimization settings
    n_trials = 1500  # Same as before for fair comparison
    timeout = 1200  # 20 minutes per gauge/dataset
    warmup_years = 2  # NEW: 2-year warm-up (2008-2009 for calibration start 2010)
    overwrite_existing_results = False

    # Meteorological datasets to test
    datasets = ["e5l", "gpcp", "e5", "mswep"]

    # Log configuration
    logger.info("=" * 80)
    logger.info("IMPROVED GR4J+CEMANEIGE CALIBRATION")
    logger.info("=" * 80)
    logger.info(f"Number of gauges: {len(full_gauges)}")
    logger.info(f"Datasets: {', '.join(datasets)}")
    logger.info(f"Calibration period: {calibration_period[0]} to {calibration_period[1]}")
    logger.info(f"Validation period: {validation_period[0]} to {validation_period[1]}")
    logger.info(f"Warm-up period: {warmup_years} years")
    logger.info(f"Trials per gauge/dataset: {n_trials}")
    logger.info(f"Timeout per gauge/dataset: {timeout}s ({timeout / 60:.1f} min)")
    logger.info(f"Results directory: {save_storage}")
    logger.info("=" * 80)

    # Run optimization in parallel
    n_processes = max(1, mp.cpu_count() - 2)  # Leave 2 cores free
    logger.info(f"Starting parallel optimization with {n_processes} processes")

    run_parallel_optimization(
        gauge_ids=full_gauges,
        process_gauge_func=partial(
            process_gr4j_gauge,
            datasets=datasets,
            calibration_period=calibration_period,
            validation_period=validation_period,
            save_storage=save_storage,
            e_obs_gauge=gauges,
            n_trials=n_trials,
            timeout=timeout,
            overwrite_results=overwrite_existing_results,
            warmup_years=warmup_years,  # NEW parameter
        ),
        n_processes=n_processes,
    )

    logger.info("=" * 80)
    logger.info("✅ Optimization completed successfully!")
    logger.info(f"Results saved to: {save_storage}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
