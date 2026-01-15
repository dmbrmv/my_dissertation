"""GR4J single-objective optimization - SIMPLIFIED & ROBUST approach.

This script uses single-objective KGE optimization, which is:
- Proven in 90% of GR4J literature
- Faster (no Pareto front)
- More interpretable
- Better for comparing datasets

Key improvements over multi-objective:
- Reduced to n_trials=1000 (sufficient for single objective)
- Clearer calibration target
- No conflicting objectives
- Easier to diagnose issues
- Parallel execution for speed
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import optuna  # type: ignore[import-untyped]
import pandas as pd
import xarray as xr

sys.path.append("./")

from src.models.gr4j import model as gr4j
from src.models.gr4j.pet import pet_oudin
from src.timeseries_stats.metrics import evaluate_model, kling_gupta_efficiency
from src.utils.logger import setup_logger

Path("logs").mkdir(exist_ok=True)
logger = setup_logger("gr4j_simple", log_file="logs/gr4j_simple_optim.log", level="INFO")

# Suppress Optuna's default logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective_single_kge(
    trial: optuna.Trial,
    data: pd.DataFrame,
    calibration_period: tuple[str, str],
    warmup_years: int = 2,
) -> float:
    """Single-objective KGE optimization for GR4J."""
    # GR4J parameters
    x1 = trial.suggest_float("x1", 10.0, 5000.0, log=True)
    x2 = trial.suggest_float("x2", -30.0, 30.0)
    x3 = trial.suggest_float("x3", 1.0, 6000.0, log=True)
    x4 = trial.suggest_float("x4", 0.05, 30.0)

    # CemaNeige parameters (corrected bounds)
    ctg = trial.suggest_float("ctg", 0.0, 1.0)
    kf = trial.suggest_float("kf", 1.0, 10.0)
    tt = trial.suggest_float("tt", -2.0, 2.5)

    params = [x1, x2, x3, x4, ctg, kf, tt]

    # Calculate warm-up period
    calib_start = pd.to_datetime(calibration_period[0])
    calib_end = pd.to_datetime(calibration_period[1])
    warmup_start = calib_start - pd.DateOffset(years=warmup_years)

    if warmup_start < data.index[0]:
        warmup_start = data.index[0]

    # Run simulation with warm-up
    try:
        warmup_data = data[warmup_start:calib_end]
        q_sim_full = gr4j.simulation(warmup_data, params)

        # Extract calibration period (exclude warm-up)
        n_warmup_days = len(data[warmup_start:calib_start]) - 1
        q_sim = q_sim_full[n_warmup_days:]
        q_obs = data[calib_start:calib_end]["q_mm_day"].values

        # Ensure arrays match
        min_len = min(len(q_obs), len(q_sim))
        q_obs = q_obs[:min_len]
        q_sim = q_sim[:min_len]

    except Exception as e:
        logger.error(f"Simulation failed for trial {trial.number}: {e}")
        return -999.0  # Minimize negative KGE

    # Calculate KGE
    kge = kling_gupta_efficiency(np.asarray(q_obs), np.asarray(q_sim))

    if pd.isna(kge):
        return -999.0

    return kge  # Maximize KGE


def process_gauge_simple(
    gauge_id: str,
    datasets: list[str],
    calibration_period: tuple[str, str],
    validation_period: tuple[str, str],
    save_storage: Path,
    e_obs_gauge: gpd.GeoDataFrame | None = None,
    n_trials: int = 1000,
    timeout: int = 600,
    overwrite_results: bool = False,
    warmup_years: int = 2,
) -> None:
    """Process gauge with SIMPLE single-objective optimization.

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

    if e_obs_gauge is None:
        raise ValueError("e_obs_gauge GeoDataFrame is required")

    point_geom = e_obs_gauge.loc[gauge_id, "geometry"]
    latitude = float(point_geom.y)  # type: ignore[union-attr]

    for dataset in datasets:
        output_file = result_path / f"{gauge_id}_{dataset}_metrics.json"
        if output_file.exists() and not overwrite_results:
            continue

        try:
            # Load data
            with xr.open_dataset(f"data/nc_all_q/{gauge_id}.nc") as ds:
                exmp = ds.to_dataframe()
                exmp["t_mean_e5l"] = (exmp["t_max_e5l"] + exmp["t_min_e5l"]) / 2

            # Calculate data start year for warm-up
            calib_start_year = pd.to_datetime(calibration_period[0]).year
            data_start_year = calib_start_year - warmup_years

            # Load extended data
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

            gr4j_data.rename(
                columns={f"prcp_{dataset}": "prcp", "t_mean_e5l": "t_mean"},
                inplace=True,
            )

            # Ensure day_of_year
            if "day_of_year" not in gr4j_data.columns:
                gr4j_data["day_of_year"] = gr4j_data.index.dayofyear  # type: ignore[attr-defined]

            # Calculate PET
            t_mean_list = gr4j_data["t_mean"].tolist()
            day_of_year_list = gr4j_data["day_of_year"].tolist()
            gr4j_data["pet_mm_day"] = pet_oudin(t_mean_list, day_of_year_list, latitude)

            # Run optimization
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                study_name=f"GR4J_simple_{gauge_id}_{dataset}",
            )

            # Define objective with captured variables
            def opt_func(
                trial: optuna.Trial,
                data: pd.DataFrame = gr4j_data,
                calib_period: tuple[str, str] = calibration_period,
                warmup: int = warmup_years,
            ) -> float:
                return objective_single_kge(trial, data, calib_period, warmup)

            study.optimize(
                opt_func,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=False,
            )

            if study.best_trial is None:
                logger.warning(f"No valid trials for {gauge_id}/{dataset}")
                continue

            # Get best parameters IN THE CORRECT ORDER
            # CRITICAL: Must match the order in objective_single_kge params list!
            best_trial_params = study.best_trial.params
            best_params = [
                best_trial_params["x1"],
                best_trial_params["x2"],
                best_trial_params["x3"],
                best_trial_params["x4"],
                best_trial_params["ctg"],
                best_trial_params["kf"],
                best_trial_params["tt"],
            ]

            # Validate with continuous simulation (preserves model states)
            # Run from calibration start through validation end
            calib_start = pd.to_datetime(calibration_period[0])
            warmup_start_calib = calib_start - pd.DateOffset(years=warmup_years)
            if warmup_start_calib < gr4j_data.index[0]:
                warmup_start_calib = gr4j_data.index[0]

            val_end = pd.to_datetime(validation_period[1])
            continuous_data = gr4j_data.loc[warmup_start_calib:val_end, :]

            q_sim_full = gr4j.simulation(continuous_data, best_params)

            # Extract validation period
            # CRITICAL: Count days in SAME dataframe we're simulating
            n_days_before_val = (
                len(continuous_data[warmup_start_calib : validation_period[0]]) - 1
            )
            q_sim = q_sim_full[n_days_before_val:]

            gr4j_validation = gr4j_data.loc[
                validation_period[0] : validation_period[1], :
            ]
            observed_values = np.array(gr4j_validation["q_mm_day"].values, dtype=float)

            # Match lengths
            min_len = min(len(observed_values), len(q_sim))
            observed_values = observed_values[:min_len]
            q_sim_np = np.array(q_sim[:min_len], dtype=float)

            # Calculate ALL metrics for comparison
            metrics = evaluate_model(observed_values, q_sim_np)

            # Save results
            import json

            output_file.write_text(json.dumps(metrics, indent=2))

            # Save best params with correct names
            param_names = ["x1", "x2", "x3", "x4", "ctg", "kf", "tt"]
            best_params_dict = dict(zip(param_names, best_params, strict=True))
            param_file = result_path / f"{gauge_id}_{dataset}_parameters.json"
            param_file.write_text(json.dumps(best_params_dict, indent=2))

            logger.info(
                f"✓ {gauge_id}/{dataset}: "
                f"KGE={metrics.get('KGE', np.nan):.3f}, "
                f"NSE={metrics.get('NSE', np.nan):.3f}"
            )

        except Exception as e:
            logger.error(f"✗ {gauge_id}/{dataset}: {str(e)}")


def main() -> None:
    """Run SIMPLIFIED single-objective GR4J calibration."""
    # Load gauge data
    # ws, gauges = load_geodata(folder_depth=".")
    gauges = gpd.read_file("res/FineTuneGauges.gpkg")
    gauges = gauges.set_index("gauge_id")

    logger.info("Finding gauges with valid data...")
    full_gauges = [
        i.stem for i in Path("data/nc_all_q").glob("*.nc") if i.stem in gauges.index
    ]
    logger.info(f"Found {len(full_gauges)} valid gauges")

    # Optimization parameters
    calibration_period = ("2010-01-01", "2016-12-31")
    validation_period = ("2017-01-01", "2018-12-31")

    save_storage = Path("data/optimization_poor_gauges/gr4j_simple/")
    save_storage.mkdir(parents=True, exist_ok=True)

    n_trials = 1000  # Sufficient for single objective
    timeout = 600  # 10 minutes per gauge/dataset
    warmup_years = 2
    overwrite_existing_results = False

    datasets = ["e5l", "gpcp", "e5", "mswep"]

    # Log configuration (to file)
    logger.info("=" * 80)
    logger.info("SIMPLIFIED SINGLE-OBJECTIVE GR4J CALIBRATION")
    logger.info("=" * 80)
    logger.info("Objective: KGE (Kling-Gupta Efficiency)")
    logger.info(f"Number of gauges: {len(full_gauges)}")
    logger.info(f"Datasets: {', '.join(datasets)}")
    logger.info(f"Calibration: {calibration_period[0]} to {calibration_period[1]}")
    logger.info(f"Validation: {validation_period[0]} to {validation_period[1]}")
    logger.info(f"Warm-up period: {warmup_years} years")
    logger.info(f"Trials per gauge: {n_trials} (TPE sampler)")
    logger.info(f"Timeout: {timeout}s ({timeout / 60:.1f} min)")
    logger.info(f"Results: {save_storage}")
    logger.info("=" * 80)

    # Check how many are already complete
    completed_count = 0
    for gauge_id in full_gauges:
        for dataset in datasets:
            result_file = save_storage / gauge_id / f"{gauge_id}_{dataset}_metrics.json"
            if result_file.exists():
                completed_count += 1

    total_tasks = len(full_gauges) * len(datasets)
    logger.info(f"Progress: {completed_count}/{total_tasks} already completed")
    logger.info(f"Remaining: {total_tasks - completed_count} tasks")

    # Process gauges in parallel
    # n_processes = max(1, mp.cpu_count() - 2)  # Leave 2 cores free
    n_processes = 10
    logger.info(f"Starting parallel optimization with {n_processes} processes")

    # Create partial function with fixed parameters
    process_func = partial(
        process_gauge_simple,
        datasets=datasets,
        calibration_period=calibration_period,
        validation_period=validation_period,
        save_storage=save_storage,
        e_obs_gauge=gauges,
        n_trials=n_trials,
        timeout=timeout,
        overwrite_results=overwrite_existing_results,
        warmup_years=warmup_years,
    )

    # Execute in parallel
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        list(executor.map(process_func, full_gauges))

    logger.info("=" * 80)
    logger.info("✅ OPTIMIZATION COMPLETED!")
    logger.info(f"Results saved to: {save_storage}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
