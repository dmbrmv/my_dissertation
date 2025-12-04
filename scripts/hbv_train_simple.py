"""HBV single-objective KGE optimization - SIMPLIFIED approach.

This script uses single-objective optimization for HBV calibration, which is:
- Faster (no Pareto front overhead)
- More interpretable (one best model)
- Standard practice in hydrological modeling
- Better for comparing datasets

Key improvements over multi-objective:
- Reduced to n_trials=1500 (sufficient for single objective)
- Clearer calibration target (KGE only)
- No conflicting objectives
- Parallel execution for speed
- Logs results as each gauge completes
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import logging
import multiprocessing as mp
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import optuna  # type: ignore[import-untyped]
import pandas as pd
import xarray as xr

sys.path.append("./")

from src.models.gr4j.pet import pet_oudin
from src.models.hbv import hbv
from src.timeseries_stats.metrics import evaluate_model, kling_gupta_efficiency
from src.utils.logger import setup_logger

Path("logs").mkdir(exist_ok=True)
logger = setup_logger("hbv_simple", log_file="logs/hbv_simple_optim.log", level="INFO")

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)


def objective_single_kge(
    trial: optuna.Trial,
    data: pd.DataFrame,
    calibration_period: tuple[str, str],
    warmup_years: int = 2,
) -> float:
    """Single-objective KGE optimization for HBV."""
    # HBV parameters (16 total) - aligned with Bergström (1986) reference bounds
    # soil moisture and runoff parameters
    par_beta = trial.suggest_float("parBETA", 1.0, 6.0)
    par_fc = trial.suggest_float("parFC", 50.0, 500.0, log=True)
    par_lp = trial.suggest_float("parLP", 0.3, 1.0)
    # response routing parameters
    par_perc = trial.suggest_float("parPERC", 0.0, 3.0)
    par_uzl = trial.suggest_float("parUZL", 0.0, 500.0)
    par_k0 = trial.suggest_float("parK0", 0.01, 0.4)
    par_k1 = trial.suggest_float("parK1", 0.01, 0.4)
    par_k2 = trial.suggest_float("parK2", 0.001, 0.15, log=True)
    par_maxbas = trial.suggest_float("parMAXBAS", 1.0, 7.0)
    par_cet = trial.suggest_float("parCET", 0.0, 0.3)
    par_pcorr = trial.suggest_float("parPCORR", 0.5, 2.0)
    # snow routing parameters
    par_tt = trial.suggest_float("parTT", -1.5, 2.5)
    par_cfmax = trial.suggest_float("parCFMAX", 1.0, 10.0)
    par_cfr = trial.suggest_float("parCFR", 0.0, 0.1)
    par_cwh = trial.suggest_float("parCWH", 0.0, 0.2)
    par_sfcf = trial.suggest_float("parSFCF", 0.4, 1.0)

    params = [
        par_beta,
        par_cet,
        par_fc,
        par_k0,
        par_k1,
        par_k2,
        par_lp,
        par_maxbas,
        par_perc,
        par_uzl,
        par_pcorr,
        par_tt,
        par_cfmax,
        par_sfcf,
        par_cfr,
        par_cwh,
    ]

    # Calculate warm-up period
    calib_start = pd.to_datetime(calibration_period[0])
    calib_end = pd.to_datetime(calibration_period[1])
    warmup_start = calib_start - pd.DateOffset(years=warmup_years)

    if warmup_start < data.index[0]:
        warmup_start = data.index[0]

    # Run simulation with warm-up
    try:
        warmup_data = data[warmup_start:calib_end]
        q_sim_full = hbv.simulation(warmup_data, params)

        # Extract calibration period (exclude warm-up)
        n_warmup_days = len(data[warmup_start:calib_start]) - 1
        q_sim = q_sim_full[n_warmup_days:]
        q_obs = data[calib_start:calib_end]["q_mm_day"].values

        # Ensure arrays match
        min_len = min(len(q_obs), len(q_sim))
        q_obs = np.asarray(q_obs[:min_len], dtype=float)
        q_sim = np.asarray(q_sim[:min_len], dtype=float)

    except Exception:
        return -999.0

    # Calculate KGE
    kge = kling_gupta_efficiency(q_obs, q_sim)

    if pd.isna(kge):
        return -999.0

    return kge


def process_gauge_simple(
    gauge_id: str,
    datasets: list[str],
    calibration_period: tuple[str, str],
    validation_period: tuple[str, str],
    save_storage: Path,
    e_obs_gauge: gpd.GeoDataFrame,
    n_trials: int = 1500,
    timeout: int = 900,
    overwrite_results: bool = False,
    warmup_years: int = 2,
) -> None:
    """Process gauge with single-objective HBV optimization."""
    result_path = save_storage / gauge_id
    result_path.mkdir(parents=True, exist_ok=True)

    point_geom = e_obs_gauge.loc[gauge_id, "geometry"]
    latitude = float(point_geom.y)  # type: ignore[union-attr]

    for dataset in datasets:
        output_file = result_path / f"{gauge_id}_{dataset}_metrics.json"
        if output_file.exists() and not overwrite_results:
            continue

        try:
            # Load data
            with xr.open_dataset(f"data/nc_all_q/{gauge_id}.nc") as ds:
                df = ds.to_dataframe()

            df["t_mean_e5l"] = (df["t_max_e5l"] + df["t_min_e5l"]) / 2

            # Calculate data start year for warm-up
            calib_start_year = pd.to_datetime(calibration_period[0]).year
            data_start_year = calib_start_year - warmup_years

            # Load extended data
            hbv_data = df.loc[
                f"{data_start_year}" : "2020",
                [
                    "q_mm_day",
                    "t_mean_e5l",
                    f"prcp_{dataset}",
                ],
            ].copy()

            # Ensure day_of_year
            if "day_of_year" not in hbv_data.columns:
                hbv_data["day_of_year"] = hbv_data.index.dayofyear  # type: ignore[attr-defined]

            # Calculate PET
            evap = pet_oudin(
                hbv_data["t_mean_e5l"].tolist(),
                hbv_data["day_of_year"].tolist(),
                latitude,
            )
            hbv_data["evap"] = np.asarray(evap, dtype=float)

            # Rename columns for HBV
            hbv_data.rename(
                columns={
                    "t_mean_e5l": "temp",
                    f"prcp_{dataset}": "prcp",
                },
                inplace=True,
            )

            # Run optimization
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                study_name=f"HBV_simple_{gauge_id}_{dataset}",
            )

            # Define objective with captured variables
            def opt_func(
                trial: optuna.Trial,
                data: pd.DataFrame = hbv_data,
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
                continue

            # Get best parameters IN THE CORRECT ORDER
            # CRITICAL: Must match the order in objective_single_kge params list!
            best_trial_params = study.best_trial.params
            best_params = [
                best_trial_params["parBETA"],
                best_trial_params["parCET"],
                best_trial_params["parFC"],
                best_trial_params["parK0"],
                best_trial_params["parK1"],
                best_trial_params["parK2"],
                best_trial_params["parLP"],
                best_trial_params["parMAXBAS"],
                best_trial_params["parPERC"],
                best_trial_params["parUZL"],
                best_trial_params["parPCORR"],
                best_trial_params["parTT"],
                best_trial_params["parCFMAX"],
                best_trial_params["parSFCF"],
                best_trial_params["parCFR"],
                best_trial_params["parCWH"],
            ]

            # Validate with continuous simulation (preserves model states)
            # Run from warmup start through validation end, then extract validation period
            calib_start = pd.to_datetime(calibration_period[0])
            warmup_start_calib = calib_start - pd.DateOffset(years=warmup_years)
            if warmup_start_calib < hbv_data.index[0]:
                warmup_start_calib = hbv_data.index[0]

            val_end = pd.to_datetime(validation_period[1])
            continuous_data = hbv_data.loc[warmup_start_calib:val_end, :]

            q_sim_full = hbv.simulation(continuous_data, best_params)

            # Extract validation period
            # CRITICAL: Count days in SAME dataframe we're simulating
            n_days_before_val = (
                len(continuous_data[warmup_start_calib : validation_period[0]]) - 1
            )
            q_sim = q_sim_full[n_days_before_val:]

            validation_df = hbv_data.loc[validation_period[0] : validation_period[1], :]
            q_obs = np.asarray(validation_df["q_mm_day"].values, dtype=float)

            # Ensure arrays match
            min_len = min(len(q_obs), len(q_sim))
            q_obs = q_obs[:min_len]
            q_sim = np.asarray(q_sim[:min_len], dtype=float)

            # Calculate metrics
            metrics = evaluate_model(q_obs, q_sim)

            # Save results
            import json

            param_names = [
                "parBETA",
                "parCET",
                "parFC",
                "parK0",
                "parK1",
                "parK2",
                "parLP",
                "parMAXBAS",
                "parPERC",
                "parUZL",
                "parPCORR",
                "parTT",
                "parCFMAX",
                "parSFCF",
                "parCFR",
                "parCWH",
            ]
            best_params_dict = dict(zip(param_names, best_params, strict=True))

            with open(result_path / f"{gauge_id}_{dataset}_params.json", "w") as f:
                json.dump(best_params_dict, f, indent=2)

            with open(output_file, "w") as f:
                json.dump(metrics, f, indent=2)

            # Log success
            logger.info(
                f"✓ {gauge_id}/{dataset}: "
                f"KGE={metrics.get('KGE', np.nan):.3f}, "
                f"NSE={metrics.get('NSE', np.nan):.3f}"
            )

        except Exception as e:
            logger.error(f"✗ {gauge_id}/{dataset}: {str(e)}")
            continue


def main() -> None:
    """Run SIMPLIFIED single-objective HBV calibration."""
    from src.readers.geom_reader import load_geodata

    # Load gauge data
    _, gauges = load_geodata(folder_depth=".")

    full_gauges = [
        i.stem for i in Path("data/nc_all_q").glob("*.nc") if i.stem in gauges.index
    ]

    # Optimization parameters
    calibration_period = ("2010-01-01", "2018-12-31")
    validation_period = ("2019-01-01", "2020-12-31")

    save_storage = Path("data/optimization/hbv_simple/")
    save_storage.mkdir(parents=True, exist_ok=True)

    n_trials = 1500
    timeout = 900  # 15 minutes per gauge/dataset
    warmup_years = 2
    overwrite_existing_results = False

    datasets = ["e5l", "gpcp", "e5", "mswep"]

    # Log configuration
    logger.info("=" * 80)
    logger.info("SIMPLIFIED SINGLE-OBJECTIVE HBV CALIBRATION")
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

    # Check progress
    completed_count = 0
    for gauge_id in full_gauges:
        for dataset in datasets:
            output_file = save_storage / gauge_id / f"{gauge_id}_{dataset}_metrics.json"
            if output_file.exists():
                completed_count += 1

    total_tasks = len(full_gauges) * len(datasets)
    logger.info(f"Progress: {completed_count}/{total_tasks} already completed")
    logger.info(f"Remaining: {total_tasks - completed_count} tasks")

    # Process gauges in parallel
    n_processes = max(1, mp.cpu_count() - 2)
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
