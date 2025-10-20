"""HBV single gauge debug script with expanded logging.

This script runs HBV calibration for ONE gauge with comprehensive logging
to identify the exact function causing errors.
"""

import json
import logging
from pathlib import Path
import sys
import traceback

import numpy as np
import optuna  # type: ignore[import-untyped]
import pandas as pd
import xarray as xr

sys.path.append("./")

from src.models.gr4j.pet import pet_oudin
from src.models.hbv import hbv
from src.utils.logger import setup_logger
from src.utils.metrics import evaluate_model, kling_gupta_efficiency

# Setup logging
Path("logs").mkdir(exist_ok=True)
logger = setup_logger(
    "hbv_debug",
    log_file="logs/hbv_debug_single.log",
    level="DEBUG",
)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.ERROR)
logging.getLogger("optuna").setLevel(logging.ERROR)


def objective_single_kge(
    trial: optuna.Trial,
    data: pd.DataFrame,
    calibration_period: tuple[str, str],
    warmup_years: int = 2,
) -> float:
    """Single-objective KGE optimization for HBV with detailed logging."""
    trial_number = trial.number
    logger.debug(f"[Trial {trial_number}] Starting objective function")

    try:
        # Sample parameters
        logger.debug(f"[Trial {trial_number}] Sampling parameters")
        par_beta = trial.suggest_float("parBETA", 1.0, 6.0)
        par_fc = trial.suggest_float("parFC", 50.0, 500.0, log=True)
        par_lp = trial.suggest_float("parLP", 0.3, 1.0)
        par_perc = trial.suggest_float("parPERC", 0.0, 3.0)
        par_uzl = trial.suggest_float("parUZL", 0.0, 500.0)
        par_k0 = trial.suggest_float("parK0", 0.01, 0.4)
        par_k1 = trial.suggest_float("parK1", 0.01, 0.4)
        par_k2 = trial.suggest_float("parK2", 0.001, 0.15, log=True)
        par_maxbas = trial.suggest_float("parMAXBAS", 1.0, 7.0)
        par_cet = trial.suggest_float("parCET", 0.0, 0.3)
        par_pcorr = trial.suggest_float("parPCORR", 0.5, 2.0)
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
        logger.debug(f"[Trial {trial_number}] Parameters sampled successfully")

        # Calculate warm-up period
        logger.debug(f"[Trial {trial_number}] Calculating warmup period")
        calib_start = pd.to_datetime(calibration_period[0])
        calib_end = pd.to_datetime(calibration_period[1])
        warmup_start = calib_start - pd.DateOffset(years=warmup_years)

        if warmup_start < data.index[0]:
            logger.debug(
                f"[Trial {trial_number}] Adjusting warmup_start "
                f"from {warmup_start} to {data.index[0]}"
            )
            warmup_start = data.index[0]

        logger.debug(
            f"[Trial {trial_number}] Warmup: {warmup_start} to {calib_start}, "
            f"Calibration: {calib_start} to {calib_end}"
        )

        # Extract warmup data
        logger.debug(f"[Trial {trial_number}] Extracting warmup data slice")
        warmup_data = data[warmup_start:calib_end]
        logger.debug(
            f"[Trial {trial_number}] Warmup data shape: {warmup_data.shape}, "
            f"columns: {warmup_data.columns.tolist()}"
        )

        # Check for missing values
        missing_count = warmup_data.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(
                f"[Trial {trial_number}] Found {missing_count} missing values "
                f"in warmup data"
            )

        # Run simulation
        logger.debug(f"[Trial {trial_number}] Starting HBV simulation")
        q_sim_full = hbv.simulation(warmup_data, params)
        logger.debug(
            f"[Trial {trial_number}] Simulation complete. "
            f"Output length: {len(q_sim_full)}"
        )

        # Extract calibration period
        logger.debug(f"[Trial {trial_number}] Extracting calibration period")
        n_warmup_days = len(data[warmup_start:calib_start]) - 1
        logger.debug(f"[Trial {trial_number}] Warmup days to skip: {n_warmup_days}")

        q_sim = q_sim_full[n_warmup_days:]
        q_obs = data[calib_start:calib_end]["q_mm_day"].values

        logger.debug(
            f"[Trial {trial_number}] Sim length: {len(q_sim)}, Obs length: {len(q_obs)}"
        )

        # Ensure arrays match
        min_len = min(len(q_obs), len(q_sim))
        if min_len != len(q_obs) or min_len != len(q_sim):
            logger.warning(
                f"[Trial {trial_number}] Array length mismatch. Truncating to {min_len}"
            )

        q_obs = np.asarray(q_obs[:min_len], dtype=float)
        q_sim = np.asarray(q_sim[:min_len], dtype=float)

        # Check for invalid values
        if np.any(np.isnan(q_obs)) or np.any(np.isnan(q_sim)):
            nan_obs = np.sum(np.isnan(q_obs))
            nan_sim = np.sum(np.isnan(q_sim))
            logger.warning(
                f"[Trial {trial_number}] Found NaN values: obs={nan_obs}, sim={nan_sim}"
            )

        # Calculate KGE
        logger.debug(f"[Trial {trial_number}] Calculating KGE")
        kge = kling_gupta_efficiency(q_obs, q_sim)
        logger.debug(f"[Trial {trial_number}] KGE = {kge:.4f}")

        if pd.isna(kge):
            logger.warning(f"[Trial {trial_number}] KGE is NaN, returning -999.0")
            return -999.0

        return float(kge)

    except Exception as e:
        logger.error(
            f"[Trial {trial_number}] Exception in objective function: {e}",
            exc_info=True,
        )
        return -999.0


def _load_and_prepare_data(
    gauge_id: str,
    dataset: str,
    data_start_year: int,
    latitude: float,
) -> pd.DataFrame | None:
    """Load and prepare HBV input data with logging.

    Args:
        gauge_id: Gauge identifier.
        dataset: Dataset name (e5l, gpcp, etc.).
        data_start_year: Start year for data extraction.
        latitude: Gauge latitude for PET calculation.

    Returns:
        Prepared DataFrame or None if error occurs.
    """
    # Step 1: Load data
    logger.info("[Step 1/8] Loading NetCDF dataset")
    nc_path = Path(f"data/nc_all_q/{gauge_id}.nc")
    if not nc_path.exists():
        logger.error(f"NetCDF file not found: {nc_path}")
        return None

    logger.debug(f"Opening dataset: {nc_path}")
    with xr.open_dataset(nc_path) as ds:
        logger.debug(f"Dataset variables: {list(ds.data_vars)}")
        logger.debug(f"Dataset coordinates: {list(ds.coords)}")
        df = ds.to_dataframe()

    logger.info(f"Loaded dataframe shape: {df.shape}")
    logger.debug(f"Dataframe columns: {df.columns.tolist()}")
    logger.debug(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Step 2: Calculate mean temperature
    logger.info("[Step 2/8] Calculating mean temperature")
    if "t_max_e5l" not in df.columns or "t_min_e5l" not in df.columns:
        logger.error("Required temperature columns not found")
        return None

    df["t_mean_e5l"] = (df["t_max_e5l"] + df["t_min_e5l"]) / 2
    logger.debug(
        f"Temperature range: "
        f"min={df['t_mean_e5l'].min():.2f}, "
        f"max={df['t_mean_e5l'].max():.2f}"
    )

    # Step 3: Slice data
    logger.info("[Step 4/8] Slicing data for calibration")
    prcp_col = f"prcp_{dataset}"
    if prcp_col not in df.columns:
        logger.error(f"Precipitation column not found: {prcp_col}")
        logger.debug(f"Available columns: {df.columns.tolist()}")
        return None

    required_cols = ["q_mm_day", "t_mean_e5l", prcp_col]
    hbv_data = df.loc[f"{data_start_year}" : "2020", required_cols].copy()
    logger.info(f"HBV data shape: {hbv_data.shape}")

    # Check for missing data
    missing = hbv_data.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"Missing data found:\n{missing}")

    # Step 4: Add day_of_year
    logger.info("[Step 5/8] Adding day_of_year column")
    if "day_of_year" not in hbv_data.columns:
        hbv_data["day_of_year"] = hbv_data.index.dayofyear  # type: ignore[attr-defined]
    logger.debug(
        f"Day of year range: "
        f"{hbv_data['day_of_year'].min()} to {hbv_data['day_of_year'].max()}"
    )

    # Step 5: Calculate PET
    logger.info("[Step 6/8] Calculating potential evapotranspiration")
    logger.debug(f"Using Oudin method with latitude={latitude:.4f}")
    evap = pet_oudin(
        hbv_data["t_mean_e5l"].tolist(),
        hbv_data["day_of_year"].tolist(),
        latitude,
    )
    hbv_data["Evap"] = np.asarray(evap, dtype=float)
    logger.debug(
        f"PET range: min={hbv_data['Evap'].min():.2f}, max={hbv_data['Evap'].max():.2f}"
    )

    # Step 6: Rename columns
    logger.info("[Step 7/8] Renaming columns for HBV model")
    hbv_data.rename(
        columns={
            "t_mean_e5l": "Temp",
            prcp_col: "Prec",
        },
        inplace=True,
    )
    logger.debug(f"Final HBV data columns: {hbv_data.columns.tolist()}")

    # Verify required columns
    required_hbv_cols = ["Temp", "Prec", "Evap", "q_mm_day"]
    for col in required_hbv_cols:
        if col not in hbv_data.columns:
            logger.error(f"Missing required column: {col}")
            return None

    return hbv_data


def _run_validation(
    hbv_data: pd.DataFrame,
    best_params: list[float],
    calibration_period: tuple[str, str],
    validation_period: tuple[str, str],
    warmup_years: int,
) -> dict[str, float]:
    """Run validation simulation with states from calibration.

    Args:
        hbv_data: Prepared HBV input data.
        best_params: Calibrated model parameters.
        calibration_period: Calibration period (to establish initial states).
        validation_period: Validation start and end dates.
        warmup_years: Number of warmup years before calibration.

    Returns:
        Dictionary of validation metrics.
    """
    logger.info("=" * 80)
    logger.info("VALIDATION PHASE")
    logger.info("=" * 80)

    logger.info(f"Validation period: {validation_period[0]} to {validation_period[1]}")

    # Run continuous simulation from calibration start through validation end
    # This ensures model states are properly initialized from calibration period
    calib_start = pd.to_datetime(calibration_period[0])
    warmup_start = calib_start - pd.DateOffset(years=warmup_years)
    if warmup_start < hbv_data.index[0]:
        warmup_start = hbv_data.index[0]

    val_end = pd.to_datetime(validation_period[1])

    logger.info(f"Running continuous simulation: {warmup_start} to {val_end}")
    logger.info("(This establishes proper model states for validation)")

    continuous_data = hbv_data.loc[warmup_start:val_end, :]
    logger.debug(f"Continuous simulation data shape: {continuous_data.shape}")

    # Check for missing values
    missing_val = continuous_data.isnull().sum()
    if missing_val.sum() > 0:
        logger.warning(f"Missing values in continuous data:\n{missing_val}")

    logger.info("Running continuous HBV simulation")
    q_sim_full = hbv.simulation(continuous_data, best_params)

    # Extract only the validation period for evaluation
    # Calculate how many days to skip based on the CONTINUOUS simulation data
    n_days_before_validation = (
        len(continuous_data[warmup_start : validation_period[0]]) - 1
    )
    q_sim = q_sim_full[n_days_before_validation:]

    logger.info(
        f"Extracted validation period: {len(q_sim)} days "
        f"(skipped {n_days_before_validation} days before validation)"
    )

    # Get observed data for validation period
    validation_df = hbv_data.loc[validation_period[0] : validation_period[1], :]

    q_obs = np.asarray(validation_df["q_mm_day"].values, dtype=float)

    logger.debug(f"Validation obs length: {len(q_obs)}")
    logger.debug(f"Validation sim length: {len(q_sim)}")

    # DIAGNOSTIC: Check for extreme values
    logger.info("=" * 80)
    logger.info("DIAGNOSTIC: Checking simulated values")
    logger.info("=" * 80)
    logger.info("Observed discharge stats:")
    logger.info(f"  Min:    {np.min(q_obs):.4f} mm/day")
    logger.info(f"  Max:    {np.max(q_obs):.4f} mm/day")
    logger.info(f"  Mean:   {np.mean(q_obs):.4f} mm/day")
    logger.info(f"  Median: {np.median(q_obs):.4f} mm/day")
    logger.info(f"  Std:    {np.std(q_obs):.4f} mm/day")

    logger.info("Simulated discharge stats:")
    logger.info(f"  Min:    {np.min(q_sim):.4f} mm/day")
    logger.info(f"  Max:    {np.max(q_sim):.4f} mm/day")
    logger.info(f"  Mean:   {np.mean(q_sim):.4f} mm/day")
    logger.info(f"  Median: {np.median(q_sim):.4f} mm/day")
    logger.info(f"  Std:    {np.std(q_sim):.4f} mm/day")

    # Check for NaN or infinite values
    nan_count_obs = np.sum(np.isnan(q_obs))
    nan_count_sim = np.sum(np.isnan(q_sim))
    inf_count_sim = np.sum(np.isinf(q_sim))

    if nan_count_obs > 0:
        logger.error(f"Found {nan_count_obs} NaN values in observed data!")
    if nan_count_sim > 0:
        logger.error(f"Found {nan_count_sim} NaN values in simulated data!")
    if inf_count_sim > 0:
        logger.error(f"Found {inf_count_sim} infinite values in simulated data!")

    # Check for unrealistic values
    max_sim = np.max(q_sim)
    min_sim = np.min(q_sim)
    if max_sim > 100:
        logger.warning(f"⚠️  Extremely high simulated discharge: {max_sim:.2f} mm/day")
    if min_sim < -0.1:
        logger.warning(f"⚠️  Negative simulated discharge: {min_sim:.4f} mm/day")

    logger.info("=" * 80)

    # Ensure arrays match
    min_len = min(len(q_obs), len(q_sim))
    q_obs = q_obs[:min_len]
    q_sim = np.asarray(q_sim[:min_len], dtype=float)

    logger.info("Calculating validation metrics")
    metrics = evaluate_model(q_obs, q_sim)

    logger.info("Validation metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    return metrics


def process_single_gauge(
    gauge_id: str,
    dataset: str,
    calibration_period: tuple[str, str],
    validation_period: tuple[str, str],
    save_storage: Path,
    latitude: float,
    n_trials: int = 100,
    timeout: int = 300,
    warmup_years: int = 2,
) -> dict[str, float] | None:
    """Process single gauge with comprehensive logging."""
    logger.info("=" * 80)
    logger.info(f"STARTING GAUGE: {gauge_id}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Latitude: {latitude:.4f}")
    logger.info("=" * 80)

    result_path = save_storage / gauge_id
    result_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load and prepare data
        calib_start_year = pd.to_datetime(calibration_period[0]).year
        data_start_year = calib_start_year - warmup_years
        logger.info("[Step 3/8] Determining data slice")
        logger.debug(f"Data start year: {data_start_year}")

        hbv_data = _load_and_prepare_data(
            gauge_id=gauge_id,
            dataset=dataset,
            data_start_year=data_start_year,
            latitude=latitude,
        )

        if hbv_data is None:
            return None

        # Step 8: Run optimization
        logger.info("[Step 8/8] Starting Optuna optimization")
        logger.info(f"Number of trials: {n_trials}")
        logger.info(f"Timeout: {timeout}s")

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"HBV_debug_{gauge_id}_{dataset}",
        )

        logger.info("Running optimization...")

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
            show_progress_bar=True,
        )

        if study.best_trial is None:
            logger.error("No best trial found")
            return None

        logger.info(f"Best KGE (calibration): {study.best_value:.4f}")
        logger.info(f"Best trial number: {study.best_trial.number}")

        # Get best parameters IN THE CORRECT ORDER (same as in objective function)
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
        logger.debug(f"Best parameters (ordered): {best_params}")

        # Run validation
        metrics = _run_validation(
            hbv_data=hbv_data,
            best_params=best_params,
            calibration_period=calibration_period,
            validation_period=validation_period,
            warmup_years=warmup_years,
        )

        # Save results
        logger.info("Saving results")

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

        params_file = result_path / f"{gauge_id}_{dataset}_params.json"
        with open(params_file, "w") as f:
            json.dump(best_params_dict, f, indent=2)
        logger.debug(f"Saved parameters to {params_file}")

        metrics_file = result_path / f"{gauge_id}_{dataset}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.debug(f"Saved metrics to {metrics_file}")

        logger.info("=" * 80)
        logger.info(f"✅ SUCCESS: {gauge_id}/{dataset}")
        logger.info(f"Validation KGE: {metrics.get('KGE', np.nan):.4f}")
        logger.info(f"Validation NSE: {metrics.get('NSE', np.nan):.4f}")
        logger.info("=" * 80)

        return metrics

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"✗ FATAL ERROR: {gauge_id}/{dataset}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {e}")
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        logger.error("=" * 80)
        return None


def main() -> None:
    """Run debug script for single gauge."""
    from src.readers.geom_reader import load_geodata

    logger.info("=" * 80)
    logger.info("HBV DEBUG SCRIPT - SINGLE GAUGE WITH EXPANDED LOGGING")
    logger.info("=" * 80)

    # Load gauge data
    logger.info("Loading gauge geometries")
    _, gauges = load_geodata(folder_depth=".")
    logger.info(f"Loaded {len(gauges)} gauges")

    # Get available gauges
    full_gauges = [
        i.stem for i in Path("data/nc_all_q").glob("*.nc") if i.stem in gauges.index
    ]
    logger.info(f"Found {len(full_gauges)} gauges with data")

    if not full_gauges:
        logger.error("No gauges found!")
        return

    # Select first gauge for debugging
    gauge_id = full_gauges[0]
    logger.info(f"Selected gauge for debugging: {gauge_id}")

    point_geom = gauges.loc[gauge_id, "geometry"]
    latitude = float(point_geom.y)  # type: ignore[union-attr]

    # Configuration
    calibration_period = ("2010-01-01", "2018-12-31")
    validation_period = ("2019-01-01", "2020-12-31")
    save_storage = Path("data/optimization/hbv_debug/")
    save_storage.mkdir(parents=True, exist_ok=True)

    # Use fewer trials for debugging, increase once model works
    n_trials = 200  # Increased from 100 to get better parameters
    timeout = 600  # 10 minutes
    warmup_years = 2

    # Test with first dataset only
    dataset = "e5l"

    logger.info("Configuration:")
    logger.info(f"  Gauge: {gauge_id}")
    logger.info(f"  Dataset: {dataset}")
    logger.info(f"  Latitude: {latitude:.4f}")
    logger.info(f"  Calibration: {calibration_period[0]} to {calibration_period[1]}")
    logger.info(f"  Validation: {validation_period[0]} to {validation_period[1]}")
    logger.info(f"  Trials: {n_trials}")
    logger.info(f"  Timeout: {timeout}s")
    logger.info(f"  Warmup: {warmup_years} years")

    # Run processing
    metrics = process_single_gauge(
        gauge_id=gauge_id,
        dataset=dataset,
        calibration_period=calibration_period,
        validation_period=validation_period,
        save_storage=save_storage,
        latitude=latitude,
        n_trials=n_trials,
        timeout=timeout,
        warmup_years=warmup_years,
    )

    if metrics is not None:
        logger.info("\n" + "=" * 80)
        logger.info("FINAL RESULTS")
        logger.info("=" * 80)
        for key, value in metrics.items():
            logger.info(f"{key:20s}: {value:.4f}")
    else:
        logger.error("\n" + "=" * 80)
        logger.error("PROCESSING FAILED")
        logger.error("Check logs above for detailed error information")
        logger.error("=" * 80)


if __name__ == "__main__":
    main()
