"""Random Forest single-objective KGE optimization - SIMPLIFIED approach.

This script uses single-objective optimization for Random Forest calibration:
- Faster (no Pareto front overhead)
- Standard ML practice (sklearn uses single metric)
- More interpretable (one best model)
- Better for comparing datasets

Key improvements over multi-objective:
- Reduced to n_trials=100 (sufficient for RF hyperparameters)
- Clearer calibration target (KGE only)
- No artificial flow regime conflicts (RF has no physical trade-offs)
- Parallel execution for speed
- Progress tracking with tqdm
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import logging
import multiprocessing as mp
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import optuna  # type: ignore[import-untyped]
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]
import xarray as xr

sys.path.append("./")

from src.models.gr4j.pet import pet_oudin
from src.utils.logger import setup_logger
from src.utils.metrics import evaluate_model, kling_gupta_efficiency

Path("logs").mkdir(exist_ok=True)
logger = setup_logger("rfr_simple", log_file="logs/rfr_simple_optim.log", level="INFO")

# Suppress verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)


def create_temporal_features(
    df: pd.DataFrame,
    rolling_windows: list[int],
    latitude: float,
) -> pd.DataFrame:
    """Create temporal features for Random Forest.

    Features:
    - PET (Potential Evapotranspiration) via Oudin formula
    - Cyclic day-of-year encoding (sin/cos)
    - Rolling window aggregations (mean of precipitation/temperature)

    Args:
        df: DataFrame with columns [prcp, t_mean, q_mm_day]
        rolling_windows: List of window sizes in days
        latitude: Gauge latitude for PET calculation

    Returns:
        DataFrame with added temporal features
    """
    result = df.copy()

    # Calculate PET
    if "day_of_year" not in result.columns:
        result["day_of_year"] = result.index.dayofyear  # type: ignore[attr-defined]

    pet = pet_oudin(
        result["t_mean"].tolist(),
        result["day_of_year"].tolist(),
        latitude,
    )
    result["pet"] = np.asarray(pet, dtype=float)

    # Cyclic temporal encoding (day of year)
    doy = result["day_of_year"].values
    doy_array = np.asarray(doy, dtype=float)
    result["doy_sin"] = np.sin(2 * np.pi * doy_array / 365.25)
    result["doy_cos"] = np.cos(2 * np.pi * doy_array / 365.25)

    # Rolling window features
    for window in rolling_windows:
        result[f"prcp_roll_{window}"] = (
            result["prcp"].rolling(window=window, min_periods=1).mean()
        )
        result[f"t_mean_roll_{window}"] = (
            result["t_mean"].rolling(window=window, min_periods=1).mean()
        )

    return result


def objective_single_kge(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Single-objective KGE optimization for Random Forest."""
    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7])

    try:
        # Train model
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
        )

        rf.fit(X_train, y_train)

        # Predict on validation set
        y_pred = rf.predict(X_val)

        # Calculate KGE
        kge = kling_gupta_efficiency(y_val, y_pred)

        if pd.isna(kge):
            return -999.0

        return kge

    except Exception:
        return -999.0


def process_gauge_simple(
    gauge_id: str,
    datasets: list[str],
    calibration_period: tuple[str, str],
    validation_period: tuple[str, str],
    save_storage: Path,
    gauges_gdf: gpd.GeoDataFrame,
    static_file: Path,
    static_columns: list[str],
    rolling_windows: list[int],
    n_trials: int = 100,
    timeout: int = 1800,
    overwrite_results: bool = False,
) -> None:
    """Process gauge with single-objective RF optimization."""
    result_path = save_storage / gauge_id
    result_path.mkdir(parents=True, exist_ok=True)

    # Get gauge location
    point_geom = gauges_gdf.loc[gauge_id, "geometry"]
    latitude = float(point_geom.y)  # type: ignore[union-attr]

    # Load static attributes
    try:
        static_df = pd.read_csv(static_file, dtype={"gauge_id": str})
        static_df.set_index("gauge_id", inplace=True)

        if gauge_id not in static_df.index:
            logger.warning(f"Gauge {gauge_id} not in static attributes")
            return

        static_attrs = static_df.loc[gauge_id][static_columns].values.astype(float)
    except Exception as e:
        logger.error(f"Failed to load static attributes for {gauge_id}: {e}")
        return

    for dataset in datasets:
        output_file = result_path / f"{gauge_id}_{dataset}_metrics.json"
        if output_file.exists() and not overwrite_results:
            continue

        try:
            # Load data
            with xr.open_dataset(f"data/nc_all_q/{gauge_id}.nc") as ds:
                df = ds.to_dataframe()

            df["t_mean"] = (df["t_max_e5l"] + df["t_min_e5l"]) / 2

            # Select relevant columns
            rf_data = df.loc[
                "2008":"2020",
                [
                    "q_mm_day",
                    "t_mean",
                    f"prcp_{dataset}",
                ],
            ].copy()

            rf_data.rename(columns={f"prcp_{dataset}": "prcp"}, inplace=True)

            # Create features
            rf_data = create_temporal_features(rf_data, rolling_windows, latitude)

            # Drop NaN values
            rf_data.dropna(inplace=True)

            # Split into calibration and validation
            calib_data = rf_data.loc[
                calibration_period[0] : calibration_period[1], :
            ].copy()
            val_data = rf_data.loc[validation_period[0] : validation_period[1], :].copy()

            if len(calib_data) < 100 or len(val_data) < 30:
                continue

            # Prepare features
            feature_cols = [
                col for col in rf_data.columns if col not in ["q_mm_day", "day_of_year"]
            ]

            X_train = calib_data[feature_cols].values
            y_train = calib_data["q_mm_day"].values

            X_val = val_data[feature_cols].values
            y_val = val_data["q_mm_day"].values

            # Add static features
            n_train = X_train.shape[0]
            n_val = X_val.shape[0]

            static_train = np.tile(static_attrs, (n_train, 1))
            static_val = np.tile(static_attrs, (n_val, 1))

            X_train = np.hstack([X_train, static_train])
            X_val = np.hstack([X_val, static_val])

            # Run optimization
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                study_name=f"RF_simple_{gauge_id}_{dataset}",
            )

            # Define objective with captured variables
            def opt_func(
                trial: optuna.Trial,
                X_tr: np.ndarray = X_train,
                y_tr: np.ndarray = y_train,
                X_v: np.ndarray = X_val,
                y_v: np.ndarray = y_val,
            ) -> float:
                return objective_single_kge(trial, X_tr, y_tr, X_v, y_v)

            study.optimize(
                opt_func,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=False,
            )

            if study.best_trial is None:
                continue

            # Train final model with best parameters
            best_params = study.best_trial.params

            rf_final = RandomForestRegressor(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["max_depth"],
                min_samples_split=best_params["min_samples_split"],
                min_samples_leaf=best_params["min_samples_leaf"],
                max_features=best_params["max_features"],
                random_state=42,
                n_jobs=-1,
            )

            rf_final.fit(X_train, y_train)

            # Predict on validation
            y_pred = rf_final.predict(X_val)

            # Calculate metrics
            metrics = evaluate_model(y_val, y_pred)

            # Save results
            with open(result_path / f"{gauge_id}_{dataset}_params.json", "w") as f:
                json.dump(best_params, f, indent=2)

            with open(output_file, "w") as f:
                json.dump(metrics, f, indent=2)

            logger.info(
                f"✓ {gauge_id}/{dataset}: "
                f"KGE={metrics.get('KGE', np.nan):.3f}, "
                f"NSE={metrics.get('NSE', np.nan):.3f}"
            )

        except Exception as e:
            logger.error(f"✗ {gauge_id}/{dataset}: {str(e)}")
            continue


def main() -> None:
    """Run SIMPLIFIED single-objective Random Forest calibration."""
    from src.readers.geom_reader import load_geodata

    # Load gauge data
    _, gauges = load_geodata(folder_depth=".")

    full_gauges = [
        i.stem for i in Path("data/nc_all_q").glob("*.nc") if i.stem in gauges.index
    ]

    # Configuration
    calibration_period = ("2008-01-01", "2018-12-31")
    validation_period = ("2019-01-01", "2020-12-31")

    save_storage = Path("data/optimization/rfr_simple/")
    save_storage.mkdir(parents=True, exist_ok=True)

    # Rolling windows
    rolling_windows = [1, 2, 4, 8, 16, 32]

    # Static attributes
    static_file = Path("data/attributes/hydro_atlas_cis_camels.csv")
    static_parameters = [
        "for_pc_sse",
        "crp_pc_sse",
        "inu_pc_ult",
        "ire_pc_sse",
        "lka_pc_use",
        "prm_pc_sse",
        "pst_pc_sse",
        "cly_pc_sav",
        "slt_pc_sav",
        "snd_pc_sav",
        "kar_pc_sse",
        "urb_pc_sse",
        "gwt_cm_sav",
        "lkv_mc_usu",
        "rev_mc_usu",
        "ws_area",
        "ele_mt_sav",
    ]

    if not static_file.exists():
        logger.error(f"Static attributes file not found: {static_file}")
        return

    n_trials = 100
    timeout = 1800  # 30 minutes
    overwrite_existing_results = False

    datasets = ["e5l", "gpcp", "e5", "mswep"]

    # Log configuration
    logger.info("=" * 80)
    logger.info("SIMPLIFIED SINGLE-OBJECTIVE RANDOM FOREST CALIBRATION")
    logger.info("=" * 80)
    logger.info("Objective: KGE (Kling-Gupta Efficiency)")
    logger.info(f"Number of gauges: {len(full_gauges)}")
    logger.info(f"Datasets: {', '.join(datasets)}")
    logger.info(f"Calibration: {calibration_period[0]} to {calibration_period[1]}")
    logger.info(f"Validation: {validation_period[0]} to {validation_period[1]}")
    logger.info(f"Rolling windows: {rolling_windows}")
    logger.info(f"Static features: {len(static_parameters)}")
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

    # Create partial function with fixed parameters (already imported at top)
    process_func = partial(
        process_gauge_simple,
        datasets=datasets,
        calibration_period=calibration_period,
        validation_period=validation_period,
        save_storage=save_storage,
        gauges_gdf=gauges,
        static_file=static_file,
        static_columns=static_parameters,
        rolling_windows=rolling_windows,
        n_trials=n_trials,
        timeout=timeout,
        overwrite_results=overwrite_existing_results,
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
