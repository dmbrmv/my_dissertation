"""Parallelization utilities for RFR Optuna optimization."""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from src.models.gr4j.pet import pet_oudin
from src.models.rfr.pareto import save_optimization_results, select_best_trial_weighted
from src.models.rfr.rfr_optuna import (
    create_temporal_features,
    run_optimization,
    train_final_model,
)
from src.utils.logger import setup_logger
from timeseries_stats.metrics import evaluate_model

logger = setup_logger("rfr_parallel", log_file="logs/rfr_parallel.log")


def run_parallel_optimization(
    gauge_ids: list[str],
    process_gauge_func,
    n_processes: int | None = None,
    **kwargs,
) -> None:
    """Run RFR optimization in parallel for multiple gauges.

    Args:
        gauge_ids: List of gauge identifiers
        process_gauge_func: Function to process a single gauge
        n_processes: Number of processes to use (None = auto-detect)
        **kwargs: Additional arguments for process_gauge_func
    """
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    n_processes = max(1, min(n_processes, mp.cpu_count()))

    logger.info(f"Starting parallel RFR optimization with {n_processes} processes")

    process_func = partial(process_gauge_func, **kwargs)

    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        list(executor.map(process_func, gauge_ids))

    logger.info("Parallel RFR optimization completed")


def _load_static_attributes(
    static_file: Path, gauge_id: str, static_columns: list[str]
) -> np.ndarray | None:
    """Load static watershed attributes for a gauge.

    Args:
        static_file: Path to static attributes CSV
        gauge_id: Gauge identifier
        static_columns: List of column names to load

    Returns:
        1D array of static features, or None if gauge not found
    """
    try:
        static_df = pd.read_csv(static_file, dtype={"gauge_id": str})
        static_df.set_index("gauge_id", inplace=True)

        if gauge_id not in static_df.index:
            logger.warning(f"Gauge {gauge_id} not found in static attributes file")
            return None

        static_row = static_df.loc[gauge_id][static_columns]
        static_values = static_row.values.astype(float)
        return static_values  # type: ignore[return-value]
    except Exception as e:
        logger.error(f"Error loading static attributes for {gauge_id}: {e}")
        return None


def process_rfr_gauge(
    gauge_id: str,
    datasets: list[str],
    calibration_period: tuple[str, str],
    validation_period: tuple[str, str],
    save_storage: Path,
    gauges_gdf: gpd.GeoDataFrame,
    static_file: Path,
    static_columns: list[str],
    rolling_windows: list[int] | None = None,
    n_trials: int = 100,
    timeout: int = 3600,
    n_jobs: int = -1,
    overwrite_results: bool = False,
) -> None:
    """Process a single gauge across multiple datasets with Random Forest.

    Loads hydro and meteo data, creates temporal features (with PET and cyclic DOY),
    adds static watershed attributes, runs model optimization, and evaluates
    performance on validation period.

    Args:
        gauge_id: Gauge identifier
        datasets: List of meteorological dataset names (e.g., ["e5l", "imerg", "gpm"])
        calibration_period: (start_date, end_date) for calibration (YYYY-MM-DD)
        validation_period: (start_date, end_date) for validation (YYYY-MM-DD)
        save_storage: Root directory to save results
        gauges_gdf: GeoDataFrame with gauge geometries (for latitude extraction)
        static_file: Path to static attributes CSV file
        static_columns: List of static attribute column names to use
        rolling_windows: List of rolling window sizes in days (default: [1,2,4,8,16,32])
        n_trials: Number of Optuna trials
        timeout: Optimization timeout in seconds
        n_jobs: Number of parallel jobs for RF training (-1 = all cores)
        overwrite_results: Whether to overwrite existing results
    """
    result_path = save_storage / gauge_id
    result_path.mkdir(parents=True, exist_ok=True)

    # Default rolling windows
    if rolling_windows is None:
        rolling_windows = [2**n for n in range(6)]  # [1, 2, 4, 8, 16, 32]

    # Common weights for all datasets
    rfr_weights: dict[str, float] = {
        "KGE": 0.25,
        "low_flow": 0.35,
        "high_flow": 0.30,
        "PBIAS": 0.10,
    }

    # Extract latitude for PET calculation
    if gauge_id not in gauges_gdf.index:
        logger.warning(f"Gauge {gauge_id} not found in GeoDataFrame, skipping")
        return

    point_geom = gauges_gdf.loc[gauge_id, "geometry"]
    latitude = float(point_geom.y)  # type: ignore[union-attr]

    # Load static attributes (once per gauge, used for all datasets)
    static_attrs = _load_static_attributes(static_file, gauge_id, static_columns)
    if static_attrs is None:
        logger.warning(f"Skipping {gauge_id}: static attributes not available")
        return

    for dataset in datasets:
        logger.info(f"Processing gauge {gauge_id} with dataset {dataset}")

        # Check if results already exist
        if (
            result_path / f"{gauge_id}_{dataset}" / "best_model.joblib"
        ).exists() and not overwrite_results:
            logger.info(f"Results for {gauge_id} with dataset {dataset} exist. Skipping.")
            continue

        try:
            # Load meteorological data
            data_path = Path(f"data/nc_all_q/{gauge_id}.nc")
            if not data_path.exists():
                logger.warning(f"Data file not found for gauge {gauge_id}, skipping")
                continue

            with xr.open_dataset(data_path) as ds:
                df = ds.to_dataframe()

            # Calculate mean temperature if needed
            if "t_mean_e5l" not in df.columns:
                df["t_mean_e5l"] = (df["t_max_e5l"] + df["t_min_e5l"]) / 2

            # Select relevant columns and time period
            required_cols = [
                "q_mm_day",
                "t_min_e5l",
                "t_max_e5l",
                f"prcp_{dataset}",
            ]

            # Check if all columns exist
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(
                    f"Missing columns {missing_cols} for {gauge_id} "
                    f"with dataset {dataset}, skipping"
                )
                continue

            rfr_data = df.loc["2008":"2020", required_cols].copy()

            # Calculate PET using Oudin formula
            t_mean_list = rfr_data["t_mean_e5l"].tolist()
            day_of_year_list = rfr_data.index.dayofyear.tolist()  # type: ignore[attr-defined]
            pet_values = pet_oudin(t_mean_list, day_of_year_list, latitude)

            # Create temporal features (includes PET and cyclic DOY encoding)
            logger.info("Creating temporal features with PET and DOY encoding...")
            base_features = [f"prcp_{dataset}", "t_min_e5l", "t_max_e5l"]
            rfr_data_features = create_temporal_features(
                rfr_data,
                rolling_windows=rolling_windows,
                base_features=base_features,
                pet_values=pet_values,
            )

            # Split into calibration and validation
            calib_start, calib_end = calibration_period
            val_start, val_end = validation_period

            calib_data = rfr_data_features.loc[calib_start:calib_end]
            val_data = rfr_data_features.loc[val_start:val_end]

            if len(calib_data) == 0 or len(val_data) == 0:
                logger.warning(
                    f"Empty calibration/validation data for {gauge_id} "
                    f"with dataset {dataset}"
                )
                continue

            # Prepare features and targets
            feature_cols = [col for col in calib_data.columns if col not in ["q_mm_day"]]

            x_train = calib_data[feature_cols].to_numpy()  # noqa: N806
            y_train = calib_data["q_mm_day"].to_numpy()
            x_val = val_data[feature_cols].to_numpy()  # noqa: N806
            y_val = val_data["q_mm_day"].to_numpy()

            # Add static attributes to feature arrays
            # Repeat static attrs for each timestep
            static_train = np.tile(static_attrs, (len(x_train), 1))
            static_val = np.tile(static_attrs, (len(x_val), 1))

            x_train = np.hstack([x_train, static_train])  # noqa: N806
            x_val = np.hstack([x_val, static_val])  # noqa: N806

            logger.info(
                f"Data shapes - Train: {x_train.shape}, Val: {x_val.shape}, "
                f"Features: {len(feature_cols)} dynamic + {len(static_columns)} static"
            )

            # Run optimization
            logger.info(
                f"Starting optimization for gauge {gauge_id} with dataset {dataset}"
            )
            study = run_optimization(
                X_train=x_train,
                y_train=y_train,
                X_val=x_val,
                y_val=y_val,
                study_name=f"RFR_multiobj_{gauge_id}_{dataset}",
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                verbose=False,
            )

            # Select best parameter set using weighted metrics
            if not study.best_trials:
                logger.warning(
                    f"No valid trials found for {gauge_id} with dataset {dataset}"
                )
                continue

            pareto_trials = study.best_trials
            best_trial = select_best_trial_weighted(
                pareto_trials, rfr_weights, "weighted_sum"
            )
            best_params = dict(best_trial.params)

            logger.info(
                f"Best trial {best_trial.number}: "
                f"KGE={best_trial.values[0]:.3f}, "
                f"LowFlow={best_trial.values[1]:.3f}, "
                f"HighFlow={best_trial.values[2]:.3f}, "
                f"PBIAS={-best_trial.values[3]:.2f}%"
            )

            # Train final model with full calibration data
            logger.info("Training final model with best parameters...")
            final_model = train_final_model(
                X_train=x_train, y_train=y_train, params=best_params, n_jobs=n_jobs
            )

            # Validate model
            y_pred = final_model.predict(x_val)
            y_pred = np.maximum(y_pred, 0.0)  # Ensure non-negative

            # Calculate validation metrics
            metrics = evaluate_model(y_val, y_pred)
            metrics["n_features_dynamic"] = float(len(feature_cols))
            metrics["n_features_static"] = float(len(static_columns))
            metrics["n_features_total"] = float(x_train.shape[1])
            metrics["n_train_samples"] = float(len(x_train))
            metrics["n_val_samples"] = float(len(x_val))
            metrics["latitude"] = float(latitude)

            logger.info(
                f"Validation metrics - KGE: {metrics.get('KGE', np.nan):.3f}, "
                f"NSE: {metrics.get('NSE', np.nan):.3f}, "
                f"PBIAS: {metrics.get('PBIAS', np.nan):.2f}%, "
                f"Features: {len(feature_cols)} dynamic + {len(static_columns)} static"
            )

            # Save results
            save_optimization_results(
                study=study,
                dataset_name=dataset,
                gauge_id=gauge_id,
                best_parameters=best_params,
                best_model=final_model,
                metrics=metrics,
                output_dir=result_path,
            )

            logger.info(
                f"Completed optimization for gauge {gauge_id} with dataset {dataset}"
            )

        except Exception as e:
            logger.error(
                f"Error processing gauge {gauge_id} with dataset {dataset}: {e}",
                exc_info=True,
            )
