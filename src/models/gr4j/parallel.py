"""Parallelization utilities for GR4J Optuna optimization."""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr

from src.models.gr4j import model as gr4j
from src.models.gr4j.gr4j_optuna import run_optimization
from src.models.gr4j.pareto import save_optimization_results, select_best_trial_weighted
from src.models.gr4j.pet import pet_oudin
from src.utils.logger import setup_logger
from src.utils.metrics import evaluate_model

logger = setup_logger("main_gr4j_optuna", log_file="logs/gr4j_optuna.log")


def run_parallel_optimization(
    gauge_ids: list[str], process_gauge_func, n_processes: int | None = None, **kwargs
) -> None:
    """Run optimization in parallel for multiple gauges.

    Args:
        gauge_ids: List of gauge identifiers.
        process_gauge_func: Function to process a single gauge.
        n_processes: Number of processes to use.
        kwargs: Additional arguments for process_gauge_func.
    """
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    n_processes = max(1, min(n_processes, mp.cpu_count()))
    logger.info(f"Starting parallel optimization with {n_processes} processes")
    process_func = partial(process_gauge_func, **kwargs)
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        list(executor.map(process_func, gauge_ids))
    logger.info("Parallel optimization completed")


def process_gr4j_gauge(
    gauge_id: str,
    datasets: list[str],
    calibration_period: tuple[str, str],
    validation_period: tuple[str, str],
    save_storage: Path,
    e_obs_gauge: gpd.GeoDataFrame | None = None,
    n_trials: int = 15,
    timeout: int = 3600,
    overwrite_results: bool = False,
) -> None:
    """Process a single gauge across multiple datasets with the GR4J model.

    Loads hydro and meteo data, calculates PET, runs model optimization,
    and evaluates performance on a validation period.

    Args:
        gauge_id: Gauge identifier
        datasets: List of meteorological dataset names
        calibration_period: Start and end dates for calibration (YYYY-MM-DD)
        validation_period: Start and end dates for validation (YYYY-MM-DD)
        save_storage: Root directory to save results
        e_obs_gauge: GeoDataFrame with gauge information
        n_trials: Number of optimization trials
        timeout: Optimization timeout in seconds
        overwrite_results: Whether to overwrite existing results
    """
    result_path = save_storage / gauge_id
    result_path.mkdir(parents=True, exist_ok=True)
    # Get latitude from GeoDataFrame - ensure we properly access the Point geometry
    # This gets the y coordinate (latitude) from the Point geometry
    point_geom = e_obs_gauge.loc[gauge_id, "geometry"]
    latitude = float(point_geom.y)
    # Common weights for all datasets
    hydro_weights: dict[str, float] = {
        "KGE": 0.5,
        "NSE": 0.5,
        "logNSE": 0.5,
        "PBIAS": 0.03,
        "RMSE": 0.02,
    }

    for dataset in datasets:
        logger.info(f"Processing gauge {gauge_id} with dataset {dataset}")

        # Check if results already exist and if overwriting is disabled
        if (result_path / f"{gauge_id}_{dataset}").exists() and not overwrite_results:
            logger.info(
                f"Results for gauge {gauge_id} with dataset {dataset} already exist. Skipping."
            )
            continue

        try:
            # Load meteorological data with xarray (efficient for NetCDF)
            with xr.open_dataset(f"../data/nc_all_q/{gauge_id}.nc") as ds:
                exmp = ds.to_dataframe()
                exmp["t_mean_e5l"] = (exmp["t_max_e5l"] + exmp["t_min_e5l"]) / 2
            gr4j_data = exmp.loc[
                "2008":"2020",
                [
                    "q_mm_day",
                    "t_min_e5l",
                    "t_mean_e5l",
                    "t_max_e5l",
                    f"prcp_{dataset}",
                ],
            ]
            # Ensure day_of_year is available (derive from the index if missing) and clean NaNs
            if "day_of_year" not in gr4j_data.columns:
                gr4j_data["day_of_year"] = gr4j_data.index.dayofyear
            else:
                missing_doy = gr4j_data["day_of_year"].isna()
                if missing_doy.any():
                    gr4j_data.loc[missing_doy, "day_of_year"] = gr4j_data.index[
                        missing_doy
                    ].dayofyear

            # Calculate PET using Oudin formula - keep native numeric types
            t_mean_list = gr4j_data["t_mean"].tolist()
            day_of_year_list = gr4j_data["day_of_year"].tolist()
            gr4j_data["pet_mm_day"] = pet_oudin(t_mean_list, day_of_year_list, latitude)

            # Run optimization
            logger.info(
                f"Starting optimization for gauge {gauge_id} with dataset {dataset}"
            )
            study = run_optimization(
                gr4j_data,
                calibration_period,
                study_name=f"GR4J_multiobj_{gauge_id}_{dataset}",
                n_trials=n_trials,
                timeout=timeout,
                verbose=False,
            )

            # Select best parameter set using weighted metrics
            if not study.best_trials:
                logger.warning(
                    f"No valid trials found for {gauge_id} with dataset {dataset}"
                )
                continue

            pareto_trials = study.best_trials
            best_hydro = select_best_trial_weighted(
                pareto_trials, hydro_weights, "weighted_sum"
            )
            best_params = dict(best_hydro.params)

            # Validate model with best parameters
            gr4j_validation = gr4j_data.loc[
                validation_period[0] : validation_period[1], :
            ]
            q_sim = gr4j.simulation(gr4j_validation, list(best_hydro.params.values()))

            # Convert to numpy arrays for evaluate_model
            observed_values = np.array(gr4j_validation["q_mm_day"].values, dtype=float)
            q_sim_np = np.array(q_sim, dtype=float)
            metrics = evaluate_model(observed_values, q_sim_np)

            # Save results
            save_optimization_results(
                study=study,
                dataset_name=dataset,
                gauge_id=gauge_id,
                best_parameters=best_params,
                metrics=metrics,
                output_dir=str(result_path),
            )
            logger.info(
                f"Completed optimization for gauge {gauge_id} with dataset {dataset}"
            )

        except Exception as e:
            logger.error(
                f"Error processing gauge {gauge_id} with dataset {dataset}: {str(e)}"
            )
