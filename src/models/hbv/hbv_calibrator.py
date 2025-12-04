"""HBV calibration helpers aligning with the GR4J Optuna workflow."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from src.models.gr4j.pet import pet_oudin
from src.models.hbv import hbv
from src.models.hbv.hbv_optuna import run_optimization
from src.models.hbv.pareto import (
    save_optimization_results,
    select_best_trial_weighted,
)
from src.utils.logger import setup_logger
from timeseries_stats.metrics import evaluate_model
from timeseries_stats.metrics_enhanced import analyze_flow_regimes

logger = setup_logger("main_hbv_optuna", log_file="logs/hbv_optuna.log")


def load_hbv_input(
    gauge_id: str,
    dataset: str,
    latitude: float,
    warmup_years: int,
    calibration_period: tuple[str, str],
) -> pd.DataFrame:
    """Load and prepare HBV inputs with warm-up extension."""
    with xr.open_dataset(f"data/nc_all_q/{gauge_id}.nc") as ds:
        df = ds.to_dataframe()

    df["t_mean_e5l"] = (df["t_max_e5l"] + df["t_min_e5l"]) / 2

    calib_start_year = pd.to_datetime(calibration_period[0]).year
    data_start_year = calib_start_year - warmup_years

    hbv_df = df.loc[
        f"{data_start_year}" : "2020",
        [
            "q_mm_day",
            "t_mean_e5l",
            f"prcp_{dataset}",
        ],
    ].copy()

    if "day_of_year" not in hbv_df.columns:
        hbv_df["day_of_year"] = hbv_df.index.dayofyear  # type: ignore[attr-defined]
    else:
        mask = hbv_df["day_of_year"].isna()
        if mask.any():
            hbv_df.loc[mask, "day_of_year"] = hbv_df.index[mask].dayofyear  # type: ignore[attr-defined]

    evap = pet_oudin(
        hbv_df["t_mean_e5l"].tolist(),
        hbv_df["day_of_year"].tolist(),
        latitude,
    )
    hbv_df["evap"] = np.asarray(evap, dtype=float)
    hbv_df.rename(
        columns={
            "t_mean_e5l": "temp",
            f"prcp_{dataset}": "prcp",
        },
        inplace=True,
    )

    return hbv_df


def simulate_with_warmup(
    data: pd.DataFrame,
    params: Iterable[float],
    simulation_period: tuple[str, str],
    warmup_years: int,
) -> np.ndarray:
    """Run HBV simulation including warm-up days."""
    sim_start = pd.to_datetime(simulation_period[0])
    sim_end = pd.to_datetime(simulation_period[1])
    warmup_start = sim_start - pd.DateOffset(years=warmup_years)

    if warmup_start < data.index[0]:
        logger.warning(
            "Validation warm-up truncated for period %s-%s. "
            "Using earliest available date %s.",
            simulation_period[0],
            simulation_period[1],
            data.index[0],
        )
        warmup_start = data.index[0]

    sim_df = data.loc[warmup_start:sim_end].copy()
    q_sim_full = hbv.simulation(sim_df, list(params))

    warmup_len = len(data.loc[warmup_start:sim_start]) - 1
    q_sim = np.asarray(q_sim_full[warmup_len:], dtype=float)
    return q_sim


def process_hbv_gauge(
    gauge_id: str,
    datasets: list[str],
    calibration_period: tuple[str, str],
    validation_period: tuple[str, str],
    save_storage: Path,
    e_obs_gauge: gpd.GeoDataFrame,
    n_trials: int = 100,
    timeout: int = 3600,
    warmup_years: int = 2,
    overwrite_results: bool = False,
) -> None:
    """Run HBV Optuna calibration for a single gauge across datasets."""
    result_path = save_storage / gauge_id
    result_path.mkdir(parents=True, exist_ok=True)

    point_geom = e_obs_gauge.loc[gauge_id, "geometry"]
    latitude = float(point_geom.y)  # type: ignore[union-attr]

    hydro_weights: dict[str, float] = {
        "KGE": 0.25,
        "low_flow": 0.35,
        "high_flow": 0.30,
        "PBIAS": 0.10,
    }

    for dataset in datasets:
        dataset_path = result_path / f"{gauge_id}_{dataset}"
        if dataset_path.exists() and not overwrite_results:
            continue

        try:
            hbv_df = load_hbv_input(
                gauge_id=gauge_id,
                dataset=dataset,
                latitude=latitude,
                warmup_years=warmup_years,
                calibration_period=calibration_period,
            )

            study = run_optimization(
                hbv_df,
                calibration_period=calibration_period,
                study_name=f"HBV_improved_{gauge_id}_{dataset}",
                n_trials=n_trials,
                timeout=timeout,
                verbose=False,
                warmup_years=warmup_years,
                use_detailed=False,
            )

            if not study.best_trials:
                logger.warning(
                    "No valid trials found for %s with dataset %s", gauge_id, dataset
                )
                continue

            best_trial = select_best_trial_weighted(
                study.best_trials, hydro_weights, method="weighted_sum"
            )
            best_params = dict(best_trial.params)

            q_sim = simulate_with_warmup(
                data=hbv_df,
                params=best_params.values(),
                simulation_period=validation_period,
                warmup_years=warmup_years,
            )

            validation_df = hbv_df.loc[validation_period[0] : validation_period[1], :]
            q_obs = np.asarray(validation_df["q_mm_day"].values, dtype=float)

            min_len = min(len(q_obs), len(q_sim))
            q_obs = q_obs[:min_len]
            q_sim = q_sim[:min_len]

            metrics = evaluate_model(q_obs, q_sim)
            metrics.update(analyze_flow_regimes(q_obs, q_sim))

            save_optimization_results(
                study=study,
                dataset_name=dataset,
                gauge_id=gauge_id,
                best_parameters=best_params,
                metrics=metrics,
                output_dir=str(result_path),
            )
            logger.info(
                "✓ %s/%s HBV calibration KGE=%.3f",
                gauge_id,
                dataset,
                metrics.get("KGE", np.nan),
            )

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("✗ %s/%s HBV calibration failed: %s", gauge_id, dataset, exc)


__all__ = [
    "process_hbv_gauge",
    "load_hbv_input",
    "simulate_with_warmup",
]
