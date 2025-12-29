"""GR4J inference script - Generate final time series predictions.

This script loads calibrated GR4J parameters and generates predictions for:
- Calibration period (2010-2018) with warm-up
- Validation period (2019-2020)
- Full continuous simulation

Parameters are loaded from:
    data/optimization/gr4j_simple/{gauge_id}/{gauge_id}_{dataset}_parameters.json

Output saved to:
    data/predictions/gr4j/{gauge_id}/{gauge_id}_{dataset}_predictions.parquet
"""

import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append("./")

from src.models.gr4j import model as gr4j
from src.models.gr4j.pet import pet_oudin
from src.readers.geom_reader import load_geodata
from src.timeseries_stats.metrics import evaluate_model
from src.utils.logger import setup_logger

Path("logs").mkdir(exist_ok=True)
logger = setup_logger("gr4j_predict", log_file="logs/gr4j_predict.log", level="INFO")


def load_gr4j_parameters(
    gauge_id: str, dataset: str, base_path: Path
) -> list[float] | None:
    """Load calibrated GR4J parameters from JSON file.

    Args:
        gauge_id: Gauge identifier
        dataset: Dataset name (e5l, gpcp, e5, mswep)
        base_path: Base directory containing optimization results

    Returns:
        List of 7 parameters [x1, x2, x3, x4, ctg, kf, tt] or None if not found
    """
    param_file = base_path / gauge_id / f"{gauge_id}_{dataset}_parameters.json"

    if not param_file.exists():
        logger.warning(f"Parameters not found: {param_file}")
        return None

    try:
        with open(param_file) as f:
            params_dict = json.load(f)

        # Extract parameters in correct order for GR4J simulation
        param_names = ["x1", "x2", "x3", "x4", "ctg", "kf", "tt"]
        params = [params_dict[name] for name in param_names]

        return params

    except Exception as e:
        logger.error(f"Failed to load parameters from {param_file}: {e}")
        return None


def predict_gr4j(
    gauge_id: str,
    dataset: str,
    params_path: Path,
    output_path: Path,
    gauges_gdf: gpd.GeoDataFrame,
    prediction_period: tuple[str, str] | None = None,
    warmup_years: int = 2,
    save_format: str = "parquet",
) -> pd.DataFrame | None:
    """Generate GR4J predictions for a gauge.

    Args:
        gauge_id: Gauge identifier
        dataset: Dataset name
        params_path: Path to calibrated parameters
        output_path: Path to save predictions
        gauges_gdf: GeoDataFrame with gauge geometries
        prediction_period: Optional (start, end) dates; if None, uses full data
        warmup_years: Years for warm-up period
        save_format: Output format ('parquet', 'csv', or 'netcdf')

    Returns:
        DataFrame with columns [q_obs, q_sim, date] or None if failed
    """
    # Load parameters
    params = load_gr4j_parameters(gauge_id, dataset, params_path)
    if params is None:
        return None

    # Get gauge location for PET calculation
    point_geom = gauges_gdf.loc[gauge_id, "geometry"]
    latitude = float(point_geom.y)  # type: ignore[union-attr]

    try:
        # Load data
        with xr.open_dataset(f"data/nc_all_q/{gauge_id}.nc") as ds:
            df = ds.to_dataframe()

        # Prepare GR4J input data
        gr4j_data = df.loc[
            :, ["q_mm_day", f"prcp_{dataset}", "t_min_e5l", "t_max_e5l"]
        ].copy()
        gr4j_data.dropna(inplace=True)
        gr4j_data["t_mean"] = (gr4j_data["t_max_e5l"] + gr4j_data["t_min_e5l"]) / 2
        # Calculate PET using Oudin formula
        gr4j_data["day_of_year"] = gr4j_data.index.to_series().dt.dayofyear.values

        pet = pet_oudin(
            gr4j_data["t_mean"].tolist(),
            gr4j_data["day_of_year"].tolist(),
            latitude,
        )
        gr4j_data["pet_mm_day"] = np.asarray(pet, dtype=float)

        # Rename columns to match GR4J expectations
        gr4j_data.rename(
            columns={f"prcp_{dataset}": "prcp"},
            inplace=True,
        )

        # Determine simulation period
        if prediction_period is None:
            sim_start = gr4j_data.index[0]
            sim_end = gr4j_data.index[-1]
        else:
            sim_start = pd.to_datetime(prediction_period[0])
            sim_end = pd.to_datetime(prediction_period[1])

        # Add warm-up period
        warmup_start = sim_start - pd.DateOffset(years=warmup_years)
        if warmup_start < gr4j_data.index[0]:
            warmup_start = gr4j_data.index[0]

        # Extract simulation data
        sim_data = gr4j_data.loc[warmup_start:sim_end, :].copy()

        # Run GR4J simulation
        q_sim_full = gr4j.simulation(sim_data, params)

        # Remove warm-up period from results
        n_warmup_days = len(sim_data[warmup_start:sim_start]) - 1
        q_sim = q_sim_full[n_warmup_days:]

        # Extract observed values for the prediction period
        q_obs = sim_data.loc[sim_start:sim_end, "q_mm_day"].values

        # Ensure arrays match
        min_len = min(len(q_obs), len(q_sim))
        q_obs_matched = q_obs[:min_len]
        q_sim_matched = q_sim[:min_len]

        # Create results DataFrame
        result_dates = sim_data.loc[sim_start:sim_end, :].index[:min_len]
        results_df = pd.DataFrame(
            {
                "date": result_dates,
                "q_obs": q_obs_matched,
                "q_sim": q_sim_matched,
                "dataset": dataset,
                "gauge_id": gauge_id,
            }
        )

        # Calculate metrics
        metrics = evaluate_model(
            np.asarray(q_obs_matched, dtype=float),
            np.asarray(q_sim_matched, dtype=float),
        )

        # Save predictions
        output_dir = output_path / gauge_id
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_format == "parquet":
            output_file = output_dir / f"{gauge_id}_{dataset}_predictions.parquet"
            results_df.to_parquet(output_file, index=False)
        elif save_format == "csv":
            output_file = output_dir / f"{gauge_id}_{dataset}_predictions.csv"
            results_df.to_csv(output_file, index=False)
        elif save_format == "netcdf":
            output_file = output_dir / f"{gauge_id}_{dataset}_predictions.nc"
            results_df.set_index("date").to_xarray().to_netcdf(output_file)
        else:
            raise ValueError(f"Unknown save format: {save_format}")

        # Save metrics summary
        metrics_file = output_dir / f"{gauge_id}_{dataset}_prediction_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(
            f"✓ {gauge_id}/{dataset}: "
            f"Predicted {len(results_df)} days, "
            f"KGE={metrics.get('KGE', np.nan):.3f}, "
            f"NSE={metrics.get('NSE', np.nan):.3f}"
        )

        return results_df

    except Exception as e:
        logger.error(f"✗ {gauge_id}/{dataset}: {e}")
        return None


def main() -> None:
    """Run GR4J inference for all gauges and datasets."""
    # Load gauge data
    _, gauges = load_geodata(folder_depth=".")

    # Configuration
    params_path = Path("data/optimization/gr4j_simple/")
    output_path = Path("data/predictions/gr4j/")
    output_path.mkdir(parents=True, exist_ok=True)

    datasets = ["e5l", "gpcp", "e5", "mswep"]

    # Find gauges with calibrated parameters
    available_gauges = []
    for gauge_dir in params_path.iterdir():
        if gauge_dir.is_dir() and gauge_dir.name in gauges.index:
            available_gauges.append(gauge_dir.name)

    logger.info("=" * 80)
    logger.info("GR4J INFERENCE - GENERATING PREDICTIONS")
    logger.info("=" * 80)
    logger.info(f"Total gauges with parameters: {len(available_gauges)}")
    logger.info(f"Datasets: {', '.join(datasets)}")
    logger.info(f"Output directory: {output_path}")
    logger.info("=" * 80)

    # Generate predictions for each gauge/dataset
    n_success = 0
    n_total = 0

    for gauge_id in available_gauges:
        for dataset in datasets:
            n_total += 1

            # Generate predictions for full available period
            result = predict_gr4j(
                gauge_id=gauge_id,
                dataset=dataset,
                params_path=params_path,
                output_path=output_path,
                gauges_gdf=gauges,
                prediction_period=None,  # Full period
                warmup_years=2,
                save_format="csv",
            )

            if result is not None:
                n_success += 1

    logger.info("=" * 80)
    logger.info(f"✅ INFERENCE COMPLETED: {n_success}/{n_total} successful")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
