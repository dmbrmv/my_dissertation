"""HBV inference script - Generate final time series predictions.

This script loads calibrated HBV parameters and generates predictions for:
- Calibration period (2010-2018) with warm-up
- Validation period (2019-2020)
- Full continuous simulation

Parameters are loaded from:
    data/optimization/hbv_simple/{gauge_id}/{gauge_id}_{dataset}_params.json

Output saved to:
    data/predictions/hbv/{gauge_id}/{gauge_id}_{dataset}_predictions.parquet
"""

import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append("./")

from src.models.gr4j.pet import pet_oudin
from src.models.hbv import hbv
from src.readers.geom_reader import load_geodata
from src.timeseries_stats.metrics import evaluate_model
from src.utils.logger import setup_logger

Path("logs").mkdir(exist_ok=True)
logger = setup_logger("hbv_predict", log_file="logs/hbv_predict.log", level="INFO")


def load_hbv_parameters(
    gauge_id: str, dataset: str, base_path: Path
) -> list[float] | None:
    """Load calibrated HBV parameters from JSON file.

    Args:
        gauge_id: Gauge identifier
        dataset: Dataset name (e5l, gpcp, e5, mswep)
        base_path: Base directory containing optimization results

    Returns:
        List of 16 parameters in correct order or None if not found
    """
    param_file = base_path / gauge_id / f"{gauge_id}_{dataset}_params.json"

    if not param_file.exists():
        logger.warning(f"Parameters not found: {param_file}")
        return None

    try:
        with open(param_file) as f:
            params_dict = json.load(f)

        # Extract parameters in correct order for HBV simulation
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
        params = [params_dict[name] for name in param_names]

        return params

    except Exception as e:
        logger.error(f"Failed to load parameters from {param_file}: {e}")
        return None


def predict_hbv(
    gauge_id: str,
    dataset: str,
    params_path: Path,
    output_path: Path,
    gauges_gdf: gpd.GeoDataFrame,
    prediction_period: tuple[str, str] | None = None,
    warmup_years: int = 2,
    save_format: str = "parquet",
) -> pd.DataFrame | None:
    """Generate HBV predictions for a gauge.

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
    params = load_hbv_parameters(gauge_id, dataset, params_path)
    if params is None:
        return None

    # Get gauge location for PET calculation
    point_geom = gauges_gdf.loc[gauge_id, "geometry"]
    latitude = float(point_geom.y)  # type: ignore[union-attr]

    try:
        # Load data
        with xr.open_dataset(f"data/nc_all_q/{gauge_id}.nc") as ds:
            df = ds.to_dataframe()

        # Prepare HBV input data
        hbv_data = df.loc[
            :, ["q_mm_day", f"prcp_{dataset}", "t_min_e5l", "t_max_e5l"]
        ].copy()

        # Calculate mean temperature
        hbv_data["t_mean_e5l"] = (hbv_data["t_max_e5l"] + hbv_data["t_min_e5l"]) / 2

        hbv_data.dropna(inplace=True)

        # Calculate PET using Oudin formula
        hbv_data["day_of_year"] = hbv_data.index.to_series().dt.dayofyear.values

        pet = pet_oudin(
            hbv_data["t_mean_e5l"].tolist(),
            hbv_data["day_of_year"].tolist(),
            latitude,
        )
        hbv_data["evap"] = np.asarray(pet, dtype=float)

        # Rename columns to match HBV expectations
        hbv_data.rename(
            columns={
                f"prcp_{dataset}": "prcp",
                "t_mean_e5l": "temp",
            },
            inplace=True,
        )

        # Determine simulation period
        if prediction_period is None:
            sim_start = hbv_data.index[0]
            sim_end = hbv_data.index[-1]
        else:
            sim_start = pd.to_datetime(prediction_period[0])
            sim_end = pd.to_datetime(prediction_period[1])

        # Add warm-up period
        warmup_start = sim_start - pd.DateOffset(years=warmup_years)
        if warmup_start < hbv_data.index[0]:
            warmup_start = hbv_data.index[0]

        # Extract simulation data
        sim_data = hbv_data.loc[warmup_start:sim_end, :].copy()

        # Run HBV simulation
        q_sim_full = hbv.simulation(sim_data, params)

        # Remove warm-up period from results
        n_warmup_days = len(sim_data[warmup_start:sim_start]) - 1
        q_sim = q_sim_full[n_warmup_days:]

        # Extract observed values for the prediction period
        q_obs = sim_data.loc[sim_start:sim_end, "q_mm_day"].values

        # Ensure arrays match
        min_len = min(len(q_obs), len(q_sim))
        q_obs_matched = np.asarray(q_obs[:min_len], dtype=float)
        q_sim_matched = np.asarray(q_sim[:min_len], dtype=float)

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
        metrics = evaluate_model(q_obs_matched, q_sim_matched)

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
    """Run HBV inference for all gauges and datasets."""
    # Load gauge data
    # _, gauges = load_geodata(folder_depth=".")
    gauges = gpd.read_file("res/FineTuneGauges.gpkg")
    gauges = gauges.set_index("gauge_id")

    # Configuration
    params_path = Path("data/optimization_poor_gauges/hbv_simple/")
    output_path = Path("data/predictions/hbv_poor_gauges/")
    output_path.mkdir(parents=True, exist_ok=True)

    datasets = ["e5l", "gpcp", "e5", "mswep"]

    # Find gauges with calibrated parameters
    available_gauges = []
    for gauge_dir in params_path.iterdir():
        if gauge_dir.is_dir() and gauge_dir.name in gauges.index:
            available_gauges.append(gauge_dir.name)

    logger.info("=" * 80)
    logger.info("HBV INFERENCE - GENERATING PREDICTIONS")
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
            result = predict_hbv(
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
