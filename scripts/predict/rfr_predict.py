"""Random Forest Regressor inference script - Generate final time series predictions.

This script loads calibrated RF parameters and generates predictions for:
- Training period (2008-2018)
- Validation period (2019-2020)
- Full continuous simulation

Parameters are loaded from:
    data/optimization/rfr_simple/{gauge_id}/{gauge_id}_{dataset}_params.json

Output saved to:
    data/predictions/rfr/{gauge_id}/{gauge_id}_{dataset}_predictions.parquet
"""

import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]
import xarray as xr

sys.path.append("./")

from src.models.gr4j.pet import pet_oudin
from src.timeseries_stats.metrics import evaluate_model
from src.utils.logger import setup_logger

Path("logs").mkdir(exist_ok=True)
logger = setup_logger("rfr_predict", log_file="logs/rfr_predict.log", level="INFO")


def load_rf_parameters(
    gauge_id: str, dataset: str, base_path: Path
) -> dict[str, float | str] | None:
    """Load calibrated Random Forest hyperparameters from JSON file.

    Args:
        gauge_id: Gauge identifier
        dataset: Dataset name (e5l, gpcp, e5, mswep)
        base_path: Base directory containing optimization results

    Returns:
        Dictionary of hyperparameters or None if not found
    """
    param_file = base_path / gauge_id / f"{gauge_id}_{dataset}_params.json"

    if not param_file.exists():
        logger.warning(f"Parameters not found: {param_file}")
        return None

    try:
        with open(param_file) as f:
            params = json.load(f)
        return params

    except Exception as e:
        logger.error(f"Failed to load parameters from {param_file}: {e}")
        return None


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
    result["day_of_year"] = result.index.to_series().dt.dayofyear.values

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


def predict_rfr(
    gauge_id: str,
    dataset: str,
    params_path: Path,
    output_path: Path,
    gauges_gdf: gpd.GeoDataFrame,
    static_file: Path,
    static_columns: list[str],
    rolling_windows: list[int],
    training_period: tuple[str, str] = ("2008-01-01", "2018-12-31"),
    prediction_period: tuple[str, str] | None = None,
    save_format: str = "parquet",
) -> pd.DataFrame | None:
    """Generate Random Forest predictions for a gauge.

    Args:
        gauge_id: Gauge identifier
        dataset: Dataset name
        params_path: Path to calibrated parameters
        output_path: Path to save predictions
        gauges_gdf: GeoDataFrame with gauge geometries
        static_file: Path to static attributes CSV
        static_columns: List of static attribute columns
        rolling_windows: List of rolling window sizes
        training_period: Period to train final model
        prediction_period: Optional (start, end) dates; if None, predicts full data
        save_format: Output format ('parquet', 'csv', or 'netcdf')

    Returns:
        DataFrame with columns [q_obs, q_sim, date] or None if failed
    """
    # Load parameters
    rf_params = load_rf_parameters(gauge_id, dataset, params_path)
    if rf_params is None:
        return None

    # Get gauge location
    point_geom = gauges_gdf.loc[gauge_id, "geometry"]
    latitude = float(point_geom.y)  # type: ignore[union-attr]

    # Load static attributes
    try:
        static_df = pd.read_csv(
            static_file, index_col="gauge_id", dtype={"gauge_id": str}
        )
        if gauge_id not in static_df.index:
            logger.error(f"Gauge {gauge_id} not found in static attributes")
            return None
        static_attrs = static_df.loc[gauge_id, static_columns].values
        static_attrs = np.asarray(static_attrs, dtype=float)
    except Exception as e:
        logger.error(f"Failed to load static attributes: {e}")
        return None

    try:
        # Load data
        with xr.open_dataset(f"data/nc_all_q/{gauge_id}.nc") as ds:
            df = ds.to_dataframe()

        df["t_mean"] = (df["t_max_e5l"] + df["t_min_e5l"]) / 2

        # Select relevant columns
        rf_data = df.loc[
            :,
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

        # Prepare training data
        train_data = rf_data.loc[training_period[0] : training_period[1], :].copy()

        if len(train_data) < 100:
            logger.warning(f"Insufficient training data for {gauge_id}/{dataset}")
            return None

        # Prepare features
        feature_cols = [
            col for col in rf_data.columns if col not in ["q_mm_day", "day_of_year"]
        ]

        X_train = train_data[feature_cols].values
        y_train = train_data["q_mm_day"].values

        # Add static features to training data
        n_train = X_train.shape[0]
        static_train = np.tile(static_attrs, (n_train, 1))
        X_train = np.hstack([X_train, static_train])

        # Train final model with best parameters
        rf_model = RandomForestRegressor(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            min_samples_split=rf_params["min_samples_split"],
            min_samples_leaf=rf_params["min_samples_leaf"],
            max_features=rf_params["max_features"],
            random_state=42,
            n_jobs=-1,
        )

        rf_model.fit(X_train, y_train)

        # Determine prediction period
        if prediction_period is None:
            pred_start = rf_data.index[0]
            pred_end = rf_data.index[-1]
        else:
            pred_start = pd.to_datetime(prediction_period[0])
            pred_end = pd.to_datetime(prediction_period[1])

        # Prepare prediction data
        pred_data = rf_data.loc[pred_start:pred_end, :].copy()

        if len(pred_data) < 1:
            logger.warning(f"No prediction data for {gauge_id}/{dataset}")
            return None

        X_pred = pred_data[feature_cols].values

        # Add static features to prediction data
        n_pred = X_pred.shape[0]
        static_pred = np.tile(static_attrs, (n_pred, 1))
        X_pred = np.hstack([X_pred, static_pred])

        # Generate predictions
        q_sim = rf_model.predict(X_pred)
        q_obs = pred_data["q_mm_day"].values

        # Create results DataFrame
        results_df = pd.DataFrame(
            {
                "date": pred_data.index,
                "q_obs": q_obs,
                "q_sim": q_sim,
                "dataset": dataset,
                "gauge_id": gauge_id,
            }
        )

        # Calculate metrics
        metrics = evaluate_model(
            np.asarray(q_obs, dtype=float), np.asarray(q_sim, dtype=float)
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
    """Run Random Forest inference for all gauges and datasets."""
    # Load gauge data
    # _, gauges = load_geodata(folder_depth=".")
    gauges = gpd.read_file("res/FineTuneGauges.gpkg")
    gauges = gauges.set_index("gauge_id")

    # Configuration
    params_path = Path("data/optimization_poor_gauges/rfr_simple/")
    output_path = Path("data/predictions/rfr_poor_gauges/")
    output_path.mkdir(parents=True, exist_ok=True)

    datasets = ["e5l", "gpcp", "e5", "mswep"]

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

    # Find gauges with calibrated parameters
    available_gauges = []
    for gauge_dir in params_path.iterdir():
        if gauge_dir.is_dir() and gauge_dir.name in gauges.index:
            available_gauges.append(gauge_dir.name)

    logger.info("=" * 80)
    logger.info("RANDOM FOREST INFERENCE - GENERATING PREDICTIONS")
    logger.info("=" * 80)
    logger.info(f"Total gauges with parameters: {len(available_gauges)}")
    logger.info(f"Datasets: {', '.join(datasets)}")
    logger.info(f"Rolling windows: {rolling_windows}")
    logger.info(f"Static features: {len(static_parameters)}")
    logger.info(f"Output directory: {output_path}")
    logger.info("=" * 80)

    # Generate predictions for each gauge/dataset
    n_success = 0
    n_total = 0

    for gauge_id in available_gauges:
        for dataset in datasets:
            n_total += 1

            # Generate predictions for full available period
            result = predict_rfr(
                gauge_id=gauge_id,
                dataset=dataset,
                params_path=params_path,
                output_path=output_path,
                gauges_gdf=gauges,
                static_file=static_file,
                static_columns=static_parameters,
                rolling_windows=rolling_windows,
                training_period=("2008-01-01", "2016-12-31"),
                prediction_period=("2017-01-01", "2018-12-31"),  # Full period
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
