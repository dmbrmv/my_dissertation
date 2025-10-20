"""RFR spatial (LOBO) inference script - Generate predictions for spatial generalization.

This script is different from the other models because LOBO (Leave-One-Basin-Out)
requires retraining for each test basin. The model is trained on ALL basins except
the target basin, then predicts on the target basin.

Workflow:
1. Load best hyperparameters from calibration
2. For each test gauge:
   - Train RF on all OTHER gauges
   - Predict on test gauge
3. Save predictions and metrics

Parameters loaded from:
    data/optimization/rfr_spatial_simple/{gauge_id}_{dataset}_metrics.json
    (Note: Full model training is done here, not loaded)

Output saved to:
    data/predictions/rfr_spatial/{gauge_id}/{gauge_id}_{dataset}_predictions.parquet
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
from src.readers.geom_reader import load_geodata
from src.utils.logger import setup_logger
from src.utils.metrics import evaluate_model

Path("logs").mkdir(exist_ok=True)
logger = setup_logger(
    "rfr_spatial_predict", log_file="logs/rfr_spatial_predict.log", level="INFO"
)


def create_temporal_features(
    df: pd.DataFrame, rolling_windows: list[int], latitude: float
) -> pd.DataFrame:
    """Create temporal features for Random Forest."""
    result = df.copy()

    # Calculate PET
    result["day_of_year"] = result.index.to_series().dt.dayofyear.values

    pet = pet_oudin(
        result["t_mean"].tolist(),
        result["day_of_year"].tolist(),
        latitude,
    )
    result["pet"] = np.asarray(pet, dtype=float)

    # Cyclic temporal encoding
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


def load_gauge_data(
    gauge_id: str,
    dataset: str,
    gauges_gdf: gpd.GeoDataFrame,
    static_file: Path,
    static_columns: list[str],
    rolling_windows: list[int],
) -> tuple[pd.DataFrame, float, np.ndarray] | None:
    """Load and prepare data for a single gauge."""
    try:
        point_geom = gauges_gdf.loc[gauge_id, "geometry"]
        latitude = float(point_geom.y)  # type: ignore[union-attr]

        # Load static attributes
        static_df = pd.read_csv(static_file, index_col=0)
        if gauge_id not in static_df.index:
            return None
        static_attrs = static_df.loc[gauge_id, static_columns].to_numpy()

        # Load meteorological data
        with xr.open_dataset(f"data/nc_all_q/{gauge_id}.nc") as ds:
            df = ds.to_dataframe()

        df["t_mean"] = (df["t_max_e5l"] + df["t_min_e5l"]) / 2

        rf_data = df.loc[
            :,
            [
                "q_mm_day",
                "t_mean",
                f"prcp_{dataset}",
            ],
        ].copy()

        rf_data.rename(columns={f"prcp_{dataset}": "prcp"}, inplace=True)
        rf_data = create_temporal_features(rf_data, rolling_windows, latitude)
        rf_data.dropna(inplace=True)
        rf_data["gauge_id"] = gauge_id

        return rf_data, latitude, static_attrs

    except Exception:
        return None


def predict_rfr_spatial(
    test_gauge_id: str,
    all_gauge_ids: list[str],
    dataset: str,
    gauges_gdf: gpd.GeoDataFrame,
    static_file: Path,
    static_columns: list[str],
    rolling_windows: list[int],
    output_path: Path,
    training_period: tuple[str, str] = ("2008-01-01", "2018-12-31"),
    prediction_period: tuple[str, str] = ("2019-01-01", "2020-12-31"),
    save_format: str = "parquet",
) -> pd.DataFrame | None:
    """Generate LOBO predictions for a test gauge.

    Note: Uses fixed RF hyperparameters optimized during calibration.
    For production use, load best params from optimization results.
    """
    logger.info(f"Processing LOBO fold: test gauge = {test_gauge_id}")

    # Load data for all gauges
    train_data_list = []
    test_data_obj = None
    test_static = None

    for gauge_id in all_gauge_ids:
        result = load_gauge_data(
            gauge_id,
            dataset,
            gauges_gdf,
            static_file,
            static_columns,
            rolling_windows,
        )

        if result is None:
            continue

        gauge_df, _, static_attrs = result

        if gauge_id == test_gauge_id:
            test_data_obj = gauge_df
            test_static = static_attrs
        else:
            train_data_list.append(gauge_df)

    if not train_data_list or test_data_obj is None:
        logger.error(f"Insufficient data for {test_gauge_id}")
        return None

    # Combine training data from multiple gauges
    train_df_combined = pd.concat(train_data_list, axis=0)

    # Split training data (use data from OTHER gauges only)
    train_split = train_df_combined.loc[training_period[0] : training_period[1], :]

    # Test period for held-out gauge
    test_split = test_data_obj.loc[prediction_period[0] : prediction_period[1], :]

    if len(train_split) < 100 or len(test_split) < 30:
        logger.warning(f"Insufficient data for {test_gauge_id}")
        return None

    # Prepare features
    feature_cols = [
        col
        for col in train_split.columns
        if col not in ["q_mm_day", "day_of_year", "gauge_id"]
    ]

    x_train = train_split[feature_cols].values
    y_train = train_split["q_mm_day"].values

    # Add static attributes to training data
    # (each training row gets the static attrs of its corresponding gauge)
    train_static_list = []
    for gauge_id in train_split["gauge_id"].unique():
        gauge_mask = train_split["gauge_id"] == gauge_id
        n_rows = gauge_mask.sum()

        # Load static for this gauge
        static_df = pd.read_csv(static_file, index_col=0)
        gauge_static = static_df.loc[gauge_id, static_columns].to_numpy()
        gauge_static = np.asarray(gauge_static, dtype=float)

        train_static_list.append(np.tile(gauge_static, (n_rows, 1)))

    train_static_combined = np.vstack(train_static_list)
    x_train = np.hstack([x_train, train_static_combined])

    # Prepare test features
    test_feature_cols = [
        col for col in test_split.columns if col not in ["q_mm_day", "day_of_year"]
    ]
    x_test = test_split[test_feature_cols].values

    # Add static attributes to test
    if test_static is not None:
        n_test = x_test.shape[0]
        static_test = np.tile(test_static, (n_test, 1))
        x_test = np.hstack([x_test, static_test])

    y_test = test_split["q_mm_day"].values

    # Use fixed hyperparameters (optimized during LOBO calibration)
    # For best results, load from optimization/rfr_spatial_simple/
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    rf_model.fit(x_train, y_train)

    # Predict on test gauge
    y_pred = rf_model.predict(x_test)

    # Create results DataFrame
    results_df = pd.DataFrame(
        {
            "date": test_split.index,
            "q_obs": y_test,
            "q_sim": y_pred,
            "dataset": dataset,
            "gauge_id": test_gauge_id,
        }
    )

    # Calculate metrics
    metrics = evaluate_model(
        np.asarray(y_test, dtype=float), np.asarray(y_pred, dtype=float)
    )

    # Save predictions
    output_dir = output_path / test_gauge_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_format == "parquet":
        output_file = output_dir / f"{test_gauge_id}_{dataset}_predictions.parquet"
        results_df.to_parquet(output_file, index=False)
    elif save_format == "csv":
        output_file = output_dir / f"{test_gauge_id}_{dataset}_predictions.csv"
        results_df.to_csv(output_file, index=False)
    elif save_format == "netcdf":
        output_file = output_dir / f"{test_gauge_id}_{dataset}_predictions.nc"
        results_df.set_index("date").to_xarray().to_netcdf(output_file)
    else:
        raise ValueError(f"Unknown save format: {save_format}")

    # Save metrics summary
    metrics_file = output_dir / f"{test_gauge_id}_{dataset}_prediction_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        f"✓ {test_gauge_id}/{dataset}: "
        f"Predicted {len(results_df)} days, "
        f"KGE={metrics.get('KGE', np.nan):.3f}, "
        f"NSE={metrics.get('NSE', np.nan):.3f}"
    )

    return results_df


def main() -> None:
    """Run RFR spatial (LOBO) inference for all gauges."""
    # Load gauge data
    _, gauges = load_geodata(folder_depth=".")

    full_gauges = [
        i.stem for i in Path("data/nc_all_q").glob("*.nc") if i.stem in gauges.index
    ]

    # Configuration
    output_path = Path("data/predictions/rfr_spatial/")
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = "e5l"  # ERA5-Land (most reliable)

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

    logger.info("=" * 80)
    logger.info("RFR SPATIAL (LOBO) INFERENCE - GENERATING PREDICTIONS")
    logger.info("=" * 80)
    logger.info(f"Total gauges: {len(full_gauges)}")
    logger.info(f"Dataset: {dataset}")
    logger.info("Strategy: Leave-One-Basin-Out (retrain for each test basin)")
    logger.info(f"Rolling windows: {rolling_windows}")
    logger.info(f"Static features: {len(static_parameters)}")
    logger.info(f"Output directory: {output_path}")
    logger.info("=" * 80)

    # Generate predictions for each gauge
    n_success = 0
    n_total = len(full_gauges)

    for test_gauge_id in full_gauges:
        result = predict_rfr_spatial(
            test_gauge_id=test_gauge_id,
            all_gauge_ids=full_gauges,
            dataset=dataset,
            gauges_gdf=gauges,
            static_file=static_file,
            static_columns=static_parameters,
            rolling_windows=rolling_windows,
            output_path=output_path,
            training_period=("2008-01-01", "2018-12-31"),
            prediction_period=("2019-01-01", "2020-12-31"),
            save_format="parquet",
        )

        if result is not None:
            n_success += 1

    logger.info("=" * 80)
    logger.info(f"✅ INFERENCE COMPLETED: {n_success}/{n_total} successful")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
