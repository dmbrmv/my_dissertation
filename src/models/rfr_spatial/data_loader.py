"""Data loading utilities for regional RFR spatial models.

Handles loading gauge data, static attributes, and meteorological data
for LOBO cross-validation.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from src.models.rfr_spatial.features import create_universal_features
from src.utils.logger import setup_logger

logger = setup_logger("rfr_spatial_data", log_file="logs/rfr_spatial.log")


def load_gauge_data(
    gauge_id: str,
    dataset: str,
    gauges_gdf: gpd.GeoDataFrame,
    static_file: Path,
    static_columns: list[str],
    rolling_windows: list[int],
) -> tuple[pd.DataFrame, float, np.ndarray] | None:
    """Load and preprocess data for a single gauge.

    Args:
        gauge_id: Gauge identifier
        dataset: Meteorological dataset name
        gauges_gdf: GeoDataFrame with gauge geometries
        static_file: Path to static attributes CSV
        static_columns: List of static attribute column names
        rolling_windows: Rolling window sizes

    Returns:
        Tuple of (feature DataFrame, latitude, static attributes) or None if error
    """
    try:
        # Check gauge exists
        if gauge_id not in gauges_gdf.index:
            logger.warning(f"Gauge {gauge_id} not in GeoDataFrame, skipping")
            return None

        # Get latitude
        point_geom = gauges_gdf.loc[gauge_id, "geometry"]
        latitude = float(point_geom.y)  # type: ignore[union-attr]

        # Load static attributes
        static_df = pd.read_csv(static_file, dtype={"gauge_id": str})
        static_df.set_index("gauge_id", inplace=True)

        if gauge_id not in static_df.index:
            logger.warning(f"Gauge {gauge_id} not in static attributes, skipping")
            return None

        static_row = static_df.loc[gauge_id][static_columns]
        static_attrs = static_row.values.astype(float)

        # Load meteorological data
        data_path = Path(f"data/nc_all_q/{gauge_id}.nc")
        if not data_path.exists():
            logger.warning(f"Data file not found for gauge {gauge_id}, skipping")
            return None

        with xr.open_dataset(data_path) as ds:
            df = ds.to_dataframe()

        # Check required columns
        required_cols = ["q_mm_day", "t_min_e5l", "t_max_e5l", f"prcp_{dataset}"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(
                f"Missing columns {missing_cols} for {gauge_id} with "
                f"dataset {dataset}, skipping"
            )
            return None

        # Select relevant data
        gauge_data = df.loc["2008":"2020", required_cols].copy()

        if len(gauge_data) == 0:
            logger.warning(f"No data for {gauge_id} in 2008-2020 period, skipping")
            return None

        # Create features
        features_df, _ = create_universal_features(
            gauge_data,
            gauge_id,
            dataset,
            latitude,
            static_attrs,
            rolling_windows,
        )

        return features_df, latitude, static_attrs

    except Exception as e:
        logger.error(f"Error loading data for gauge {gauge_id}: {e}", exc_info=True)
        return None


def collect_inner_split_data(
    train_gauge_ids: list[str],
    gauge_data_dict: dict,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Collect training/validation data for inner temporal split.

    Args:
        train_gauge_ids: List of training gauge IDs
        gauge_data_dict: Dictionary mapping gauge_id -> data dict

    Returns:
        Tuple of (x_train_list, y_train_list, x_val_list, y_val_list)
    """
    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []

    for train_gauge_id in train_gauge_ids:
        features_df = gauge_data_dict[train_gauge_id]["features"]

        # Inner temporal split: 2008-2015 train, 2016-2018 val
        train_data = features_df.loc["2008":"2015"]
        val_data = features_df.loc["2016":"2018"]

        if len(train_data) == 0 or len(val_data) == 0:
            logger.warning(
                f"Empty train/val data for {train_gauge_id} in inner split, skipping"
            )
            continue

        # Separate features and targets
        feature_cols = [col for col in train_data.columns if col != "q_mm_day"]
        x_train_list.append(train_data[feature_cols].to_numpy())
        y_train_list.append(train_data["q_mm_day"].to_numpy())
        x_val_list.append(val_data[feature_cols].to_numpy())
        y_val_list.append(val_data["q_mm_day"].to_numpy())

    return x_train_list, y_train_list, x_val_list, y_val_list


def collect_full_training_data(
    train_gauge_ids: list[str],
    gauge_data_dict: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect full training data (2008-2018) from training basins.

    Args:
        train_gauge_ids: List of training gauge IDs
        gauge_data_dict: Dictionary mapping gauge_id -> data dict

    Returns:
        Tuple of (X_full_train, y_full_train)
    """
    x_full_train_list = []
    y_full_train_list = []

    for train_gauge_id in train_gauge_ids:
        features_df = gauge_data_dict[train_gauge_id]["features"]
        full_train_data = features_df.loc["2008":"2018"]

        if len(full_train_data) == 0:
            continue

        feature_cols = [col for col in full_train_data.columns if col != "q_mm_day"]
        x_full_train_list.append(full_train_data[feature_cols].to_numpy())
        y_full_train_list.append(full_train_data["q_mm_day"].to_numpy())

    x_full_train = np.vstack(x_full_train_list)
    y_full_train = np.concatenate(y_full_train_list)

    return x_full_train, y_full_train
