"""Utilities for loading and processing hydrological data.

This module provides functions for identifying valid gauge stations,
loading meteorological and hydrological data, and creating datasets
for machine learning models.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr

from src.utils.logger import setup_logger

logger = setup_logger("main_catboost_optuna", log_file="logs/catboost_optuna.log", level="INFO")
# Define constants
LAGS = [2**n for n in range(6)]  # 1,2,4,8,16,32
ROLL_MEAN = ["t_min", "t_max"]


def find_valid_gauges(e_obs_ws: gpd.GeoDataFrame, hydro_dir: Path) -> tuple[list[str], list[str]]:
    """Find gauges with valid hydro data (no missing values).

    Args:
        e_obs_ws: GeoDataFrame with watershed information
        hydro_dir: Path to directory containing hydro data files

    Returns:
        tuple: (full_gauges, partial_gauges)
            - full_gauges: List of gauge IDs with complete data
            - partial_gauges: List of gauge IDs with some missing values
    """
    full_gauges = []
    partial_gauges = []
    for gauge_id in e_obs_ws.index:
        try:
            hydro_file = pd.read_csv(
                f"{hydro_dir}/{gauge_id}.csv", parse_dates=["date"], index_col="date"
            )[["q_mm_day"]]
        except FileNotFoundError:
            continue
        is_completely_nan = hydro_file["q_mm_day"].isna().all()
        if is_completely_nan:
            pass
        elif hydro_file["q_mm_day"].isna().sum() > 0:
            partial_gauges.append(gauge_id)
        else:
            full_gauges.append(gauge_id)

    return full_gauges, partial_gauges


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged features for precipitation and temperature variables.

    Creates rolling sum for precipitation and rolling mean for temperature
    variables using predefined lags in LAGS list.

    Args:
        df: DataFrame containing 'prcp' and temperature columns defined in ROLL_MEAN

    Returns:
        DataFrame with additional lagged features
    """
    for d in LAGS:
        df[f"prcp_{d}"] = df["prcp"].rolling(d, min_periods=1).sum()
        for v in ROLL_MEAN:
            df[f"{v}_mean_{d}"] = df[v].rolling(d, min_periods=1).mean()
    return df


def load_one_site(
    gid: str, meteo_dir: Path, hydro_dir: Path, temp_dir: Path | None = None
) -> pd.DataFrame:
    """Load meteorological and hydrological data for a specific gauge.

    Reads NetCDF meteorological data and CSV hydrological data for the
    specified gauge ID and merges them based on date.

    Args:
        gid: Gauge ID string
        meteo_dir: Path to directory containing meteorological data
        hydro_dir: Path to directory containing hydrological data
        temp_dir: Optional path for temporary files (not used in this function)

    Returns:
        DataFrame containing merged meteorological and hydrological data
    """
    # Load meteorological data with xarray (efficient for NetCDF)
    if "mswep" == meteo_dir.parent.stem:
        with xr.open_dataset(f"{meteo_dir}/{gid}.nc") as ds:
            mswep_file = ds.to_dataframe()
        # Load ERA5-Land data and merge with MSWEP
        with xr.open_dataset(f"{temp_dir}/{gid}.nc") as ds:
            meteo = ds.to_dataframe()
        meteo.loc[:, "prcp"] = mswep_file.loc[:, "prcp"]
    else:
        with xr.open_dataset(f"{meteo_dir}/{gid}.nc") as ds:
            meteo = ds.to_dataframe()
    try:
        q = pd.read_csv(
            f"{hydro_dir}/Discharge/full/decent/{gid}.csv", parse_dates=["date"], index_col="date"
        ).rename(columns={"q_mm_day": "q"})[["q"]]
    except FileNotFoundError:
        try:
            q = pd.read_csv(
                f"{hydro_dir}/Discharge/partial/decent/{gid}.csv",
                parse_dates=["date"],
                index_col="date",
            ).rename(columns={"q_mm_day": "q"})[["q"]]
        except FileNotFoundError:
            logger.error(f"Gauge {gid} not found in hydro data directory.")
    # --- merge & add site id ---
    df = pd.concat([meteo, q], axis=1).loc["2008":"2022", :]
    df["gauge_id"] = gid
    df = df.reset_index()

    return df


def data_creator(
    full_gauges: list[str],
    static_data: pd.DataFrame,
    meteo_dir: Path,
    hydro_dir: Path,
    temp_dir: Path | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Create a DataFrame with static and dynamic features.

    Args:
        full_gauges: List of gauge IDs to include in the dataset
        static_data: DataFrame with static features
        meteo_dir: Path to directory containing meteorological data
        hydro_dir: Path to directory containing hydrological data
        temp_dir: Optional path for temperature files (not used in this function)
        n_workers: Number of processes for parallel data loading

    Returns:
        DataFrame with static and dynamic features, indexed by date and gauge ID
    """
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(load_one_site, gid, meteo_dir, hydro_dir, temp_dir): gid
            for gid in full_gauges
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    dynamic = pd.concat(results, ignore_index=True)
    dynamic = (
        dynamic.groupby("gauge_id", group_keys=False)
        .apply(add_lags)
        .dropna(subset=[f"prcp_{max(LAGS)}"])  # avoid warm-up NaNs
    )
    # Merge static and dynamic data
    if "gauge_id" not in static_data.columns:
        static_data = static_data.reset_index()

    merged = dynamic.merge(static_data, on="gauge_id", how="left")
    # Return the merged dataset
    return merged


def select_uncorrelated_features(
    data: pd.DataFrame, threshold: float = 0.7, min_valid_fraction: float = 0.8
) -> list[str]:
    """Select features from the DataFrame.

    Features are not highly correlated, have sufficient valid data,
    and do not contain '_cl_' in their names.

    Args:
        data: Input DataFrame with features.
        threshold: Absolute correlation threshold above which features are considered correlated.
        min_valid_fraction: Minimum fraction of non-zero and non-NaN values required to keep a feature.

    Returns:
        List of column names representing uncorrelated features with sufficient valid data
        and without '_cl_' in their names.
    """
    import numpy as np

    # Exclude columns containing '_cl_' in their names
    filtered_cols = [col for col in data.columns if "_cl_" not in col]
    filtered_data = data[filtered_cols]

    # Filter out columns with less than min_valid_fraction valid (non-zero, non-NaN) data
    valid_mask = (filtered_data != 0) & (~filtered_data.isna())
    valid_fraction = valid_mask.sum(axis=0) / len(filtered_data)
    sufficient_data_cols = valid_fraction[valid_fraction >= min_valid_fraction].index.tolist()

    # Subset data to columns with sufficient valid data
    filtered_data = filtered_data[sufficient_data_cols]

    # Compute the absolute correlation matrix
    corr_matrix = filtered_data.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Identify columns to drop based on correlation threshold
    to_drop = set()
    for col in upper.columns:
        if any(upper[col] > threshold):
            to_drop.add(col)

    # Features to keep are those not in to_drop
    selected_features = [col for col in filtered_data.columns if col not in to_drop]
    return selected_features
