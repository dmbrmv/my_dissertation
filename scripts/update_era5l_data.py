#!/usr/bin/env python3
"""
Script to update NetCDF time series files with updated ERA5-Land data from CSV files.

Source: /media/dmbrmv/hdd_data/Russia/MeteoData/CamelsRU/era5_land/{gauge_id}.csv
  - Columns: t_max, t_min, prcp (and date as index)
  - Period: 2007-2024

Target: /home/dmbrmv/Development/Dissertation/data/time_series/{gauge_id}.nc
  - Columns to update: t_max_e5l, t_min_e5l, prcp_e5l
  - Update period: 2007-2020
"""

from pathlib import Path
import logging
import pandas as pd
import xarray as xr
from tqdm import tqdm

# Setup logging
log_file = Path("/home/dmbrmv/Development/Dissertation/logs/update_era5l.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def update_nc_with_csv(
    nc_path: Path,
    csv_path: Path,
    column_mapping: dict[str, str],
    update_period: tuple[str, str] = ("2007-01-01", "2020-12-31"),
) -> bool:
    """Update NetCDF file columns with data from CSV file, expanding date range if needed.

    Args:
        nc_path: Path to the NetCDF file to update
        csv_path: Path to source CSV file
        column_mapping: Dict mapping CSV columns to NC columns (e.g., {"t_max": "t_max_e5l"})
        update_period: Tuple of (start_date, end_date) for update period

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load CSV data
        csv_df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")

        # Filter CSV to update period
        csv_df = csv_df.loc[update_period[0]:update_period[1]]

        if csv_df.empty:
            logger.warning(f"No CSV data in update period for {csv_path.stem}")
            return False

        # Load NC file
        with xr.open_dataset(nc_path) as ds:
            nc_df = ds.to_dataframe()

        # Check that all target columns exist in NC file
        missing_cols = [nc_col for nc_col in column_mapping.values() if nc_col not in nc_df.columns]
        if missing_cols:
            logger.warning(f"Missing columns {missing_cols} in {nc_path.stem}, skipping")
            return False

        # Find dates in CSV that are not in NC (for expansion)
        new_dates = csv_df.index.difference(nc_df.index)
        
        # If there are new dates, expand the NC dataframe
        if len(new_dates) > 0:
            # Create rows for new dates with NaN values
            new_rows = pd.DataFrame(index=new_dates, columns=nc_df.columns)
            new_rows.index.name = nc_df.index.name
            
            # Concatenate and sort by date
            nc_df = pd.concat([nc_df, new_rows]).sort_index()
            
            logger.info(f"Expanded {nc_path.stem} with {len(new_dates)} new dates")

        # Update NC dataframe with CSV values for the update period
        update_idx = nc_df.index.intersection(csv_df.index)
        if len(update_idx) == 0:
            logger.warning(f"No dates to update for {nc_path.stem}")
            return False

        for csv_col, nc_col in column_mapping.items():
            if csv_col in csv_df.columns:
                nc_df.loc[update_idx, nc_col] = csv_df.loc[update_idx, csv_col].values

        # Convert back to xarray Dataset and save
        updated_ds = xr.Dataset.from_dataframe(nc_df)
        
        # Write to temporary file first, then replace
        tmp_path = nc_path.with_suffix(".tmp.nc")
        updated_ds.to_netcdf(tmp_path)
        
        # Replace original with updated file
        tmp_path.replace(nc_path)

        return True

    except Exception as e:
        logger.error(f"Error processing {nc_path.stem}: {e}")
        return False


def main():
    logger.info("Starting ERA5-Land data update")
    
    # Paths
    csv_dir = Path("/media/dmbrmv/hdd_data/Russia/MeteoData/CamelsRU/era5_land")
    nc_dir = Path("/home/dmbrmv/Development/Dissertation/data/time_series")

    # Column mapping: CSV column -> NC column
    column_mapping = {
        "t_max": "t_max_e5l",
        "t_min": "t_min_e5l",
        "prcp": "prcp_e5l",
    }

    # Update period
    update_period = ("2007-01-01", "2020-12-31")

    # Get all NC files
    nc_files = sorted(nc_dir.glob("*.nc"))
    logger.info(f"Found {len(nc_files)} NC files to process")

    # Process each NC file
    success_count = 0
    skip_count = 0
    error_count = 0

    for nc_path in tqdm(nc_files, desc="Updating NC files"):
        gauge_id = nc_path.stem
        csv_path = csv_dir / f"{gauge_id}.csv"

        if not csv_path.exists():
            skip_count += 1
            continue

        if update_nc_with_csv(nc_path, csv_path, column_mapping, update_period):
            success_count += 1
        else:
            error_count += 1

    logger.info("Complete!")
    logger.info(f"  Successfully updated: {success_count}")
    logger.info(f"  Skipped (no CSV): {skip_count}")
    logger.info(f"  Errors: {error_count}")


if __name__ == "__main__":
    main()
