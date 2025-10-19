"""Main script to run Random Forest Regressor Optuna optimization for multiple gauges.

This script uses multi-objective optimization with:
- Temporal feature engineering with rolling windows
- PET calculation via Oudin formula
- Cyclic temporal encoding (day of year with sin/cos)
- Static watershed attributes from HydroATLAS
- Multi-objective optimization targeting all flow regimes
- Hyperparameter tuning based on scikit-learn best practices
- Enhanced metric weighting emphasizing low flows
"""

import multiprocessing as mp
from pathlib import Path
import sys

sys.path.append("./")
from src.models.rfr.parallel import process_rfr_gauge, run_parallel_optimization
from src.readers.geom_reader import load_geodata
from src.utils.logger import setup_logger

# Create log directories before importing modules that might use them
Path("logs").mkdir(exist_ok=True)

# Setup logger
logger = setup_logger("rfr_train", log_file="logs/rfr_train.log", level="INFO")


def main() -> None:
    """Main entry point for Random Forest training with enhanced features.

    Features added:
    - PET (Potential Evapotranspiration) via Oudin formula
    - Cyclic temporal encoding (day of year: sin/cos)
    - Static watershed attributes from HydroATLAS (17 features)
    - Previous observations via rolling window aggregations

    These enhancements enable the model to:
    1. Account for water balance (precipitation vs evapotranspiration)
    2. Capture seasonal patterns through cyclic encoding
    3. Incorporate catchment characteristics for spatial patterns
    4. Learn temporal dependencies through rolling statistics
    """
    # Load gauge geometries (for latitude extraction)
    logger.info("Loading gauge geometries...")
    _, gauges = load_geodata(folder_depth=".")
    logger.info(f"Loaded {len(gauges)} gauges")

    # Configuration
    calibration_period = ("2008-01-01", "2018-12-31")
    validation_period = ("2019-01-01", "2020-12-31")

    # Meteorological datasets to use
    datasets = ["e5l", "gpcp", "e5", "mswep"]

    # Rolling window sizes (days) - captures temporal memory
    rolling_windows = [1, 2, 4, 8, 16, 32]

    # Static watershed attributes from HydroATLAS
    static_file = Path("data/attributes/hydro_atlas_cis_camels.csv")
    static_parameters = [
        "for_pc_sse",  # Forest cover %
        "crp_pc_sse",  # Cropland %
        "inu_pc_ult",  # Inland water %
        "ire_pc_sse",  # Irrigated area %
        "lka_pc_use",  # Lake area %
        "prm_pc_sse",  # Permeable surface %
        "pst_pc_sse",  # Pasture %
        "cly_pc_sav",  # Clay content %
        "slt_pc_sav",  # Silt content %
        "snd_pc_sav",  # Sand content %
        "kar_pc_sse",  # Karst area %
        "urb_pc_sse",  # Urban area %
        "gwt_cm_sav",  # Groundwater table depth [cm]
        "lkv_mc_usu",  # Lake volume [million m³]
        "rev_mc_usu",  # Reservoir volume [million m³]
        "ws_area",  # Watershed area [km²]
        "ele_mt_sav",  # Mean elevation [m]
    ]

    if not static_file.exists():
        logger.error(f"Static attributes file not found: {static_file}")
        return

    # Optimization settings
    n_trials = 100  # Number of Optuna trials per gauge
    timeout = 3600  # 1 hour per gauge
    n_jobs = -1  # Use all available cores for RF training

    # Parallel processing settings
    n_processes = max(1, mp.cpu_count() - 1)

    # Output directory
    save_storage = Path("data/optimization/rfr_results")
    save_storage.mkdir(parents=True, exist_ok=True)

    # Get list of gauges
    gauge_data_dir = Path("data/nc_all_q")
    if not gauge_data_dir.exists():
        logger.error(f"Gauge data directory not found: {gauge_data_dir}")
        return

    gauge_ids = [f.stem for f in gauge_data_dir.glob("*.nc")]

    if not gauge_ids:
        logger.error("No gauge data files found")
        return

    logger.info("=" * 80)
    logger.info("RANDOM FOREST REGRESSOR WITH ENHANCED FEATURES")
    logger.info("=" * 80)
    logger.info(f"Found {len(gauge_ids)} gauges to process")
    logger.info(f"Calibration period: {calibration_period}")
    logger.info(f"Validation period: {validation_period}")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Rolling windows: {rolling_windows}")
    logger.info(f"Static parameters: {len(static_parameters)} features")
    logger.info("  - Land cover: forest, cropland, pasture, urban")
    logger.info("  - Soil: clay, silt, sand content")
    logger.info("  - Hydrology: lakes, reservoirs, groundwater depth")
    logger.info("  - Topography: elevation, watershed area")
    logger.info(f"Optimization: {n_trials} trials, {timeout}s timeout")
    logger.info(f"Parallel processing: {n_processes} processes")
    logger.info("=" * 80)
    logger.info("Feature engineering pipeline:")
    logger.info("  1. PET calculation (Oudin formula) using latitude + temperature")
    logger.info("  2. Cyclic temporal encoding (DOY sin/cos for seasonality)")
    logger.info("  3. Rolling window aggregations (1-32 days)")
    logger.info("  4. Static catchment attributes (17 HydroATLAS features)")
    logger.info("=" * 80)

    # Process gauges in parallel
    try:
        run_parallel_optimization(
            gauge_ids=gauge_ids,
            process_gauge_func=process_rfr_gauge,
            n_processes=n_processes,
            datasets=datasets,
            calibration_period=calibration_period,
            validation_period=validation_period,
            save_storage=save_storage,
            gauges_gdf=gauges,
            static_file=static_file,
            static_columns=static_parameters,
            rolling_windows=rolling_windows,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            overwrite_results=False,
        )

        logger.info("=" * 80)
        logger.info("✅ Random Forest optimization completed successfully")
        logger.info(f"Results saved to: {save_storage}")
        logger.info("=" * 80)
        logger.info("Next steps for ungauged basin prediction:")
        logger.info("  1. Extract spatial patterns from static attributes")
        logger.info("  2. Build region-specific models or transfer learning")
        logger.info("  3. Validate on held-out spatial test set")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during optimization: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
