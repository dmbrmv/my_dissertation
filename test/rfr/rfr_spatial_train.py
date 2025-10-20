"""Training script for regional RFR with nested LOBO cross-validation.

This script implements a universal (spatial) Random Forest model for ungauged
basin prediction using Leave-One-Basin-Out (LOBO) cross-validation.

Nested CV structure:
- Outer loop: LOBO - each basin held out once for testing
- Inner loop: Fixed temporal split (2008-2015 train, 2016-2018 val) for tuning
- Test period: 2019-2020 on the held-out basin

The universal model learns spatial patterns across catchments by training on
multiple basins simultaneously, enabling predictions for ungauged basins.
"""

import multiprocessing as mp
from pathlib import Path
import sys

sys.path.append("./")

from src.models.rfr_spatial.lobo import run_lobo_optimization
from src.readers.geom_reader import load_geodata
from src.utils.logger import setup_logger

# Create log directories
Path("logs").mkdir(exist_ok=True)

# Setup logger
logger = setup_logger("rfr_spatial_train", log_file="logs/rfr_spatial_train.log")


def main() -> None:
    """Main entry point for regional RFR training with LOBO cross-validation.

    Strategy:
    1. Load all gauge data with features (temporal + static)
    2. For each basin (outer LOBO loop):
       a. Hold out basin for testing (2019-2020)
       b. Train on remaining basins with inner temporal split:
          - 2008-2015: training
          - 2016-2018: validation for hyperparameter tuning
       c. Train final model on full data (2008-2018) from training basins
       d. Test on held-out basin (2019-2020)
    3. Report cross-validation metrics across all folds

    This provides an unbiased estimate of spatial generalization performance.
    """
    # Load gauge geometries (for latitude extraction)
    logger.info("Loading gauge geometries...")
    _, gauges = load_geodata(folder_depth=".")
    logger.info(f"Loaded {len(gauges)} gauges")

    # Configuration
    # Meteorological dataset to use (single dataset for universal model)
    dataset = "e5l"  # ERA5-Land (most complete and reliable)

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

    # Optimization settings per fold
    n_trials = 10  # Number of Optuna trials per LOBO fold (reduce for speed)
    timeout = 1800  # 30 minutes per fold
    n_jobs = -1  # Use all available cores for RF training

    # Output directory
    save_storage = Path("data/optimization/rfr_spatial_results")
    save_storage.mkdir(parents=True, exist_ok=True)

    # Get list of gauges from data directory
    gauge_data_dir = Path("data/nc_all_q")
    if not gauge_data_dir.exists():
        logger.error(f"Gauge data directory not found: {gauge_data_dir}")
        return

    # Get all available gauge IDs
    gauge_files = sorted(gauge_data_dir.glob("*.nc"))
    gauge_ids = [f.stem for f in gauge_files if f.stem in gauges.index]

    logger.info(f"Found {len(gauge_ids)} gauges in {gauge_data_dir}")

    # Optional: Limit to subset for testing
    # gauge_ids = gauge_ids[:6]  # Uncomment to test on first 10 gauges
    # logger.info(f"Testing on {len(gauge_ids)} gauges (subset)")

    # Run LOBO cross-validation
    logger.info(
        f"\n{'=' * 80}\n"
        f"Starting LOBO Cross-Validation for Regional RFR\n"
        f"{'=' * 80}\n"
        f"Configuration:\n"
        f"  - Dataset: {dataset}\n"
        f"  - Basins: {len(gauge_ids)}\n"
        f"  - Trials per fold: {n_trials}\n"
        f"  - Timeout per fold: {timeout}s\n"
        f"  - Static features: {len(static_parameters)}\n"
        f"  - Rolling windows: {rolling_windows}\n"
        f"  - Output: {save_storage}\n"
    )

    lobo_results = run_lobo_optimization(
        gauge_ids=gauge_ids,
        dataset=dataset,
        gauges_gdf=gauges,
        static_file=static_file,
        static_columns=static_parameters,
        rolling_windows=rolling_windows,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        output_dir=save_storage,
    )

    if lobo_results:
        logger.info(
            f"\n{'=' * 80}\n"
            f"LOBO Cross-Validation Complete\n"
            f"{'=' * 80}\n"
            f"Successfully completed {len(lobo_results)} folds\n"
            f"Results saved to: {save_storage}\n"
        )

        # Quick summary
        import pandas as pd

        results_df = pd.DataFrame(lobo_results).T
        logger.info("\nPerformance Summary:")
        for metric in ["KGE", "NSE", "RMSE", "MAE", "PBIAS"]:
            if metric in results_df.columns:
                mean_val = results_df[metric].mean()
                std_val = results_df[metric].std()
                median_val = results_df[metric].median()
                logger.info(
                    f"  {metric}: mean={mean_val:.3f}, "
                    f"std={std_val:.3f}, median={median_val:.3f}"
                )

        # Identify best and worst performing basins
        if "KGE" in results_df.columns:
            best_basin = results_df["KGE"].idxmax()
            worst_basin = results_df["KGE"].idxmin()
            logger.info(
                f"\n  Best basin: {best_basin} "
                f"(KGE={results_df.loc[best_basin, 'KGE']:.3f})"
            )
            logger.info(
                f"  Worst basin: {worst_basin} "
                f"(KGE={results_df.loc[worst_basin, 'KGE']:.3f})"
            )

    else:
        logger.error("LOBO cross-validation failed - no results returned")


if __name__ == "__main__":
    # Ensure multiprocessing works correctly
    mp.set_start_method("spawn", force=True)
    main()
