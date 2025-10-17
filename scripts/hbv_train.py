"""HBV calibration launcher mirroring the GR4J pipeline."""

from __future__ import annotations

from functools import partial
import multiprocessing as mp
from pathlib import Path

import geopandas as gpd

from src.models.hbv.hbv_calibrator import process_hbv_gauge
from src.models.hbv.parallel import run_parallel_optimization
from src.readers.geom_reader import load_geodata
from src.utils.logger import setup_logger


def main() -> None:
    """Entry point for HBV calibration when used as a module."""
    Path("logs").mkdir(exist_ok=True)
    logger = setup_logger("main_hbv_optuna", log_file="logs/hbv_optuna.log", level="INFO")

    _, gauges = load_geodata(folder_depth=".")

    if not isinstance(gauges, gpd.GeoDataFrame):
        raise RuntimeError("Gauge GeoDataFrame is required for HBV calibration.")

    logger.info("Scanning available NetCDF gauge files...")
    available_gauges = [
        i.stem for i in Path("data/nc_all_q").glob("*.nc") if i.stem in gauges.index
    ]
    logger.info("Found %s gauges with spatial metadata.", len(available_gauges))

    calibration_period = ("2010-01-01", "2018-12-31")
    validation_period = ("2019-01-01", "2020-12-31")
    warmup_years = 2
    n_trials = 4200
    timeout = 1200
    overwrite_results = False
    datasets = ["e5l", "gpcp", "e5", "mswep"]

    save_storage = Path("data/res/hbv_optuna")
    save_storage.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("IMPROVED HBV CALIBRATION")
    logger.info("=" * 80)
    logger.info("Total gauges: %s", len(available_gauges))
    logger.info("Datasets: %s", ", ".join(datasets))
    logger.info("Calibration period: %s – %s", *calibration_period)
    logger.info("Validation period: %s – %s", *validation_period)
    logger.info("Warm-up period: %s years", warmup_years)
    logger.info("Trials per gauge/dataset: %s", n_trials)
    logger.info("Timeout per gauge/dataset: %s seconds", timeout)
    logger.info("Results directory: %s", save_storage)
    logger.info("=" * 80)
    logger.info("Key improvements:")
    logger.info("  ✓ Warm-up aware HBV simulation")
    logger.info("  ✓ Composite multi-objective calibration")
    logger.info("  ✓ Weighted Pareto selection & diagnostics")
    logger.info("=" * 80)

    n_processes = max(1, mp.cpu_count() - 2)
    logger.info("Starting HBV calibration with %s workers", n_processes)

    run_parallel_optimization(
        gauge_ids=available_gauges,
        process_gauge_func=partial(
            process_hbv_gauge,
            datasets=datasets,
            calibration_period=calibration_period,
            validation_period=validation_period,
            save_storage=save_storage,
            e_obs_gauge=gauges,
            n_trials=n_trials,
            timeout=timeout,
            warmup_years=warmup_years,
            overwrite_results=overwrite_results,
        ),
        n_processes=n_processes,
    )

    logger.info("=" * 80)
    logger.info("✅ HBV calibration finished.")
    logger.info("Results stored in %s", save_storage)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
