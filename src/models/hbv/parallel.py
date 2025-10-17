"""Parallelization utilities for HBV calibration workflow."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing as mp

from src.utils.logger import setup_logger

logger = setup_logger("main_hbv_optuna", log_file="logs/hbv_optuna.log")


def run_parallel_optimization(
    gauge_ids: list[str],
    process_gauge_func,
    n_processes: int | None = None,
    **kwargs,
) -> None:
    """Run gauge processing in parallel using ProcessPoolExecutor."""
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)

    n_processes = max(1, min(n_processes, mp.cpu_count()))
    logger.info("Starting HBV parallel optimization with %s workers", n_processes)

    worker = partial(process_gauge_func, **kwargs)
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        list(executor.map(worker, gauge_ids))

    logger.info("HBV parallel optimization finished")


__all__ = ["run_parallel_optimization"]
