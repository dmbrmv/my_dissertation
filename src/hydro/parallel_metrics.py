"""Parallel computation of hydrological metrics for multiple gauges.

This module provides multiprocessing-based parallelization for calculating
comprehensive hydrological metrics across multiple gauge stations efficiently.
Optimized for large-scale hydrological analysis with proper error handling
and progress tracking.
"""

import logging
from multiprocessing import Pool, cpu_count
from typing import Literal

import pandas as pd
from tqdm import tqdm

from src.hydro.period_based_metrics import calculate_comprehensive_metrics

logger = logging.getLogger(__name__)


def _calculate_metrics_worker(
    gauge_id: str,
    discharge: pd.Series,
    period_type: Literal["calendar", "hydrological"],
    hydro_year_start_month: int,
    min_data_fraction: float,
    min_periods: int,
    aggregation: Literal["mean", "median"],
) -> tuple[str, dict[str, float] | None]:
    """Worker function for parallel metric calculation.

    Args:
        gauge_id: Unique gauge identifier
        discharge: Time series of discharge values
        period_type: Type of period ("hydrological" or "calendar")
        hydro_year_start_month: Start month for hydrological year (1-12)
        min_data_fraction: Minimum fraction of valid data required (0-1)
        min_periods: Minimum number of periods required
        aggregation: Aggregation method ("mean", "median", etc.)

    Returns:
        Tuple of (gauge_id, metrics_dict) or (gauge_id, None) if error
    """
    try:
        metrics = calculate_comprehensive_metrics(
            discharge,
            period_type=period_type,
            hydro_year_start_month=hydro_year_start_month,
            min_data_fraction=min_data_fraction,
            min_periods=min_periods,
            aggregation=aggregation,
        )
        return (gauge_id, metrics)
    except Exception as e:
        # Log error but don't raise - allow other gauges to process
        logger.error(f"Metric calculation failed for gauge {gauge_id}: {e!s}")
        return (gauge_id, None)


def calculate_metrics_parallel(
    discharge_data: dict[str, pd.Series],
    gauge_ids: pd.Index | None = None,
    n_workers: int | None = None,
    period_type: Literal["calendar", "hydrological"] = "hydrological",
    hydro_year_start_month: int = 10,
    min_data_fraction: float = 0.7,
    min_periods: int = 5,
    aggregation: Literal["mean", "median"] = "mean",
    show_progress: bool = True,
) -> dict[str, dict[str, float]]:
    """Calculate comprehensive metrics for multiple gauges in parallel.

    Uses multiprocessing to distribute metric calculations across CPU cores,
    significantly improving performance for large gauge networks.

    Args:
        discharge_data: Dict mapping gauge_id -> discharge time series
        gauge_ids: Optional index/list of gauge IDs to process.
            If None, processes all gauges in discharge_data
        n_workers: Number of parallel workers. Defaults to (CPU count - 1)
        period_type: Type of period ("hydrological" or "calendar")
        hydro_year_start_month: Start month for hydrological year (1-12)
        min_data_fraction: Minimum fraction of valid data required (0-1)
        min_periods: Minimum number of periods required
        aggregation: Aggregation method ("mean", "median", etc.)
        show_progress: Show progress bar during computation

    Returns:
        Dict mapping gauge_id -> metrics dict
            Only includes successfully calculated metrics

    Example:
        >>> discharge_dict = {"gauge_1": pd.Series(...), "gauge_2": pd.Series(...)}
        >>> metrics = calculate_metrics_parallel(
        ...     discharge_data=discharge_dict,
        ...     n_workers=8,
        ...     period_type="hydrological",
        ... )
        >>> print(metrics["gauge_1"]["mean_discharge"])
    """
    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    # Filter gauges to process
    if gauge_ids is None:
        gauges_to_process = list(discharge_data.keys())
    else:
        gauges_to_process = [gid for gid in discharge_data.keys() if gid in gauge_ids]

    logger.info(
        f"Starting parallel metric calculation for {len(gauges_to_process)} "
        f"gauges using {n_workers} workers"
    )

    # Prepare arguments for workers
    worker_args = [
        (
            gauge_id,
            discharge_data[gauge_id],
            period_type,
            hydro_year_start_month,
            min_data_fraction,
            min_periods,
            aggregation,
        )
        for gauge_id in gauges_to_process
    ]

    # Execute in parallel
    all_metrics: dict[str, dict[str, float]] = {}

    with Pool(processes=n_workers) as pool:
        # Use starmap to unpack argument tuples
        if show_progress:
            results = list(
                tqdm(
                    pool.starmap(_calculate_metrics_worker, worker_args),
                    total=len(worker_args),
                    desc="Calculating metrics",
                )
            )
        else:
            results = pool.starmap(_calculate_metrics_worker, worker_args)

    # Collect successful results
    failed_count = 0
    for gauge_id, metrics in results:
        if metrics is not None:
            all_metrics[gauge_id] = metrics
        else:
            failed_count += 1

    logger.info(f"Metric calculation complete: {len(all_metrics)} successful, {failed_count} failed")

    return all_metrics


def calculate_metrics_batched(
    discharge_data: dict[str, pd.Series],
    gauge_ids: pd.Index | None = None,
    batch_size: int = 100,
    n_workers: int | None = None,
    period_type: Literal["calendar", "hydrological"] = "hydrological",
    hydro_year_start_month: int = 10,
    min_data_fraction: float = 0.7,
    min_periods: int = 5,
    aggregation: Literal["mean", "median"] = "mean",
) -> dict[str, dict[str, float]]:
    """Calculate metrics in batches for better memory management.

    Useful when processing very large gauge networks (>1000 gauges) to
    prevent memory overflow.

    Args:
        discharge_data: Dict mapping gauge_id -> discharge time series
        gauge_ids: Optional index/list of gauge IDs to process
        batch_size: Number of gauges to process per batch
        n_workers: Number of parallel workers per batch
        period_type: Type of period ("hydrological" or "calendar")
        hydro_year_start_month: Start month for hydrological year (1-12)
        min_data_fraction: Minimum fraction of valid data required (0-1)
        min_periods: Minimum number of periods required
        aggregation: Aggregation method ("mean", "median", etc.)

    Returns:
        Dict mapping gauge_id -> metrics dict
    """
    # Filter gauges
    if gauge_ids is None:
        gauges_to_process = list(discharge_data.keys())
    else:
        gauges_to_process = [gid for gid in discharge_data.keys() if gid in gauge_ids]

    logger.info(f"Processing {len(gauges_to_process)} gauges in batches of {batch_size}")

    all_metrics: dict[str, dict[str, float]] = {}

    # Process in batches
    for i in range(0, len(gauges_to_process), batch_size):
        batch_ids = gauges_to_process[i : i + batch_size]
        batch_data = {gid: discharge_data[gid] for gid in batch_ids}

        logger.info(
            f"Processing batch {i // batch_size + 1}: "
            f"gauges {i + 1}-{min(i + batch_size, len(gauges_to_process))}"
        )

        batch_metrics = calculate_metrics_parallel(
            discharge_data=batch_data,
            gauge_ids=None,  # Already filtered
            n_workers=n_workers,
            period_type=period_type,
            hydro_year_start_month=hydro_year_start_month,
            min_data_fraction=min_data_fraction,
            min_periods=min_periods,
            aggregation=aggregation,
            show_progress=True,
        )

        all_metrics.update(batch_metrics)

    logger.info(f"Batch processing complete: {len(all_metrics)} total gauges")
    return all_metrics
