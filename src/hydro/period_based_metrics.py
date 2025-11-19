"""Period-based hydrological metrics calculation.

This module provides period-based (annual) calculation of hydrological metrics,
aggregating over multiple years to get robust estimates while handling NaNs properly.
"""

from typing import Literal

import numpy as np
import pandas as pd

from ..utils.logger import setup_logger
from .base_flow import calculate_bfi
from .flow_duration import calculate_fdc_metrics
from .flow_extremes import calculate_extreme_metrics
from .flow_timing import calculate_timing_metrics

logger = setup_logger("period_based_metrics", log_file="logs/hydrological.log")


def split_by_period(
    discharge: pd.Series,
    period_type: Literal["calendar", "hydrological"] = "hydrological",
    hydro_year_start_month: int = 10,
) -> dict[int, pd.Series]:
    """Split discharge time series into annual periods.

    Args:
        discharge: Discharge time series with datetime index
        period_type: 'calendar' (Jan-Dec) or 'hydrological' (Oct-Sep by default)
        hydro_year_start_month: Starting month for hydrological year (1-12)

    Returns:
        Dictionary mapping year -> discharge series for that period
    """
    if not isinstance(discharge.index, pd.DatetimeIndex):
        raise ValueError("Discharge series must have DatetimeIndex")

    periods = {}

    if period_type == "calendar":
        for year in discharge.index.year.unique():
            year_data = discharge[discharge.index.year == year]
            if len(year_data) > 0:
                periods[year] = year_data

    elif period_type == "hydrological":
        # Hydrological year: e.g., Oct 2008 - Sep 2009 -> Year 2009
        df = discharge.to_frame(name="discharge")
        df["hydro_year"] = df.index.year
        df.loc[df.index.month >= hydro_year_start_month, "hydro_year"] += 1

        for year in df["hydro_year"].unique():
            year_data = df[df["hydro_year"] == year]["discharge"]
            if len(year_data) > 0:
                periods[year] = year_data

    return periods


def calculate_period_metrics(
    discharge: pd.Series,
    min_data_fraction: float = 0.7,
) -> dict[str, float]:
    """Calculate hydrological metrics for a single period with NaN handling.

    Args:
        discharge: Discharge series for one period
        min_data_fraction: Minimum fraction of valid data required (0-1)

    Returns:
        Dictionary of metrics (NaN if insufficient data)
    """
    # Check data availability
    valid_frac = discharge.notna().sum() / len(discharge)
    if valid_frac < min_data_fraction:
        logger.debug(f"Insufficient data: {valid_frac:.2%} < {min_data_fraction:.2%}")
        return _nan_metrics()

    metrics = {}

    # 1. Mean daily discharge (NaN-aware)
    metrics["mean_discharge"] = float(np.nanmean(discharge))
    metrics["median_discharge"] = float(np.nanmedian(discharge))
    metrics["std_discharge"] = float(np.nanstd(discharge))

    # 2. Total annual discharge
    metrics["total_discharge"] = float(np.nansum(discharge))

    # 3. Coefficient of variation
    mean_val = metrics["mean_discharge"]
    metrics["cv_discharge"] = metrics["std_discharge"] / mean_val if mean_val > 0 else np.nan

    # 4. Flow Duration Curve metrics
    try:
        fdc_metrics = calculate_fdc_metrics(discharge)
        metrics["fdc_slope"] = fdc_metrics.get("fdc_slope", np.nan)
        metrics["q05"] = fdc_metrics.get("q05", np.nan)
        metrics["q50"] = fdc_metrics.get("q50", np.nan)
        metrics["q95"] = fdc_metrics.get("q95", np.nan)
    except Exception as e:
        logger.debug(f"FDC calculation failed: {e}")
        metrics["fdc_slope"] = np.nan
        metrics["q05"] = np.nan
        metrics["q50"] = np.nan
        metrics["q95"] = np.nan

    # 5. Baseflow index
    try:
        metrics["baseflow_index"] = calculate_bfi(discharge)
    except Exception as e:
        logger.debug(f"BFI calculation failed: {e}")
        metrics["baseflow_index"] = np.nan

    # 6. Flow extremes
    try:
        extreme_metrics = calculate_extreme_metrics(discharge)
        metrics["high_flow_frequency"] = extreme_metrics.get("high_flow_frequency", np.nan)
        metrics["high_flow_avg_duration"] = extreme_metrics.get("high_flow_avg_duration", np.nan)
        metrics["low_flow_frequency"] = extreme_metrics.get("low_flow_frequency", np.nan)
        metrics["low_flow_avg_duration"] = extreme_metrics.get("low_flow_avg_duration", np.nan)
    except Exception as e:
        logger.debug(f"Extreme metrics calculation failed: {e}")
        metrics["high_flow_frequency"] = np.nan
        metrics["high_flow_avg_duration"] = np.nan
        metrics["low_flow_frequency"] = np.nan
        metrics["low_flow_avg_duration"] = np.nan

    # 7. Timing metrics (if available)
    try:
        timing = calculate_timing_metrics(discharge)
        metrics["mean_half_flow_date"] = timing.get("hfd_mean", np.nan)
    except Exception as e:
        logger.debug(f"Timing metrics calculation failed: {e}")
        metrics["mean_half_flow_date"] = np.nan

    return metrics


def _nan_metrics() -> dict[str, float]:
    """Return dictionary of NaN metrics when data is insufficient."""
    return {
        "mean_discharge": np.nan,
        "median_discharge": np.nan,
        "std_discharge": np.nan,
        "total_discharge": np.nan,
        "cv_discharge": np.nan,
        "fdc_slope": np.nan,
        "q05": np.nan,
        "q50": np.nan,
        "q95": np.nan,
        "baseflow_index": np.nan,
        "high_flow_frequency": np.nan,
        "high_flow_avg_duration": np.nan,
        "low_flow_frequency": np.nan,
        "low_flow_avg_duration": np.nan,
        "mean_half_flow_date": np.nan,
    }


def aggregate_period_metrics(
    period_metrics: dict[int, dict[str, float]],
    aggregation: Literal["mean", "median", "std"] = "mean",
    min_periods: int = 3,
) -> dict[str, float]:
    """Aggregate metrics across multiple periods.

    Args:
        period_metrics: Dictionary mapping year -> metrics dict
        aggregation: Aggregation method ('mean', 'median', 'std')
        min_periods: Minimum number of valid periods required

    Returns:
        Dictionary of aggregated metrics
    """
    if len(period_metrics) < min_periods:
        logger.warning(f"Insufficient periods: {len(period_metrics)} < {min_periods}")
        return _nan_metrics()

    # Convert to DataFrame for easier aggregation
    df = pd.DataFrame.from_dict(period_metrics, orient="index")

    # Check minimum valid data per metric
    result = {}
    for col in df.columns:
        valid_count = df[col].notna().sum()

        if valid_count < min_periods:
            result[col] = np.nan
            continue

        if aggregation == "mean":
            result[col] = float(np.nanmean(df[col]))
        elif aggregation == "median":
            result[col] = float(np.nanmedian(df[col]))
        elif aggregation == "std":
            result[col] = float(np.nanstd(df[col]))
        else:
            raise ValueError(f"Invalid aggregation: {aggregation}")

    # Add inter-annual variability metrics
    result["n_valid_periods"] = len(period_metrics)
    result["mean_discharge_std"] = float(np.nanstd(df["mean_discharge"]))
    result["mean_discharge_cv"] = (
        result["mean_discharge_std"] / result.get("mean_discharge", np.nan)
        if result.get("mean_discharge", 0) > 0
        else np.nan
    )

    return result


def calculate_comprehensive_metrics(
    discharge: pd.Series,
    period_type: Literal["calendar", "hydrological"] = "hydrological",
    hydro_year_start_month: int = 10,
    min_data_fraction: float = 0.7,
    min_periods: int = 3,
    aggregation: Literal["mean", "median"] = "mean",
) -> dict[str, float]:
    """Calculate comprehensive period-based hydrological metrics.

    This is the main entry point for period-based metrics calculation.

    Args:
        discharge: Full discharge time series with datetime index
        period_type: 'calendar' or 'hydrological'
        hydro_year_start_month: Starting month for hydrological year
        min_data_fraction: Minimum data fraction per period (0-1)
        min_periods: Minimum number of valid periods required
        aggregation: Method to aggregate across periods

    Returns:
        Dictionary of aggregated hydrological metrics

    Example:
        >>> discharge = pd.Series(...)  # 2008-2023 data
        >>> metrics = calculate_comprehensive_metrics(
        ...     discharge, period_type="hydrological", min_periods=5
        ... )
        >>> # Returns mean of annual metrics (e.g., ~15 years -> 15 values -> mean)
    """
    # Split into periods
    periods = split_by_period(
        discharge,
        period_type=period_type,
        hydro_year_start_month=hydro_year_start_month,
    )

    # Calculate metrics for each period
    period_metrics = {}
    for year, period_discharge in periods.items():
        try:
            metrics = calculate_period_metrics(
                period_discharge,
                min_data_fraction=min_data_fraction,
            )
            period_metrics[year] = metrics
        except Exception as e:
            logger.error(f"Error calculating metrics for year {year}: {e}")
            continue

    # Aggregate across periods
    aggregated = aggregate_period_metrics(
        period_metrics,
        aggregation=aggregation,
        min_periods=min_periods,
    )

    return aggregated


def calculate_runoff_ratio(
    discharge: pd.Series,
    precipitation: pd.Series,
    period_type: Literal["calendar", "hydrological"] = "hydrological",
    hydro_year_start_month: int = 10,
    min_periods: int = 3,
) -> float:
    """Calculate runoff ratio from paired discharge and precipitation data.

    Args:
        discharge: Discharge time series (mm/day)
        precipitation: Precipitation time series (mm/day)
        period_type: Period definition
        hydro_year_start_month: Starting month for hydrological year
        min_periods: Minimum number of valid periods

    Returns:
        Mean annual runoff ratio (dimensionless)
    """
    # Split both series
    discharge_periods = split_by_period(discharge, period_type, hydro_year_start_month)
    precip_periods = split_by_period(precipitation, period_type, hydro_year_start_month)

    # Find common years
    common_years = set(discharge_periods.keys()) & set(precip_periods.keys())

    if len(common_years) < min_periods:
        logger.warning(f"Insufficient common years: {len(common_years)} < {min_periods}")
        return np.nan

    # Calculate annual runoff ratios
    annual_ratios = []
    for year in sorted(common_years):
        total_q = np.nansum(discharge_periods[year])
        total_p = np.nansum(precip_periods[year])

        if total_p > 0:
            annual_ratios.append(total_q / total_p)

    if len(annual_ratios) < min_periods:
        return np.nan

    return float(np.nanmean(annual_ratios))
