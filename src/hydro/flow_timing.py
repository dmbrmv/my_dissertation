"""Flow timing analysis for hydrological characterization.

This module provides tools for analyzing temporal characteristics of discharge,
including seasonal patterns, timing of extremes, and flow duration metrics.
"""

import numpy as np
import pandas as pd

from ..utils.logger import setup_logger

logger = setup_logger("flow_timing_analysis", log_file="logs/hydrological.log")


class FlowTiming:
    """Analysis of temporal flow characteristics and seasonal patterns.

    This class provides comprehensive timing analysis including seasonal flows,
    timing of extremes, and various temporal flow indices.
    """

    def __init__(self, discharge: pd.Series):
        """Initialize flow timing analysis.

        Args:
            discharge: Discharge time series with datetime index
        """
        if not isinstance(discharge.index, pd.DatetimeIndex):
            raise ValueError("Discharge series must have datetime index")

        self.discharge = discharge.dropna()
        self.daily_data = self._prepare_daily_data()

    def _prepare_daily_data(self) -> pd.DataFrame:
        """Prepare daily data with hydrological year information.

        Returns:
            DataFrame with discharge and temporal attributes
        """
        df = pd.DataFrame({"discharge": self.discharge})

        # Add temporal attributes
        df["month"] = df.index.month
        df["day_of_year"] = df.index.dayofyear
        df["year"] = df.index.year

        # Calculate hydrological year (starting October 1)
        df["hydro_year"] = df["year"].copy()
        df.loc[df["month"] >= 10, "hydro_year"] += 1

        return df

    def calculate_half_flow_date(self) -> dict[str, float]:
        """Calculate half-flow date for each hydrological year.

        The half-flow date is when cumulative discharge since Oct 1
        reaches half of the annual discharge.

        Returns:
            Dictionary with mean and std of half-flow dates (day of hydrological year)
        """
        hfd_values = []

        for hydro_year in self.daily_data["hydro_year"].unique():
            year_data = self.daily_data[self.daily_data["hydro_year"] == hydro_year].copy()

            if len(year_data) < 300:  # Skip incomplete years
                continue

            # Sort by date within hydrological year
            year_data = year_data.sort_index()

            # Calculate cumulative discharge
            year_data["cum_discharge"] = year_data["discharge"].cumsum()
            half_annual = year_data["cum_discharge"].iloc[-1] / 2

            # Find half-flow date
            try:
                hfd_idx = year_data[year_data["cum_discharge"] >= half_annual].index[0]
                # Convert to day of hydrological year (Oct 1 = day 1)
                if hfd_idx.month >= 10:
                    hfd_day = hfd_idx.dayofyear - 274 + 1  # Oct 1 = day 274
                else:
                    hfd_day = hfd_idx.dayofyear + 92  # Jan 1 = day 92 of hydro year

                hfd_values.append(hfd_day)
            except IndexError:
                continue

        if not hfd_values:
            return {"hfd_mean": np.nan, "hfd_std": np.nan}

        return {
            "hfd_mean": float(np.mean(hfd_values)),
            "hfd_std": float(np.std(hfd_values)),
        }

    def calculate_seasonal_flows(self) -> dict[str, float]:
        """Calculate seasonal flow statistics.

        Returns:
            Dictionary with seasonal flow metrics
        """
        # Define seasons (Northern Hemisphere)
        season_map = {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }

        seasonal_data = self.daily_data.copy()
        seasonal_data["season"] = seasonal_data["month"].map(season_map)

        seasonal_stats = {}
        for season in ["winter", "spring", "summer", "autumn"]:
            season_flows = seasonal_data[seasonal_data["season"] == season]["discharge"]
            if len(season_flows) > 0:
                mean_val = np.mean(season_flows)
                std_val = np.std(season_flows)
                seasonal_stats[f"{season}_mean"] = float(mean_val) if np.isfinite(mean_val) else float(np.nan)
                seasonal_stats[f"{season}_std"] = float(std_val) if np.isfinite(std_val) else float(np.nan)
                if np.isfinite(mean_val) and mean_val != 0:
                    seasonal_stats[f"{season}_cv"] = float(std_val / mean_val)
                else:
                    seasonal_stats[f"{season}_cv"] = float(np.nan)

        # Calculate seasonal flow ratios
        mean_flow = np.mean(self.discharge)
        for season in ["winter", "spring", "summer", "autumn"]:
            if f"{season}_mean" in seasonal_stats:
                seasonal_stats[f"{season}_ratio"] = seasonal_stats[f"{season}_mean"] / mean_flow

        return seasonal_stats

    def calculate_extreme_timing(self) -> dict[str, float]:
        """Calculate timing of flow extremes.

        Returns:
            Dictionary with timing metrics for extremes
        """
        # Annual maximum timing
        annual_max_timing = []
        annual_min_timing = []

        for year in self.daily_data["year"].unique():
            year_data = self.daily_data[self.daily_data["year"] == year]

            if len(year_data) < 300:  # Skip incomplete years
                continue

            # Find timing of annual maximum
            max_idx = year_data["discharge"].idxmax()
            max_timing = max_idx.dayofyear
            annual_max_timing.append(max_timing)

            # Find timing of annual minimum
            min_idx = year_data["discharge"].idxmin()
            min_timing = min_idx.dayofyear
            annual_min_timing.append(min_timing)

        timing_stats = {}

        if annual_max_timing:
            timing_stats.update(
                {
                    "max_timing_mean": float(np.mean(annual_max_timing)),
                    "max_timing_std": float(np.std(annual_max_timing)),
                }
            )

        if annual_min_timing:
            timing_stats.update(
                {
                    "min_timing_mean": float(np.mean(annual_min_timing)),
                    "min_timing_std": float(np.std(annual_min_timing)),
                }
            )

        return timing_stats

    def calculate_flow_duration_metrics(self) -> dict[str, float]:
        """Calculate flow duration and frequency metrics.

        Returns:
            Dictionary with duration-based metrics
        """
        mean_flow = np.mean(self.discharge)
        median_flow = np.median(self.discharge)

        # High flow threshold (2x median)
        high_flow_threshold = 2 * median_flow
        # Low flow threshold (0.2x mean)
        low_flow_threshold = 0.2 * mean_flow

        # Calculate durations
        high_flow_days = np.sum(self.discharge > high_flow_threshold)
        low_flow_days = np.sum(self.discharge < low_flow_threshold)
        total_days = len(self.discharge)

        # Calculate frequencies (percentage of time)
        high_flow_freq = (high_flow_days / total_days) * 100
        low_flow_freq = (low_flow_days / total_days) * 100

        # Calculate average duration of high/low flow events
        high_flow_events = self._get_event_durations(self.discharge > high_flow_threshold)
        low_flow_events = self._get_event_durations(self.discharge < low_flow_threshold)

        return {
            "high_flow_freq": high_flow_freq,
            "low_flow_freq": low_flow_freq,
            "high_flow_avg_duration": float(np.mean(high_flow_events)) if high_flow_events else 0.0,
            "low_flow_avg_duration": float(np.mean(low_flow_events)) if low_flow_events else 0.0,
            "high_flow_max_duration": float(np.max(high_flow_events)) if high_flow_events else 0.0,
            "low_flow_max_duration": float(np.max(low_flow_events)) if low_flow_events else 0.0,
        }

    def _get_event_durations(self, condition_series: pd.Series) -> list[int]:
        """Calculate durations of consecutive events.

        Args:
            condition_series: Boolean series indicating event occurrence

        Returns:
            List of event durations in days
        """
        # Find consecutive True periods
        diff = condition_series.astype(int).diff()
        starts = condition_series.index[diff == 1]
        ends = condition_series.index[diff == -1]

        # Handle edge cases
        if condition_series.iloc[0]:
            starts = pd.Index([condition_series.index[0]]).append(starts)
        if condition_series.iloc[-1]:
            ends = ends.append(pd.Index([condition_series.index[-1]]))

        if len(starts) == 0 or len(ends) == 0:
            return []

        # Calculate durations
        durations = []
        for start, end in zip(starts, ends, strict=False):
            duration = (end - start).days + 1
            durations.append(duration)

        return durations


def calculate_timing_metrics(discharge: pd.Series) -> dict[str, float]:
    """Calculate comprehensive flow timing metrics.

    Args:
        discharge: Discharge time series with datetime index

    Returns:
        Dictionary of timing metrics
    """
    timing = FlowTiming(discharge)

    metrics = {}

    # Half-flow date
    hfd_metrics = timing.calculate_half_flow_date()
    metrics.update(hfd_metrics)

    # Seasonal flows
    seasonal_metrics = timing.calculate_seasonal_flows()
    metrics.update(seasonal_metrics)

    # Extreme timing
    extreme_metrics = timing.calculate_extreme_timing()
    metrics.update(extreme_metrics)

    # Flow duration metrics
    duration_metrics = timing.calculate_flow_duration_metrics()
    metrics.update(duration_metrics)

    return metrics


def calculate_flow_regime_stability(discharge: pd.Series) -> dict[str, float]:
    """Calculate metrics indicating flow regime stability.

    Args:
        discharge: Discharge time series with datetime index

    Returns:
        Dictionary with stability metrics
    """
    timing = FlowTiming(discharge)

    # Inter-annual variability of seasonal flows
    seasonal_cv = {}
    annual_seasonal = {}

    for year in timing.daily_data["year"].unique():
        year_data = timing.daily_data[timing.daily_data["year"] == year]
        if len(year_data) < 300:
            continue

        # Calculate seasonal means for this year
        season_map = {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }

        year_data = year_data.copy()
        year_data["season"] = year_data["month"].map(season_map)

        for season in ["winter", "spring", "summer", "autumn"]:
            season_flows = year_data[year_data["season"] == season]["discharge"]
            if len(season_flows) > 0:
                if season not in annual_seasonal:
                    annual_seasonal[season] = []
                annual_seasonal[season].append(np.mean(season_flows))

    # Calculate coefficient of variation for each season
    for season, values in annual_seasonal.items():
        if len(values) > 1:
            seasonal_cv[f"{season}_interannual_cv"] = float(np.std(values) / np.mean(values))

    return seasonal_cv
