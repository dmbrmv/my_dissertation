"""Flow extremes analysis for hydrological characterization.

This module provides tools for analyzing high and low flow extremes,
including frequency analysis, duration analysis, and extreme value statistics.
"""

import numpy as np
import pandas as pd

from ..utils.logger import setup_logger

logger = setup_logger("flow_extremes_analysis", log_file="logs/hydrological.log")


class FlowExtremes:
    """Analysis of flow extremes including high and low flow events.

    This class provides comprehensive extreme flow analysis including
    frequency analysis, duration statistics, and magnitude-frequency relationships.
    """

    def __init__(self, discharge: pd.Series):
        """Initialize flow extremes analysis.

        Args:
            discharge: Discharge time series with datetime index
        """
        self.discharge = discharge.dropna()
        self.mean_flow = np.mean(self.discharge)
        self.median_flow = np.median(self.discharge)

    def calculate_flow_quantiles(
        self, quantiles: list[float] | None = None
    ) -> dict[str, float]:
        """Calculate flow quantiles for extreme analysis.

        IMPORTANT HYDROLOGICAL CONVENTION:
            - Q5 (High Flow) = flow exceeded 5% of time = 95th percentile = quantile(0.95)
            - Q95 (Low Flow) = flow exceeded 95% of time = 5th percentile = quantile(0.05)

        This function returns PERCENTILES (q05_0, q95_0), NOT exceedance probabilities.
        To get hydrological Q5 and Q95, map as follows:
            - obs_q5 = quantiles["q95_0"]   # High flow (95th percentile)
            - obs_q95 = quantiles["q05_0"]  # Low flow (5th percentile)

        Args:
            quantiles: List of quantiles (0-1). Default: standard extreme quantiles

        Returns:
            Dictionary of flow quantiles with keys like "q05_0" (5th percentile),
            "q95_0" (95th percentile), etc.

        Example:
            >>> extremes = FlowExtremes(discharge_series)
            >>> quantiles = extremes.calculate_flow_quantiles([0.05, 0.95])
            >>> high_flow_q5 = quantiles["q95_0"]  # 95th percentile = Q5
            >>> low_flow_q95 = quantiles["q05_0"]  # 5th percentile = Q95
        """
        if quantiles is None:
            quantiles = [0.01, 0.05, 0.1, 0.9, 0.95, 0.99]

        result = {}
        for q in quantiles:
            percentile = q * 100
            result[f"q{percentile:04.1f}".replace(".", "_")] = float(
                np.nanquantile(self.discharge, q)
            )

        return result

    def analyze_high_flows(self, threshold_multiplier: float = 2.0) -> dict[str, float]:
        """Analyze high flow characteristics.

        Args:
            threshold_multiplier: Multiplier of median flow for high flow threshold

        Returns:
            Dictionary of high flow metrics
        """
        threshold = threshold_multiplier * self.median_flow
        high_flows = self.discharge[self.discharge > threshold]

        if len(high_flows) == 0:
            return {
                "high_flow_threshold": threshold,
                "high_flow_frequency": 0.0,
                "high_flow_mean": np.nan,
                "high_flow_max": np.nan,
                "high_flow_events": 0,
                "high_flow_avg_duration": 0.0,
                "high_flow_max_duration": 0,
            }

        # Calculate frequency (percentage of time)
        frequency = (len(high_flows) / len(self.discharge)) * 100

        # Calculate event durations
        high_flow_mask = self.discharge > threshold
        event_durations = self._calculate_event_durations(high_flow_mask)

        return {
            "high_flow_threshold": threshold,
            "high_flow_frequency": frequency,
            "high_flow_mean": float(np.mean(high_flows)),
            "high_flow_max": float(np.max(high_flows)),
            "high_flow_events": len(event_durations),
            "high_flow_avg_duration": float(np.mean(event_durations))
            if event_durations
            else 0.0,
            "high_flow_max_duration": int(np.max(event_durations))
            if event_durations
            else 0,
        }

    def analyze_low_flows(self, threshold_multiplier: float = 0.2) -> dict[str, float]:
        """Analyze low flow characteristics.

        Args:
            threshold_multiplier: Multiplier of mean flow for low flow threshold

        Returns:
            Dictionary of low flow metrics
        """
        threshold = threshold_multiplier * self.mean_flow
        low_flows = self.discharge[self.discharge < threshold]

        if len(low_flows) == 0:
            return {
                "low_flow_threshold": threshold,
                "low_flow_frequency": 0.0,
                "low_flow_mean": np.nan,
                "low_flow_min": np.nan,
                "low_flow_events": 0,
                "low_flow_avg_duration": 0.0,
                "low_flow_max_duration": 0,
            }

        # Calculate frequency (percentage of time)
        frequency = (len(low_flows) / len(self.discharge)) * 100

        # Calculate event durations
        low_flow_mask = self.discharge < threshold
        event_durations = self._calculate_event_durations(low_flow_mask)

        return {
            "low_flow_threshold": threshold,
            "low_flow_frequency": frequency,
            "low_flow_mean": float(np.mean(low_flows)),
            "low_flow_min": float(np.min(low_flows)),
            "low_flow_events": len(event_durations),
            "low_flow_avg_duration": float(np.mean(event_durations))
            if event_durations
            else 0.0,
            "low_flow_max_duration": int(np.max(event_durations))
            if event_durations
            else 0,
        }

    def calculate_drought_indices(self) -> dict[str, float]:
        """Calculate drought-related flow indices.

        Returns:
            Dictionary of drought indices
        """
        # IMPORTANT: np.quantile(0.05) = 5th percentile = Q95 (flow exceeded 95% of time)
        # This is the LOW FLOW metric, not to be confused with the 95th percentile
        q95 = np.nanquantile(self.discharge, 0.05)  # Low flow: 5th percentile = Q95
        q90 = np.nanquantile(self.discharge, 0.10)  # 10th percentile

        # Base flow index approximation using Q95
        bfi_approx = q95 / self.mean_flow if self.mean_flow > 0 else np.nan

        # Low flow variability
        low_flows = self.discharge[self.discharge <= q95]
        valid_low_flows = low_flows[~np.isnan(low_flows)]
        if len(valid_low_flows) > 0:
            mean_lf = np.nanmean(valid_low_flows)
            std_lf = np.nanstd(valid_low_flows)
            low_flow_cv = (
                std_lf / mean_lf if mean_lf != 0 and not np.isnan(mean_lf) else np.nan
            )
        else:
            low_flow_cv = np.nan

        return {
            "q95_flow": q95,
            "q90_flow": q90,
            "bfi_approx": bfi_approx,
            "low_flow_cv": low_flow_cv,
            "q95_ratio": q95 / self.median_flow if self.median_flow > 0 else np.nan,
        }

    def calculate_flood_indices(self) -> dict[str, float]:
        """Calculate flood-related flow indices.

        Returns:
            Dictionary of flood indices
        """
        # High flow quantiles
        q05 = np.nanquantile(self.discharge, 0.95)  # 5th percentile exceedance
        q10 = np.nanquantile(self.discharge, 0.90)  # 10th percentile exceedance

        # Flood threshold (commonly 2x mean annual flood)
        annual_maxima = self._get_annual_maxima()
        mean_annual_flood = (
            np.mean(annual_maxima) if len(annual_maxima) > 0 else self.mean_flow
        )

        flood_threshold = 2 * mean_annual_flood
        flood_events = self.discharge[self.discharge > flood_threshold]

        # Flash flood index (rate of rise)
        if isinstance(self.discharge.index, pd.DatetimeIndex):
            daily_change = self.discharge.diff()
            max_daily_rise = np.nanmax(daily_change)
            mean_daily_rise = np.nanmean(daily_change[daily_change > 0])
        else:
            max_daily_rise = np.nan
            mean_daily_rise = np.nan

        return {
            "q05_flow": q05,
            "q10_flow": q10,
            "mean_annual_flood": mean_annual_flood,
            "flood_threshold": flood_threshold,
            "flood_frequency": (len(flood_events) / len(self.discharge)) * 100,
            "max_daily_rise": max_daily_rise,
            "mean_daily_rise": mean_daily_rise,
            "flashiness_index": max_daily_rise / self.mean_flow
            if self.mean_flow > 0 and not np.isnan(max_daily_rise)
            else np.nan,
        }

    def _calculate_event_durations(self, condition_mask: pd.Series) -> list[int]:
        """Calculate durations of consecutive events.

        Args:
            condition_mask: Boolean series indicating event occurrence

        Returns:
            List of event durations
        """
        if not condition_mask.any():
            return []

        # Find start and end points of events
        transitions = condition_mask.astype(int).diff()
        starts = condition_mask.index[transitions == 1]
        ends = condition_mask.index[transitions == -1]

        # Handle edge cases
        if condition_mask.iloc[0]:
            starts = pd.Index([condition_mask.index[0]]).append(starts)
        if condition_mask.iloc[-1]:
            ends = ends.append(pd.Index([condition_mask.index[-1]]))

        # Calculate durations
        durations = []
        for start, end in zip(starts, ends, strict=True):
            if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
                duration = (end - start).days + 1
            else:
                # For non-datetime indices, assume daily data
                duration = 1
            durations.append(duration)

        return durations

    def _get_annual_maxima(self) -> list[float]:
        """Extract annual maximum flows.

        Returns:
            List of annual maximum flows
        """
        if not isinstance(self.discharge.index, pd.DatetimeIndex):
            return [float(np.max(self.discharge))]

        annual_maxima = []
        for year in self.discharge.index.year.unique():
            year_data = self.discharge[self.discharge.index.year == year]
            if len(year_data) > 300:  # Only use complete years
                annual_maxima.append(float(np.max(year_data)))

        return annual_maxima


def calculate_extreme_metrics(
    discharge: pd.Series,
    high_flow_threshold: float = 2.0,
    low_flow_threshold: float = 0.2,
) -> dict[str, float]:
    """Calculate comprehensive extreme flow metrics.

    Args:
        discharge: Discharge time series
        high_flow_threshold: Multiplier of median for high flow threshold
        low_flow_threshold: Multiplier of mean for low flow threshold

    Returns:
        Dictionary of extreme flow metrics
    """
    extremes = FlowExtremes(discharge)

    metrics = {}

    # Flow quantiles
    quantile_metrics = extremes.calculate_flow_quantiles()
    metrics.update(quantile_metrics)

    # High flow analysis
    high_flow_metrics = extremes.analyze_high_flows(high_flow_threshold)
    metrics.update(high_flow_metrics)

    # Low flow analysis
    low_flow_metrics = extremes.analyze_low_flows(low_flow_threshold)
    metrics.update(low_flow_metrics)

    # Drought indices
    drought_metrics = extremes.calculate_drought_indices()
    metrics.update(drought_metrics)

    # Flood indices
    flood_metrics = extremes.calculate_flood_indices()
    metrics.update(flood_metrics)

    return metrics


def calculate_extreme_ratios(discharge: pd.Series) -> dict[str, float]:
    """Calculate ratios between extreme flows for regime characterization.

    Args:
        discharge: Discharge time series

    Returns:
        Dictionary of extreme flow ratios
    """
    q05 = np.nanquantile(discharge, 0.95)  # 5th percentile exceedance
    q50 = np.nanquantile(discharge, 0.50)  # 50th percentile exceedance (median)
    q95 = np.nanquantile(discharge, 0.05)  # 95th percentile exceedance

    return {
        "q05_q95_ratio": q05 / q95 if q95 > 0 else np.nan,
        "q05_q50_ratio": q05 / q50 if q50 > 0 else np.nan,
        "q50_q95_ratio": q50 / q95 if q95 > 0 else np.nan,
        "extreme_range_ratio": (q05 - q95) / q50 if q50 > 0 else np.nan,
    }
