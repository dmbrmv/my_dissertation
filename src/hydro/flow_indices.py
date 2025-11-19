"""Comprehensive hydrological indices for flow characterization.

This module provides a unified interface for calculating comprehensive
hydrological indices from discharge time series, combining metrics from
all other modules into standardized indices.
"""

import numpy as np
import pandas as pd

from ..utils.logger import setup_logger
from .base_flow import calculate_bfi
from .flow_duration import calculate_fdc_metrics
from .flow_extremes import calculate_extreme_metrics
from .flow_timing import calculate_timing_metrics
from .flow_variability import calculate_variability_metrics

logger = setup_logger("hydrological_indices", log_file="logs/hydrological.log")


class HydrologicalIndices:
    """Comprehensive hydrological indices calculation.

    This class provides a unified interface for calculating standardized
    hydrological indices that characterize various aspects of flow regime.
    """

    def __init__(self, discharge: pd.Series):
        """Initialize with discharge data.

        Args:
            discharge: Discharge time series with datetime index
        """
        self.discharge = discharge.dropna()

    def calculate_magnitude_indices(self) -> dict[str, float]:
        """Calculate magnitude-related flow indices.

        Returns:
            Dictionary of magnitude indices
        """
        data = self.discharge

        # Basic magnitude statistics
        ma1 = float(np.mean(data))  # Mean annual daily flow
        ma2 = float(np.median(data))  # Median annual daily flow

        # Monthly means
        if isinstance(data.index, pd.DatetimeIndex):
            monthly_means = data.groupby(data.index.month).mean()
            ma3_to_ma14 = {
                f"ma{i + 3:02d}": float(monthly_means.iloc[i])
                for i in range(min(12, len(monthly_means)))
            }
        else:
            ma3_to_ma14 = {}

        # Additional magnitude metrics
        ma15 = float(np.std(data))  # Standard deviation of daily flows
        ma16 = ma15 / ma1 if ma1 > 0 else np.nan  # CV of daily flows

        magnitude_indices = {
            "ma01": ma1,
            "ma02": ma2,
            "ma15": ma15,
            "ma16": ma16,
        }
        magnitude_indices.update(ma3_to_ma14)

        return magnitude_indices

    def calculate_frequency_indices(self) -> dict[str, float]:
        """Calculate frequency-related flow indices.

        Returns:
            Dictionary of frequency indices
        """
        data = self.discharge
        mean_flow = np.mean(data)

        # High flow frequency (flows > 3x median)
        high_threshold = 3 * np.median(data)
        high_flow_count = np.sum(data > high_threshold)
        fh1 = (high_flow_count / len(data)) * 100  # Percentage

        # Low flow frequency (flows < 0.1x mean)
        low_threshold = 0.1 * mean_flow
        low_flow_count = np.sum(data < low_threshold)
        fl1 = (low_flow_count / len(data)) * 100  # Percentage

        # Frequency of high/low pulses
        fh2 = self._count_pulses(data > high_threshold)
        fl2 = self._count_pulses(data < low_threshold)

        return {
            "fh01": fh1,
            "fh02": float(fh2),
            "fl01": fl1,
            "fl02": float(fl2),
        }

    def calculate_duration_indices(self) -> dict[str, float]:
        """Calculate duration-related flow indices.

        Returns:
            Dictionary of duration indices
        """
        data = self.discharge
        mean_flow = np.mean(data)
        median_flow = np.median(data)

        # High flow duration (flows > 3x median)
        high_threshold = 3 * median_flow
        high_durations = self._get_event_durations(data > high_threshold)
        dh1 = float(np.mean(high_durations)) if high_durations else 0.0
        dh2 = float(np.max(high_durations)) if high_durations else 0.0

        # Low flow duration (flows < 0.1x mean)
        low_threshold = 0.1 * mean_flow
        low_durations = self._get_event_durations(data < low_threshold)
        dl1 = float(np.mean(low_durations)) if low_durations else 0.0
        dl2 = float(np.max(low_durations)) if low_durations else 0.0

        return {
            "dh01": dh1,
            "dh02": dh2,
            "dl01": dl1,
            "dl02": dl2,
        }

    def calculate_timing_indices(self) -> dict[str, float]:
        """Calculate timing-related flow indices.

        Returns:
            Dictionary of timing indices
        """
        if not isinstance(self.discharge.index, pd.DatetimeIndex):
            return {}

        data = self.discharge

        # Timing of annual maximum
        annual_max_timing = []
        for year in data.index.year.unique():
            year_data = data[data.index.year == year]
            if len(year_data) > 300:  # Complete year
                max_date = year_data.idxmax()
                annual_max_timing.append(max_date.dayofyear)

        ta1 = float(np.mean(annual_max_timing)) if annual_max_timing else np.nan

        # Timing of annual minimum
        annual_min_timing = []
        for year in data.index.year.unique():
            year_data = data[data.index.year == year]
            if len(year_data) > 300:  # Complete year
                min_date = year_data.idxmin()
                annual_min_timing.append(min_date.dayofyear)

        ta2 = float(np.mean(annual_min_timing)) if annual_min_timing else np.nan

        return {
            "ta01": ta1,
            "ta02": ta2,
        }

    def calculate_rate_of_change_indices(self) -> dict[str, float]:
        """Calculate rate of change indices.

        Returns:
            Dictionary of rate of change indices
        """
        if len(self.discharge) < 2:
            return {}

        # Daily changes
        daily_changes = self.discharge.diff().dropna()
        positive_changes = daily_changes[daily_changes > 0]
        negative_changes = daily_changes[daily_changes < 0]

        # Rate indices
        ra1 = float(np.mean(positive_changes)) if len(positive_changes) > 0 else 0.0
        ra2 = float(np.median(positive_changes)) if len(positive_changes) > 0 else 0.0
        ra3 = float(np.mean(np.abs(negative_changes))) if len(negative_changes) > 0 else 0.0
        ra4 = float(np.median(np.abs(negative_changes))) if len(negative_changes) > 0 else 0.0

        # Number of flow reversals
        ra5 = float(len(daily_changes[daily_changes.diff() != 0]))

        return {
            "ra01": ra1,
            "ra02": ra2,
            "ra03": ra3,
            "ra04": ra4,
            "ra05": ra5,
        }

    def _count_pulses(self, condition_series: pd.Series) -> int:
        """Count number of discrete pulses (events).

        Args:
            condition_series: Boolean series indicating event occurrence

        Returns:
            Number of separate events
        """
        if not condition_series.any():
            return 0

        # Count transitions from False to True
        transitions = condition_series.astype(int).diff()
        return int(np.sum(transitions == 1))

    def _get_event_durations(self, condition_series: pd.Series) -> list[int]:
        """Get durations of consecutive events.

        Args:
            condition_series: Boolean series indicating event occurrence

        Returns:
            List of event durations
        """
        if not condition_series.any():
            return []

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
        for start, end in zip(starts, ends, strict=True):
            if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
                duration = (end - start).days + 1
            else:
                duration = 1  # Default for non-datetime indices
            durations.append(duration)

        return durations


def calculate_comprehensive_metrics(
    discharge: pd.Series,
    include_bfi: bool = True,
    include_all_modules: bool = True,
) -> dict[str, float]:
    """Calculate comprehensive hydrological metrics from all modules.

    Args:
        discharge: Discharge time series
        include_bfi: Whether to calculate BFI (computationally intensive)
        include_all_modules: Whether to include metrics from all modules

    Returns:
        Dictionary of comprehensive hydrological metrics
    """
    metrics = {}

    # Hydrological indices (always included)
    indices_calc = HydrologicalIndices(discharge)

    magnitude_metrics = indices_calc.calculate_magnitude_indices()
    metrics.update({f"magnitude_{k}": v for k, v in magnitude_metrics.items()})

    frequency_metrics = indices_calc.calculate_frequency_indices()
    metrics.update({f"frequency_{k}": v for k, v in frequency_metrics.items()})

    duration_metrics = indices_calc.calculate_duration_indices()
    metrics.update({f"duration_{k}": v for k, v in duration_metrics.items()})

    timing_metrics = indices_calc.calculate_timing_indices()
    metrics.update({f"timing_{k}": v for k, v in timing_metrics.items()})

    rate_metrics = indices_calc.calculate_rate_of_change_indices()
    metrics.update({f"rate_{k}": v for k, v in rate_metrics.items()})

    if include_all_modules:
        # Flow Duration Curve metrics
        fdc_metrics = calculate_fdc_metrics(discharge)
        metrics.update({f"fdc_{k}": v for k, v in fdc_metrics.items()})

        # Flow extremes metrics
        extreme_metrics = calculate_extreme_metrics(discharge)
        metrics.update({f"extreme_{k}": v for k, v in extreme_metrics.items()})

        # Flow timing metrics
        timing_detailed = calculate_timing_metrics(discharge)
        metrics.update({f"timing_detailed_{k}": v for k, v in timing_detailed.items()})

        # Flow variability metrics
        variability_metrics = calculate_variability_metrics(discharge)
        metrics.update({f"variability_{k}": v for k, v in variability_metrics.items()})

    # Base Flow Index (optional due to computational cost)
    if include_bfi:
        try:
            bfi_value = calculate_bfi(discharge)
            metrics["baseflow_bfi"] = bfi_value
        except Exception:
            metrics["baseflow_bfi"] = np.nan

    return metrics


def calculate_regime_classification_metrics(discharge: pd.Series) -> dict[str, str | float]:
    """Calculate metrics for flow regime classification.

    Args:
        discharge: Discharge time series

    Returns:
        Dictionary with regime classification metrics
    """
    # Calculate key metrics for classification
    cv = np.std(discharge) / np.mean(discharge)

    # FDC slope
    q33 = np.nanpercentile(discharge, 100 - 33)
    q66 = np.nanpercentile(discharge, 100 - 66)

    if q33 > 0 and q66 > 0:
        fdc_slope = (np.log(q33) - np.log(q66)) / (66 - 33) * 100
    else:
        fdc_slope = np.nan

    # Baseflow ratio approximation
    q95 = np.nanpercentile(discharge, 100 - 95)
    bf_ratio = q95 / np.mean(discharge) if np.mean(discharge) > 0 else np.nan

    # Simple regime classification
    regime_class = "unclassified"

    if not np.isnan(fdc_slope) and not np.isnan(cv):
        if fdc_slope < -2.0 and cv > 1.0:
            regime_class = "flashy"
        elif fdc_slope > -0.8 and cv < 0.7:
            regime_class = "baseflow_dominated"
        elif -2.0 <= fdc_slope <= -0.8:
            regime_class = "intermediate"
        else:
            regime_class = "variable"

    return {
        "regime_classification": regime_class,
        "coefficient_of_variation": float(cv),
        "fdc_slope": float(fdc_slope) if not np.isnan(fdc_slope) else np.nan,
        "baseflow_ratio_approx": float(bf_ratio) if not np.isnan(bf_ratio) else np.nan,
    }
