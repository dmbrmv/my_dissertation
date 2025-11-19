"""Flow Duration Curve analysis for hydrological characterization.

This module provides tools for constructing and analyzing Flow Duration Curves (FDC),
including calculation of various FDC-derived indices and slope metrics.
"""

import math

import numpy as np
import pandas as pd

from ..utils.logger import setup_logger

logger = setup_logger("flow_duration_curves", log_file="logs/hydrological.log")


class FlowDurationCurve:
    """Flow Duration Curve analysis and metrics calculation.

    This class provides comprehensive FDC analysis including curve construction,
    slope calculations, and various percentile-based indices.
    """

    def __init__(self, discharge: pd.Series):
        """Initialize FDC with discharge data.

        Args:
            discharge: Discharge time series with datetime index
        """
        self.discharge = discharge.dropna()
        self.sorted_discharge = np.sort(self.discharge)[::-1]  # Descending order
        self.exceedance_prob = self._calculate_exceedance_probabilities()

    def _calculate_exceedance_probabilities(self) -> np.ndarray:
        """Calculate exceedance probabilities for FDC.

        Returns:
            Array of exceedance probabilities (0-100%)
        """
        n = len(self.sorted_discharge)
        ranks = np.arange(1, n + 1)
        # Weibull plotting position formula
        return (ranks / (n + 1)) * 100

    def get_percentile_flow(self, percentile: float) -> float:
        """Get flow value at specified exceedance percentile.

        Args:
            percentile: Exceedance percentage (0-100)

        Returns:
            Flow value at specified percentile
        """
        if not 0 <= percentile <= 100:
            raise ValueError("Percentile must be between 0 and 100")

        return float(np.nanpercentile(self.discharge, 100 - percentile))

    def calculate_fdc_slope(self, p1: float = 33, p2: float = 66) -> float:
        """Calculate FDC slope between two percentiles.

        Args:
            p1: Lower percentile (default 33%)
            p2: Upper percentile (default 66%)

        Returns:
            FDC slope value
        """
        q1 = self.get_percentile_flow(p1)
        q2 = self.get_percentile_flow(p2)

        if q1 <= 0 or q2 <= 0:
            return np.nan

        slope = (math.log(q1) - math.log(q2)) / (p2 - p1) * 100
        return slope

    def calculate_flow_percentiles(self, percentiles: list[float] | None = None) -> dict[str, float]:
        """Calculate multiple flow percentiles.

        Args:
            percentiles: List of percentiles to calculate (default: standard set)

        Returns:
            Dictionary of percentile flows
        """
        if percentiles is None:
            percentiles = [5, 10, 20, 30, 50, 70, 80, 90, 95, 99]

        result = {}
        for p in percentiles:
            result[f"q{p:02d}"] = self.get_percentile_flow(p)

        return result

    def calculate_variability_indices(self) -> dict[str, float]:
        """Calculate FDC-based flow variability indices.

        Returns:
            Dictionary of variability indices
        """
        q5 = self.get_percentile_flow(5)
        q50 = self.get_percentile_flow(50)
        q95 = self.get_percentile_flow(95)

        indices = {
            "q5_q95_ratio": q5 / q95 if q95 > 0 else np.nan,
            "q5_q50_ratio": q5 / q50 if q50 > 0 else np.nan,
            "q50_q95_ratio": q50 / q95 if q95 > 0 else np.nan,
            "iqr_ratio": (self.get_percentile_flow(25) - self.get_percentile_flow(75)) / q50
            if q50 > 0
            else np.nan,
        }

        return indices

    def get_curve_data(self) -> pd.DataFrame:
        """Get FDC curve data for plotting.

        Returns:
            DataFrame with exceedance probabilities and flows
        """
        return pd.DataFrame(
            {
                "exceedance_probability": self.exceedance_prob,
                "discharge": self.sorted_discharge,
            }
        )


def calculate_fdc_metrics(
    discharge: pd.Series,
    percentiles: list[float] | None = None,
    slope_percentiles: tuple[float, float] = (33, 66),
) -> dict[str, float]:
    """Calculate comprehensive FDC-based metrics.

    Args:
        discharge: Discharge time series
        percentiles: Percentiles to calculate (default: standard set)
        slope_percentiles: Percentile pair for slope calculation

    Returns:
        Dictionary of FDC metrics
    """
    fdc = FlowDurationCurve(discharge)

    metrics = {}

    # Flow percentiles
    percentile_flows = fdc.calculate_flow_percentiles(percentiles)
    metrics.update(percentile_flows)

    # FDC slope
    metrics["fdc_slope"] = fdc.calculate_fdc_slope(*slope_percentiles)

    # Variability indices
    variability = fdc.calculate_variability_indices()
    metrics.update(variability)

    # Additional standard metrics
    metrics.update(
        {
            "mean_flow": float(np.mean(discharge.dropna())),
            "median_flow": float(np.median(discharge.dropna())),
            "cv_flow": float(np.std(discharge.dropna()) / np.mean(discharge.dropna())),
            "skewness": float(discharge.dropna().skew()),
            "kurtosis": float(discharge.dropna().kurtosis()),
        }
    )

    return metrics


def calculate_flow_regime_classification(discharge: pd.Series) -> dict[str, float]:
    """Classify flow regime based on FDC characteristics.

    Args:
        discharge: Discharge time series

    Returns:
        Dictionary with regime classification
    """
    fdc = FlowDurationCurve(discharge)
    slope = fdc.calculate_fdc_slope()
    cv = np.std(discharge.dropna()) / np.mean(discharge.dropna())

    # Simple classification based on FDC slope and CV
    if slope < -1.5:
        slope_class = "steep"
    elif slope > -0.5:
        slope_class = "flat"
    else:
        slope_class = "moderate"

    if cv < 0.5:
        variability_class = "low"
    elif cv > 1.5:
        variability_class = "high"
    else:
        variability_class = "moderate"

    # Combine classifications
    if slope_class == "steep" and variability_class == "high":
        regime = "flashy"
    elif slope_class == "flat" and variability_class == "low":
        regime = "stable"
    elif slope_class == "moderate":
        regime = "intermediate"
    else:
        regime = f"{variability_class}_{slope_class}"

    return {
        "flow_regime": regime,
        "slope_class": slope_class,
        "variability_class": variability_class,
        "fdc_slope": slope,
        "coefficient_variation": cv,
    }
