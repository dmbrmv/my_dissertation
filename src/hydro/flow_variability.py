"""Flow variability analysis for hydrological characterization.

This module provides tools for analyzing discharge variability at multiple
temporal scales, including daily, seasonal, and inter-annual variability.
"""

import numpy as np
import pandas as pd

from ..utils.logger import setup_logger

logger = setup_logger("flow_variability_analysis", log_file="logs/hydrological.log")


class FlowVariability:
    """Analysis of flow variability at multiple temporal scales.

    This class provides comprehensive variability analysis including
    statistical measures, temporal autocorrelation, and regime stability metrics.
    """

    def __init__(self, discharge: pd.Series):
        """Initialize flow variability analysis.

        Args:
            discharge: Discharge time series with datetime index
        """
        self.discharge = discharge.dropna()
        self.daily_data = self._prepare_daily_data()

    def _prepare_daily_data(self) -> pd.DataFrame:
        """Prepare daily data with temporal attributes.

        Returns:
            DataFrame with discharge and temporal information
        """
        df = pd.DataFrame({"discharge": self.discharge})

        if isinstance(self.discharge.index, pd.DatetimeIndex):
            df["year"] = self.discharge.index.year
            df["month"] = self.discharge.index.month
            df["day_of_year"] = self.discharge.index.dayofyear

        return df

    def calculate_basic_statistics(self) -> dict[str, float]:
        """Calculate basic statistical measures of variability.

        Returns:
            Dictionary of basic statistical metrics
        """
        data = self.discharge

        return {
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
            "variance": float(np.var(data)),
            "cv": float(np.std(data) / np.mean(data)),
            "skewness": float(data.skew()) if hasattr(data, "skew") else np.nan,
            "kurtosis": float(data.kurtosis()) if hasattr(data, "kurtosis") else np.nan,
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "range": float(np.max(data) - np.min(data)),
            "iqr": float(np.quantile(data, 0.75) - np.quantile(data, 0.25)),
        }

    def calculate_temporal_variability(self) -> dict[str, float]:
        """Calculate variability metrics at different temporal scales.

        Returns:
            Dictionary of temporal variability metrics
        """
        metrics = {}

        # Daily variability (change between consecutive days)
        if len(self.discharge) > 1:
            daily_changes = self.discharge.diff().dropna()
            metrics["daily_change_mean"] = float(np.mean(daily_changes))
            metrics["daily_change_std"] = float(np.std(daily_changes))
            metrics["daily_change_cv"] = float(
                np.std(daily_changes) / np.abs(np.mean(daily_changes))
                if np.mean(daily_changes) != 0
                else np.nan
            )

        # Weekly variability (if enough data)
        if len(self.discharge) > 7:
            weekly_means = self.discharge.rolling(window=7).mean().dropna()
            weekly_changes = weekly_means.diff().dropna()
            if len(weekly_changes) > 0:
                metrics["weekly_change_std"] = float(np.std(weekly_changes))
                metrics["weekly_cv"] = float(np.std(weekly_means) / np.mean(weekly_means))

        # Monthly variability (if datetime index available)
        if isinstance(self.discharge.index, pd.DatetimeIndex) and len(self.discharge) > 30:
            monthly_means = self.discharge.resample("ME").mean().dropna()
            if len(monthly_means) > 1:
                metrics["monthly_cv"] = float(np.std(monthly_means) / np.mean(monthly_means))

        return metrics

    def calculate_autocorrelation_metrics(self) -> dict[str, float]:
        """Calculate autocorrelation-based variability metrics.

        Returns:
            Dictionary of autocorrelation metrics
        """
        metrics = {}

        # Lag-1 autocorrelation
        if len(self.discharge) > 1:
            lag1_corr = self.discharge.autocorr(lag=1)
            metrics["lag1_autocorr"] = float(lag1_corr) if not np.isnan(lag1_corr) else 0.0

        # Higher order autocorrelations
        for lag in [7, 30, 365]:
            if len(self.discharge) > lag:
                lag_corr = self.discharge.autocorr(lag=lag)
                if not np.isnan(lag_corr):
                    metrics[f"lag{lag}_autocorr"] = float(lag_corr)

        return metrics

    def calculate_flashiness_index(self) -> dict[str, float]:
        """Calculate flashiness index and related metrics.

        Returns:
            Dictionary of flashiness metrics
        """
        if len(self.discharge) < 2:
            return {"flashiness_index": np.nan}

        # Richards-Baker Flashiness Index
        daily_changes = np.abs(self.discharge.diff().dropna())
        total_flow = np.sum(self.discharge[1:])  # Exclude first value due to diff

        flashiness = np.sum(daily_changes) / total_flow if total_flow > 0 else np.nan

        # Additional flashiness metrics
        mean_abs_change = np.mean(daily_changes)
        max_abs_change = np.max(daily_changes)

        return {
            "flashiness_index": float(flashiness) if not np.isnan(flashiness) else np.nan,
            "mean_abs_daily_change": float(mean_abs_change),
            "max_abs_daily_change": float(max_abs_change),
            "change_variability": float(np.std(daily_changes) / np.mean(daily_changes))
            if np.mean(daily_changes) > 0
            else np.nan,
        }

    def calculate_seasonal_variability(self) -> dict[str, float]:
        """Calculate seasonal variability metrics.

        Returns:
            Dictionary of seasonal variability metrics
        """
        if not isinstance(self.discharge.index, pd.DatetimeIndex):
            return {}

        # Group by month and calculate statistics
        monthly_groups = self.discharge.groupby(self.discharge.index.month)
        monthly_means = monthly_groups.mean()
        monthly_stds = monthly_groups.std()

        # Seasonal variability measures
        seasonal_cv = np.std(monthly_means) / np.mean(monthly_means)
        max_monthly_mean = np.max(monthly_means)
        min_monthly_mean = np.min(monthly_means)
        seasonal_range = max_monthly_mean - min_monthly_mean

        # Within-month variability
        avg_within_month_cv = np.mean(monthly_stds / monthly_means)

        return {
            "seasonal_cv": float(seasonal_cv),
            "seasonal_range": float(seasonal_range),
            "max_monthly_mean": float(max_monthly_mean),
            "min_monthly_mean": float(min_monthly_mean),
            "seasonal_ratio": float(max_monthly_mean / min_monthly_mean)
            if min_monthly_mean > 0
            else np.nan,
            "avg_within_month_cv": float(avg_within_month_cv)
            if not np.isnan(avg_within_month_cv)
            else np.nan,
        }

    def calculate_interannual_variability(self) -> dict[str, float]:
        """Calculate inter-annual variability metrics.

        Returns:
            Dictionary of inter-annual variability metrics
        """
        if not isinstance(self.discharge.index, pd.DatetimeIndex):
            return {}

        # Group by year and calculate annual statistics
        annual_groups = self.discharge.groupby(self.discharge.index.year)
        annual_means = annual_groups.mean()
        annual_maxs = annual_groups.max()
        annual_mins = annual_groups.min()

        if len(annual_means) < 2:
            return {}

        # Inter-annual variability measures
        interannual_cv = np.std(annual_means) / np.mean(annual_means)
        max_annual_ratio = np.max(annual_means) / np.min(annual_means)

        # Variability in extremes
        annual_max_cv = np.std(annual_maxs) / np.mean(annual_maxs)
        annual_min_cv = np.std(annual_mins) / np.mean(annual_mins)

        return {
            "interannual_cv": float(interannual_cv),
            "max_annual_ratio": float(max_annual_ratio),
            "annual_max_cv": float(annual_max_cv),
            "annual_min_cv": float(annual_min_cv),
            "annual_mean_trend": self._calculate_trend(annual_means),
        }

    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate linear trend using least squares.

        Args:
            series: Time series data

        Returns:
            Trend slope per year
        """
        if len(series) < 3:
            return np.nan

        x = np.arange(len(series))
        y = series.values

        # Simple linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return float(slope)


def calculate_variability_metrics(discharge: pd.Series) -> dict[str, float]:
    """Calculate comprehensive flow variability metrics.

    Args:
        discharge: Discharge time series

    Returns:
        Dictionary of variability metrics
    """
    variability = FlowVariability(discharge)

    metrics = {}

    # Basic statistics
    basic_stats = variability.calculate_basic_statistics()
    metrics.update(basic_stats)

    # Temporal variability
    temporal_vars = variability.calculate_temporal_variability()
    metrics.update(temporal_vars)

    # Autocorrelation
    autocorr_metrics = variability.calculate_autocorrelation_metrics()
    metrics.update(autocorr_metrics)

    # Flashiness
    flashiness_metrics = variability.calculate_flashiness_index()
    metrics.update(flashiness_metrics)

    # Seasonal variability
    seasonal_metrics = variability.calculate_seasonal_variability()
    metrics.update(seasonal_metrics)

    # Inter-annual variability
    interannual_metrics = variability.calculate_interannual_variability()
    metrics.update(interannual_metrics)

    return metrics


def calculate_flow_stability_index(discharge: pd.Series) -> dict[str, float]:
    """Calculate overall flow stability index.

    Args:
        discharge: Discharge time series

    Returns:
        Dictionary with stability indices
    """
    variability = FlowVariability(discharge)

    # Component metrics for stability
    cv = np.std(discharge) / np.mean(discharge)

    autocorr_metrics = variability.calculate_autocorrelation_metrics()
    lag1_autocorr = autocorr_metrics.get("lag1_autocorr", 0.0)

    flashiness_metrics = variability.calculate_flashiness_index()
    flashiness = flashiness_metrics.get("flashiness_index", np.nan)

    # Composite stability index (lower values = more stable)
    # Normalize components to 0-1 scale
    cv_norm = min(cv / 2.0, 1.0)  # CV > 2 is very unstable
    lag1_stability = max(0, lag1_autocorr)  # Higher autocorr = more stable
    flashiness_norm = min(flashiness / 0.5, 1.0) if not np.isnan(flashiness) else 0.5

    stability_index = 1.0 - (cv_norm * 0.4 + (1 - lag1_stability) * 0.3 + flashiness_norm * 0.3)

    return {
        "stability_index": float(max(0, min(1, stability_index))),
        "cv_component": float(cv_norm),
        "autocorr_component": float(lag1_stability),
        "flashiness_component": float(flashiness_norm),
    }
