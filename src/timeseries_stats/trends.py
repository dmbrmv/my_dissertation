"""Trend analysis for time series data.

This module provides comprehensive trend analysis including Mann-Kendall tests,
linear regression, and Sen's slope estimation for time series data.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr

from ..utils.logger import setup_logger

logger = setup_logger("trend_analysis", log_file="logs/timeseries.log")

try:
    import pymannkendall as mk

    HAS_PYMANNKENDALL = True
except ImportError:
    HAS_PYMANNKENDALL = False
    logger.warning(
        "pymannkendall not available. Using basic Mann-Kendall implementation."
    )


class TrendAnalysis:
    """Comprehensive trend analysis for time series data."""

    def __init__(
        self, data: pd.Series | xr.DataArray, variable_name: str = "variable"
    ) -> None:
        """Initialize trend analysis.

        Args:
            data: Time series data with datetime index.
            variable_name: Name of the variable for labeling.

        Raises:
            ValueError: If input data is invalid.
        """
        try:
            if isinstance(data, pd.Series):
                if data.empty:
                    raise ValueError("Input data is empty")
                if not isinstance(data.index, pd.DatetimeIndex):
                    raise ValueError("Data must have DatetimeIndex")
                self.data = data
            elif isinstance(data, xr.DataArray):
                if data.size == 0:
                    raise ValueError("Input data is empty")
                self.data = data.to_series()

            self.variable_name = variable_name

        except Exception as e:
            logger.error("Failed to initialize trend analysis: %s", e)
            raise

    def linear_trend(self) -> dict[str, Any]:
        """Calculate linear trend using ordinary least squares regression.

        Returns:
            Dictionary with linear trend statistics.
        """
        try:
            # Create time variable (years from start)
            start_year = self.data.index[0].year
            time_years = (self.data.index.year - start_year).astype(float)

            # Remove missing values
            valid_mask = ~self.data.isna()
            if valid_mask.sum() < 3:
                raise ValueError(
                    "Insufficient valid data points for trend analysis"
                )

            x = time_years[valid_mask]
            y = self.data[valid_mask]

            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x, y
            )

            return {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value**2),
                "p_value": float(p_value),
                "std_error": float(std_err),
                "trend_significant": p_value < 0.05,
                "trend_per_decade": float(slope * 10),
                "trend_direction": "increasing"
                if slope > 0
                else "decreasing"
                if slope < 0
                else "no trend",
                "start_year": int(start_year),
                "end_year": int(self.data.index[-1].year),
                "n_observations": int(valid_mask.sum()),
            }

        except Exception as e:
            logger.error("Failed to calculate linear trend: %s", e)
            raise

    def mann_kendall_test(self, alpha: float = 0.05) -> dict[str, Any]:
        """Perform Mann-Kendall trend test.

        Args:
            alpha: Significance level for the test.

        Returns:
            Dictionary with Mann-Kendall test results.
        """
        try:
            valid_data = self.data.dropna()

            if len(valid_data) < 3:
                raise ValueError("Insufficient data for Mann-Kendall test")

            if HAS_PYMANNKENDALL:
                # Use pymannkendall library if available
                result = mk.original_test(valid_data.values, alpha=alpha)

                return {
                    "trend": result.trend,
                    "h": result.h,  # True if trend is present
                    "p_value": float(result.p),
                    "z_statistic": float(result.z),
                    "tau": float(result.Tau),
                    "s_statistic": float(result.s),
                    "var_s": float(result.var_s),
                    "slope": float(result.slope),
                    "intercept": float(result.intercept),
                    "trend_significant": result.h,
                    "test_used": "pymannkendall",
                }
            else:
                # Use basic implementation
                return self._basic_mann_kendall(valid_data, alpha)

        except Exception as e:
            logger.error("Failed to perform Mann-Kendall test: %s", e)
            raise

    def _basic_mann_kendall(
        self, data: pd.Series, alpha: float
    ) -> dict[str, Any]:
        """Basic Mann-Kendall test implementation."""
        n = len(data)
        s = 0

        # Calculate S statistic
        for i in range(n - 1):
            for j in range(i + 1, n):
                if data.iloc[j] > data.iloc[i]:
                    s += 1
                elif data.iloc[j] < data.iloc[i]:
                    s -= 1

        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18

        # Handle ties (simplified)
        unique_vals, counts = np.unique(data.values, return_counts=True)
        ties = counts[counts > 1]
        if len(ties) > 0:
            tie_correction = np.sum(ties * (ties - 1) * (2 * ties + 5))
            var_s -= tie_correction / 18

        # Calculate Z statistic
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0

        # Calculate p-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # Determine trend
        if p_value < alpha:
            if s > 0:
                trend = "increasing"
            else:
                trend = "decreasing"
            h = True
        else:
            trend = "no trend"
            h = False

        # Calculate Kendall's tau
        tau = s / (n * (n - 1) / 2)

        # Sen's slope (simplified)
        slopes = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                if i != j:
                    slope = (data.iloc[j] - data.iloc[i]) / (j - i)
                    slopes.append(slope)

        sen_slope = np.median(slopes) if slopes else 0

        return {
            "trend": trend,
            "h": h,
            "p_value": float(p_value),
            "z_statistic": float(z),
            "tau": float(tau),
            "s_statistic": float(s),
            "var_s": float(var_s),
            "slope": float(sen_slope),
            "trend_significant": h,
            "test_used": "basic_implementation",
        }

    def seasonal_mann_kendall(self, alpha: float = 0.05) -> dict[str, Any]:
        """Perform seasonal Mann-Kendall test.

        Args:
            alpha: Significance level for the test.

        Returns:
            Dictionary with seasonal Mann-Kendall test results.
        """
        try:
            if not HAS_PYMANNKENDALL:
                logger.warning(
                    "Seasonal Mann-Kendall requires pymannkendall library"
                )
                return self._seasonal_mk_basic(alpha)

            valid_data = self.data.dropna()

            if len(valid_data) < 12:
                raise ValueError(
                    "Insufficient data for seasonal Mann-Kendall test"
                )

            result = mk.seasonal_test(
                valid_data.values, period=12, alpha=alpha
            )

            return {
                "trend": result.trend,
                "h": result.h,
                "p_value": float(result.p),
                "z_statistic": float(result.z),
                "tau": float(result.Tau),
                "s_statistic": float(result.s),
                "var_s": float(result.var_s),
                "slope": float(result.slope),
                "trend_significant": result.h,
                "test_used": "seasonal_mann_kendall",
            }

        except Exception as e:
            logger.error("Failed to perform seasonal Mann-Kendall test: %s", e)
            raise

    def _seasonal_mk_basic(self, alpha: float) -> dict[str, Any]:
        """Basic seasonal Mann-Kendall implementation."""
        # Group by month and perform MK test for each season
        monthly_groups = self.data.groupby(self.data.index.month)

        s_total = 0
        var_s_total = 0

        for month, group_data in monthly_groups:
            if len(group_data) >= 3:
                mk_result = self._basic_mann_kendall(group_data, alpha)
                s_total += mk_result["s_statistic"]
                var_s_total += mk_result["var_s"]

        # Calculate seasonal statistics
        if var_s_total > 0:
            if s_total > 0:
                z = (s_total - 1) / np.sqrt(var_s_total)
            elif s_total < 0:
                z = (s_total + 1) / np.sqrt(var_s_total)
            else:
                z = 0

            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            h = p_value < alpha
            trend = (
                "increasing"
                if s_total > 0 and h
                else "decreasing"
                if s_total < 0 and h
                else "no trend"
            )
        else:
            z = 0
            p_value = 1.0
            h = False
            trend = "no trend"

        return {
            "trend": trend,
            "h": h,
            "p_value": float(p_value),
            "z_statistic": float(z),
            "s_statistic": float(s_total),
            "var_s": float(var_s_total),
            "trend_significant": h,
            "test_used": "basic_seasonal",
        }

    def piecewise_trends(
        self, breakpoint_years: list[int] | None = None
    ) -> dict[str, Any]:
        """Calculate piecewise linear trends.

        Args:
            breakpoint_years: List of years where breaks should occur.
                             If None, attempts to detect breakpoints automatically.

        Returns:
            Dictionary with piecewise trend results.
        """
        try:
            if breakpoint_years is None:
                # Simple breakpoint detection using change in variance
                breakpoint_years = self._detect_simple_breakpoints()

            if not breakpoint_years:
                # No breakpoints detected, return single trend
                return {"single_trend": self.linear_trend(), "breakpoints": []}

            # Add start and end years
            years = (
                [self.data.index[0].year]
                + sorted(breakpoint_years)
                + [self.data.index[-1].year]
            )
            segments = []

            for i in range(len(years) - 1):
                start_year, end_year = years[i], years[i + 1]
                mask = (self.data.index.year >= start_year) & (
                    self.data.index.year <= end_year
                )
                segment_data = self.data[mask]

                if len(segment_data) >= 3:
                    segment_analyzer = TrendAnalysis(
                        segment_data, f"{self.variable_name}_segment_{i + 1}"
                    )
                    segment_trend = segment_analyzer.linear_trend()
                    segment_trend["start_year"] = start_year
                    segment_trend["end_year"] = end_year
                    segments.append(segment_trend)

            return {
                "breakpoints": breakpoint_years,
                "segments": segments,
                "n_segments": len(segments),
            }

        except Exception as e:
            logger.error("Failed to calculate piecewise trends: %s", e)
            raise

    def _detect_simple_breakpoints(self) -> list[int]:
        """Simple breakpoint detection using rolling variance."""
        try:
            # Calculate rolling variance
            window = max(5, len(self.data) // 10)
            rolling_var = self.data.rolling(window=window, center=True).var()

            # Find points where variance changes significantly
            var_diff = rolling_var.diff().abs()
            threshold = var_diff.quantile(0.9)

            breakpoint_candidates = var_diff[var_diff > threshold].index

            # Convert to years and remove duplicates
            breakpoint_years = sorted(
                set(date.year for date in breakpoint_candidates)
            )

            # Remove years too close to start/end
            start_year = self.data.index[0].year
            end_year = self.data.index[-1].year

            breakpoint_years = [
                year
                for year in breakpoint_years
                if year > start_year + 2 and year < end_year - 2
            ]

            return breakpoint_years[:3]  # Limit to 3 breakpoints

        except Exception as e:
            logger.warning("Failed to detect breakpoints: %s", e)
            return []

    def annual_trends(self) -> pd.DataFrame:
        """Calculate trends for annual values.

        Returns:
            DataFrame with annual trend statistics.
        """
        try:
            # Calculate annual means
            annual_data = self.data.resample("YE").mean()

            if len(annual_data) < 3:
                raise ValueError("Insufficient annual data for trend analysis")

            # Create analyzer for annual data
            annual_analyzer = TrendAnalysis(
                annual_data, f"{self.variable_name}_annual"
            )

            # Calculate trends
            linear_trend = annual_analyzer.linear_trend()
            mk_trend = annual_analyzer.mann_kendall_test()

            # Combine results into DataFrame
            trend_stats = {
                "linear_slope": linear_trend["slope"],
                "linear_p_value": linear_trend["p_value"],
                "linear_r_squared": linear_trend["r_squared"],
                "mk_trend": mk_trend["trend"],
                "mk_p_value": mk_trend["p_value"],
                "mk_tau": mk_trend["tau"],
                "mk_slope": mk_trend["slope"],
                "trend_significant": linear_trend["trend_significant"]
                or mk_trend["trend_significant"],
            }

            return pd.DataFrame([trend_stats], index=["annual"])

        except Exception as e:
            logger.error("Failed to calculate annual trends: %s", e)
            raise


def analyze_trends(
    data: pd.Series | xr.DataArray,
    variable_name: str = "variable",
    include_seasonal: bool = True,
    include_piecewise: bool = False,
    include_annual: bool = True,
    alpha: float = 0.05,
    breakpoint_years: list[int] | None = None,
) -> dict[str, Any]:
    """Perform comprehensive trend analysis.

    Args:
        data: Time series data with datetime index.
        variable_name: Name of the variable for labeling.
        include_seasonal: Whether to perform seasonal Mann-Kendall test.
        include_piecewise: Whether to perform piecewise trend analysis.
        include_annual: Whether to analyze annual trends.
        alpha: Significance level for statistical tests.
        breakpoint_years: Predefined breakpoint years for piecewise analysis.

    Returns:
        Dictionary containing all trend analysis results.
    """
    try:
        analyzer = TrendAnalysis(data, variable_name)

        results = {
            "linear_trend": analyzer.linear_trend(),
            "mann_kendall": analyzer.mann_kendall_test(alpha),
        }

        if include_seasonal:
            try:
                results["seasonal_mann_kendall"] = (
                    analyzer.seasonal_mann_kendall(alpha)
                )
            except Exception as e:
                logger.warning(
                    "Failed to calculate seasonal Mann-Kendall: %s", e
                )

        if include_piecewise:
            try:
                results["piecewise_trends"] = analyzer.piecewise_trends(
                    breakpoint_years
                )
            except Exception as e:
                logger.warning("Failed to calculate piecewise trends: %s", e)

        if include_annual:
            try:
                results["annual_trends"] = analyzer.annual_trends()
            except Exception as e:
                logger.warning("Failed to calculate annual trends: %s", e)

        return results

    except Exception as e:
        logger.error("Failed to perform comprehensive trend analysis: %s", e)
        raise
