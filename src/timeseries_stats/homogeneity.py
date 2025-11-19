"""Homogeneity testing for time series data.

This module provides tests for detecting inhomogeneities and change points
in time series data, essential for climate data quality assessment.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr

from ..utils.logger import setup_logger

logger = setup_logger("homogeneity_tests", log_file="logs/timeseries.log")


class HomogeneityTests:
    """Statistical tests for time series homogeneity."""

    def __init__(self, data: pd.Series | xr.DataArray, variable_name: str = "variable") -> None:
        """Initialize homogeneity testing.

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
                self.data = data
            elif isinstance(data, xr.DataArray):
                if data.size == 0:
                    raise ValueError("Input data is empty")
                self.data = data.to_series()
            else:
                raise ValueError("Data must be pandas Series or xarray DataArray")

            self.variable_name = variable_name

        except Exception as e:
            logger.error("Failed to initialize homogeneity tests: %s", e)
            raise

    def pettitt_test(self, alpha: float = 0.05) -> dict[str, Any]:
        """Perform Pettitt test for change point detection.

        Args:
            alpha: Significance level for the test.

        Returns:
            Dictionary with Pettitt test results.
        """
        try:
            data = self.data.dropna()
            n = len(data)

            if n < 10:
                raise ValueError("Insufficient data for Pettitt test")

            # Calculate Ut statistic for each potential change point
            ut_values = []
            for t in range(1, n):
                ut = 0
                for i in range(t):
                    for j in range(t, n):
                        if data.iloc[i] > data.iloc[j]:
                            ut += 1
                        elif data.iloc[i] < data.iloc[j]:
                            ut -= 1
                ut_values.append(abs(ut))

            # Find maximum Ut and its position
            kt = max(ut_values)
            change_point_idx = ut_values.index(kt)

            # Calculate p-value (approximation)
            p_value = 2 * np.exp(-6 * kt**2 / (n**3 + n**2))

            return {
                "change_point_detected": p_value < alpha,
                "p_value": float(p_value),
                "kt_statistic": float(kt),
                "change_point_index": change_point_idx + 1,
                "change_point_date": str(data.index[change_point_idx])
                if change_point_idx < len(data.index)
                else None,
                "test_statistic": float(kt),
                "critical_value": self._pettitt_critical_value(n, alpha),
                "n_observations": n,
            }

        except Exception as e:
            logger.error("Failed to perform Pettitt test: %s", e)
            raise

    def _pettitt_critical_value(self, n: int, alpha: float) -> float:
        """Calculate critical value for Pettitt test (approximation)."""
        if alpha == 0.05:
            return 1.36 * np.sqrt(n)
        elif alpha == 0.01:
            return 1.63 * np.sqrt(n)
        else:
            # Rough approximation
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            return z_alpha * np.sqrt(n / 2)

    def buishand_test(self, alpha: float = 0.05) -> dict[str, Any]:
        """Perform Buishand range test for homogeneity.

        Args:
            alpha: Significance level for the test.

        Returns:
            Dictionary with Buishand test results.
        """
        try:
            data = self.data.dropna()
            n = len(data)

            if n < 10:
                raise ValueError("Insufficient data for Buishand test")

            # Center the data
            mean_val = data.mean()
            centered_data = data - mean_val

            # Calculate cumulative deviations
            cumsum = centered_data.cumsum()

            # Calculate rescaled adjusted range
            range_val = cumsum.max() - cumsum.min()
            std_val = data.std()

            if std_val == 0:
                raise ValueError("Data has zero variance")

            # Rescaled range statistic
            q_stat = range_val / (std_val * np.sqrt(n))

            # Critical values (approximation)
            if n >= 50:
                critical_val = 1.747 if alpha == 0.05 else 2.05 if alpha == 0.01 else 1.96
            else:
                critical_val = 1.4 + 0.3 * np.log10(n) if alpha == 0.05 else 1.6 + 0.3 * np.log10(n)

            return {
                "homogeneous": q_stat < critical_val,
                "q_statistic": float(q_stat),
                "critical_value": float(critical_val),
                "p_value_approx": float(1 - stats.norm.cdf(q_stat)),
                "range": float(range_val),
                "standardized_range": float(q_stat),
                "n_observations": n,
            }

        except Exception as e:
            logger.error("Failed to perform Buishand test: %s", e)
            raise

    def standard_normal_homogeneity_test(self, alpha: float = 0.05) -> dict[str, Any]:
        """Perform Standard Normal Homogeneity Test (SNHT).

        Args:
            alpha: Significance level for the test.

        Returns:
            Dictionary with SNHT test results.
        """
        try:
            data = self.data.dropna()
            n = len(data)

            if n < 10:
                raise ValueError("Insufficient data for SNHT")

            # Standardize the data
            z_scores = (data - data.mean()) / data.std()

            # Calculate SNHT statistic for each potential change point
            t_values = []
            for i in range(1, n):
                # Before change point
                z1 = z_scores.iloc[:i]
                # After change point
                z2 = z_scores.iloc[i:]

                # Calculate means
                mean1 = z1.mean() if len(z1) > 0 else 0
                mean2 = z2.mean() if len(z2) > 0 else 0

                # SNHT statistic
                t_val = i * mean1**2 + (n - i) * mean2**2
                t_values.append(t_val)

            # Find maximum T and its position
            t_max = max(t_values)
            change_point_idx = t_values.index(t_max)

            # Critical value (approximation)
            critical_val = 8.45 if alpha == 0.05 else 10.5 if alpha == 0.01 else 9.0

            return {
                "change_point_detected": t_max > critical_val,
                "t_statistic": float(t_max),
                "critical_value": float(critical_val),
                "change_point_index": change_point_idx + 1,
                "change_point_date": str(data.index[change_point_idx])
                if change_point_idx < len(data.index)
                else None,
                "homogeneous": t_max <= critical_val,
                "n_observations": n,
            }

        except Exception as e:
            logger.error("Failed to perform SNHT: %s", e)
            raise

    def von_neumann_test(self, alpha: float = 0.05) -> dict[str, Any]:
        """Perform von Neumann ratio test for randomness.

        Args:
            alpha: Significance level for the test.

        Returns:
            Dictionary with von Neumann test results.
        """
        try:
            data = self.data.dropna()
            n = len(data)

            if n < 10:
                raise ValueError("Insufficient data for von Neumann test")

            # Calculate von Neumann ratio
            mean_val = data.mean()
            numerator = sum((data.iloc[i] - data.iloc[i - 1]) ** 2 for i in range(1, n))
            denominator = sum((data.iloc[i] - mean_val) ** 2 for i in range(n))

            if denominator == 0:
                raise ValueError("Data has zero variance")

            von_neumann_ratio = numerator / denominator

            # Expected value and variance under null hypothesis
            expected_vn = 2.0
            var_vn = (4 * (n - 2)) / ((n + 1) * (n - 1))

            # Standardized statistic
            if var_vn > 0:
                z_stat = (von_neumann_ratio - expected_vn) / np.sqrt(var_vn)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                z_stat = 0
                p_value = 1.0

            return {
                "random": p_value > alpha,
                "von_neumann_ratio": float(von_neumann_ratio),
                "expected_ratio": expected_vn,
                "z_statistic": float(z_stat),
                "p_value": float(p_value),
                "significant": p_value < alpha,
                "n_observations": n,
            }

        except Exception as e:
            logger.error("Failed to perform von Neumann test: %s", e)
            raise

    def runs_test(self, alpha: float = 0.05) -> dict[str, Any]:
        """Perform runs test for randomness.

        Args:
            alpha: Significance level for the test.

        Returns:
            Dictionary with runs test results.
        """
        try:
            data = self.data.dropna()
            n = len(data)

            if n < 10:
                raise ValueError("Insufficient data for runs test")

            # Convert to binary sequence (above/below median)
            median_val = data.median()
            binary_seq = (data > median_val).astype(int)

            # Count runs
            runs = 1
            for i in range(1, n):
                if binary_seq.iloc[i] != binary_seq.iloc[i - 1]:
                    runs += 1

            # Count number of 1s and 0s
            n1 = binary_seq.sum()
            n0 = n - n1

            if n0 == 0 or n1 == 0:
                return {
                    "random": False,
                    "runs": runs,
                    "expected_runs": 0,
                    "z_statistic": 0,
                    "p_value": 0,
                    "insufficient_variation": True,
                }

            # Expected number of runs and variance
            expected_runs = (2 * n0 * n1) / n + 1
            var_runs = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n**2 * (n - 1))

            # Standardized statistic (with continuity correction)
            if var_runs > 0:
                if runs > expected_runs:
                    z_stat = (runs - 0.5 - expected_runs) / np.sqrt(var_runs)
                else:
                    z_stat = (runs + 0.5 - expected_runs) / np.sqrt(var_runs)

                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                z_stat = 0
                p_value = 1.0

            return {
                "random": p_value > alpha,
                "runs": runs,
                "expected_runs": float(expected_runs),
                "z_statistic": float(z_stat),
                "p_value": float(p_value),
                "significant": p_value < alpha,
                "n_above_median": n1,
                "n_below_median": n0,
                "n_observations": n,
            }

        except Exception as e:
            logger.error("Failed to perform runs test: %s", e)
            raise

    def ljung_box_test(self, lags: int = 10, alpha: float = 0.05) -> dict[str, Any]:
        """Perform Ljung-Box test for autocorrelation.

        Args:
            lags: Number of lags to test.
            alpha: Significance level for the test.

        Returns:
            Dictionary with Ljung-Box test results.
        """
        try:
            data = self.data.dropna()
            n = len(data)

            if n < lags + 5:
                raise ValueError(f"Insufficient data for Ljung-Box test with {lags} lags")

            # Calculate autocorrelations
            autocorrs = []
            for k in range(1, lags + 1):
                # Calculate lag-k autocorrelation
                y1 = data.iloc[:-k]
                y2 = data.iloc[k:]
                if len(y1) > 0 and len(y2) > 0:
                    corr = np.corrcoef(y1, y2)[0, 1]
                    autocorrs.append(corr if not np.isnan(corr) else 0)
                else:
                    autocorrs.append(0)

            # Calculate Ljung-Box statistic
            lb_stat = 0
            for k, rho_k in enumerate(autocorrs, 1):
                lb_stat += (rho_k**2) / (n - k)

            lb_stat *= n * (n + 2)

            # Degrees of freedom
            df = lags

            # P-value from chi-square distribution
            p_value = 1 - stats.chi2.cdf(lb_stat, df)

            return {
                "no_autocorrelation": p_value > alpha,
                "lb_statistic": float(lb_stat),
                "p_value": float(p_value),
                "degrees_of_freedom": df,
                "lags_tested": lags,
                "autocorrelations": [float(r) for r in autocorrs],
                "significant": p_value < alpha,
                "n_observations": n,
            }

        except Exception as e:
            logger.error("Failed to perform Ljung-Box test: %s", e)
            raise


def test_homogeneity(
    data: pd.Series | xr.DataArray,
    variable_name: str = "variable",
    include_pettitt: bool = True,
    include_buishand: bool = True,
    include_snht: bool = True,
    include_von_neumann: bool = True,
    include_runs: bool = True,
    include_ljung_box: bool = True,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Perform comprehensive homogeneity testing.

    Args:
        data: Time series data with datetime index.
        variable_name: Name of the variable for labeling.
        include_pettitt: Whether to perform Pettitt test.
        include_buishand: Whether to perform Buishand test.
        include_snht: Whether to perform SNHT.
        include_von_neumann: Whether to perform von Neumann test.
        include_runs: Whether to perform runs test.
        include_ljung_box: Whether to perform Ljung-Box test.
        alpha: Significance level for all tests.

    Returns:
        Dictionary containing all homogeneity test results.
    """
    try:
        tester = HomogeneityTests(data, variable_name)

        results = {}

        if include_pettitt:
            try:
                results["pettitt_test"] = tester.pettitt_test(alpha)
            except Exception as e:
                logger.warning("Failed to perform Pettitt test: %s", e)

        if include_buishand:
            try:
                results["buishand_test"] = tester.buishand_test(alpha)
            except Exception as e:
                logger.warning("Failed to perform Buishand test: %s", e)

        if include_snht:
            try:
                results["snht_test"] = tester.standard_normal_homogeneity_test(alpha)
            except Exception as e:
                logger.warning("Failed to perform SNHT: %s", e)

        if include_von_neumann:
            try:
                results["von_neumann_test"] = tester.von_neumann_test(alpha)
            except Exception as e:
                logger.warning("Failed to perform von Neumann test: %s", e)

        if include_runs:
            try:
                results["runs_test"] = tester.runs_test(alpha)
            except Exception as e:
                logger.warning("Failed to perform runs test: %s", e)

        if include_ljung_box:
            try:
                results["ljung_box_test"] = tester.ljung_box_test(alpha=alpha)
            except Exception as e:
                logger.warning("Failed to perform Ljung-Box test: %s", e)

        # Summary
        inhomogeneity_detected = any(
            [
                results.get("pettitt_test", {}).get("change_point_detected", False),
                not results.get("buishand_test", {}).get("homogeneous", True),
                results.get("snht_test", {}).get("change_point_detected", False),
            ]
        )

        results["summary"] = {
            "inhomogeneity_detected": inhomogeneity_detected,
            "tests_performed": list(results.keys()),
            "alpha_level": alpha,
        }

        return results

    except Exception as e:
        logger.error("Failed to perform comprehensive homogeneity testing: %s", e)
        raise
