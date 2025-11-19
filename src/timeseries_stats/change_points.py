"""Change point detection for time series data.

This module provides various methods for detecting change points in time series,
including statistical and algorithmic approaches.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr

from ..utils.logger import setup_logger

logger = setup_logger("change_point_detection", log_file="logs/timeseries.log")


class ChangePointDetection:
    """Change point detection methods for time series analysis."""

    def __init__(self, data: pd.Series | xr.DataArray, variable_name: str = "variable") -> None:
        """Initialize change point detection.

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
                self.data = data.dropna()
            elif isinstance(data, xr.DataArray):
                if data.size == 0:
                    raise ValueError("Input data is empty")
                self.data = data.to_series().dropna()
            else:
                raise ValueError("Data must be pandas Series or xarray DataArray")

            self.variable_name = variable_name

        except Exception as e:
            logger.error("Failed to initialize change point detection: %s", e)
            raise

    def cusum_change_point(self, threshold: float | None = None) -> dict[str, Any]:
        """Detect change points using CUSUM algorithm.

        Args:
            threshold: Threshold for change point detection. If None, uses data std.

        Returns:
            Dictionary with CUSUM change point results.
        """
        try:
            n = len(self.data)
            if n < 10:
                raise ValueError("Insufficient data for CUSUM analysis")

            # Use threshold or estimate from data
            if threshold is None:
                threshold = 2 * self.data.std()

            # Calculate CUSUM statistics
            mean_val = self.data.mean()
            cusum_pos = np.zeros(n)
            cusum_neg = np.zeros(n)

            change_points = []

            for i in range(1, n):
                # Positive CUSUM (detecting increases)
                cusum_pos[i] = max(0, cusum_pos[i - 1] + (self.data.iloc[i] - mean_val))

                # Negative CUSUM (detecting decreases)
                cusum_neg[i] = min(0, cusum_neg[i - 1] + (self.data.iloc[i] - mean_val))

                # Check for change points
                if abs(cusum_pos[i]) > threshold or abs(cusum_neg[i]) > threshold:
                    change_points.append(
                        {
                            "index": i,
                            "date": str(self.data.index[i]),
                            "cusum_pos": float(cusum_pos[i]),
                            "cusum_neg": float(cusum_neg[i]),
                            "value": float(self.data.iloc[i]),
                        }
                    )

                    # Reset CUSUM after detection
                    cusum_pos[i] = 0
                    cusum_neg[i] = 0

            return {
                "change_points": change_points,
                "n_change_points": len(change_points),
                "cusum_positive": cusum_pos.tolist(),
                "cusum_negative": cusum_neg.tolist(),
                "threshold": float(threshold),
                "mean_value": float(mean_val),
            }

        except Exception as e:
            logger.error("Failed to perform CUSUM change point detection: %s", e)
            raise

    def binary_segmentation(
        self, min_segment_length: int = 10, max_segments: int = 10
    ) -> dict[str, Any]:
        """Detect change points using binary segmentation.

        Args:
            min_segment_length: Minimum length of each segment.
            max_segments: Maximum number of segments to detect.

        Returns:
            Dictionary with binary segmentation results.
        """
        try:
            n = len(self.data)
            if n < 2 * min_segment_length:
                raise ValueError("Data too short for binary segmentation")

            change_points = []
            segments_to_split = [(0, n - 1)]

            while segments_to_split and len(change_points) < max_segments:
                # Find segment with maximum variance change
                best_segment = None
                best_change_point = None
                best_score = 0

                for start, end in segments_to_split:
                    if end - start < 2 * min_segment_length:
                        continue

                    # Find best change point in this segment
                    cp, score = self._find_best_change_point(start, end, min_segment_length)

                    if score > best_score:
                        best_score = score
                        best_change_point = cp
                        best_segment = (start, end)

                if best_change_point is not None and best_segment is not None:
                    change_points.append(
                        {
                            "index": best_change_point,
                            "date": str(self.data.index[best_change_point]),
                            "value": float(self.data.iloc[best_change_point]),
                            "score": float(best_score),
                        }
                    )

                    # Split the segment
                    start, end = best_segment
                    segments_to_split.remove(best_segment)

                    # Add new segments if they're large enough
                    if best_change_point - start >= min_segment_length:
                        segments_to_split.append((start, best_change_point))
                    if end - best_change_point >= min_segment_length:
                        segments_to_split.append((best_change_point, end))
                else:
                    break

            # Sort change points by index
            change_points.sort(key=lambda x: x["index"])

            return {
                "change_points": change_points,
                "n_change_points": len(change_points),
                "segments": self._get_segments(change_points, n),
            }

        except Exception as e:
            logger.error("Failed to perform binary segmentation: %s", e)
            raise

    def _find_best_change_point(self, start: int, end: int, min_length: int) -> tuple[int | None, float]:
        """Find the best change point in a segment."""
        best_cp = None
        best_score = 0

        for cp in range(start + min_length, end - min_length):
            # Calculate variance before and after change point
            before = self.data.iloc[start:cp]
            after = self.data.iloc[cp:end]

            if len(before) > 0 and len(after) > 0:
                var_before = before.var()
                var_after = after.var()
                var_total = self.data.iloc[start:end].var()

                # Score based on variance reduction
                if var_total > 0:
                    score = 1 - (len(before) * var_before + len(after) * var_after) / (
                        (end - start) * var_total
                    )
                    if score > best_score:
                        best_score = score
                        best_cp = cp

        return best_cp, best_score

    def _get_segments(self, change_points: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        """Get segment information from change points."""
        segments = []
        start_indices = [0] + [cp["index"] for cp in change_points] + [n - 1]

        for i in range(len(start_indices) - 1):
            start_idx = start_indices[i]
            end_idx = start_indices[i + 1]

            segment_data = self.data.iloc[start_idx:end_idx]
            segments.append(
                {
                    "start_index": start_idx,
                    "end_index": end_idx,
                    "start_date": str(self.data.index[start_idx]),
                    "end_date": str(self.data.index[end_idx - 1]),
                    "length": end_idx - start_idx,
                    "mean": float(segment_data.mean()),
                    "std": float(segment_data.std()),
                    "min": float(segment_data.min()),
                    "max": float(segment_data.max()),
                }
            )

        return segments

    def variance_change_point(self, window_size: int = 50) -> dict[str, Any]:
        """Detect change points in variance using rolling windows.

        Args:
            window_size: Size of the rolling window for variance calculation.

        Returns:
            Dictionary with variance change point results.
        """
        try:
            n = len(self.data)
            if n < 2 * window_size:
                raise ValueError("Data too short for variance change point detection")

            # Calculate rolling variance
            rolling_var = self.data.rolling(window=window_size, center=True).var()

            # Calculate variance changes
            var_diff = rolling_var.diff().abs()

            # Find significant changes (above 95th percentile)
            threshold = var_diff.quantile(0.95)
            significant_changes = var_diff[var_diff > threshold]

            change_points = []
            for date, change in significant_changes.items():
                if not np.isnan(change):
                    idx = self.data.index.get_loc(date)
                    change_points.append(
                        {
                            "index": idx,
                            "date": str(date),
                            "variance_change": float(change),
                            "variance_before": float(rolling_var.iloc[idx - 1]) if idx > 0 else np.nan,
                            "variance_after": float(rolling_var.iloc[idx]),
                            "value": float(self.data.iloc[idx]),
                        }
                    )

            return {
                "change_points": change_points,
                "n_change_points": len(change_points),
                "rolling_variance": rolling_var,
                "variance_changes": var_diff,
                "threshold": float(threshold),
                "window_size": window_size,
            }

        except Exception as e:
            logger.error("Failed to detect variance change points: %s", e)
            raise

    def mean_shift_detection(self, min_segment_length: int = 20) -> dict[str, Any]:
        """Detect mean shifts in the time series.

        Args:
            min_segment_length: Minimum length of segments for mean shift detection.

        Returns:
            Dictionary with mean shift detection results.
        """
        try:
            n = len(self.data)
            if n < 2 * min_segment_length:
                raise ValueError("Data too short for mean shift detection")

            change_points = []

            # Use t-test to detect significant mean changes
            for i in range(min_segment_length, n - min_segment_length):
                before = self.data.iloc[:i]
                after = self.data.iloc[i:]

                # Perform t-test
                t_stat, p_value = stats.ttest_ind(before, after)

                if p_value < 0.01:  # Significant at 1% level
                    change_points.append(
                        {
                            "index": i,
                            "date": str(self.data.index[i]),
                            "t_statistic": float(t_stat),
                            "p_value": float(p_value),
                            "mean_before": float(before.mean()),
                            "mean_after": float(after.mean()),
                            "mean_difference": float(after.mean() - before.mean()),
                            "value": float(self.data.iloc[i]),
                        }
                    )

            # Remove overlapping detections (keep most significant)
            if len(change_points) > 1:
                filtered_cps = []
                for cp in sorted(change_points, key=lambda x: x["p_value"]):
                    if not any(
                        abs(cp["index"] - fcp["index"]) < min_segment_length for fcp in filtered_cps
                    ):
                        filtered_cps.append(cp)
                change_points = sorted(filtered_cps, key=lambda x: x["index"])

            return {
                "change_points": change_points,
                "n_change_points": len(change_points),
                "min_segment_length": min_segment_length,
            }

        except Exception as e:
            logger.error("Failed to detect mean shifts: %s", e)
            raise


def detect_change_points(
    data: pd.Series | xr.DataArray,
    variable_name: str = "variable",
    methods: list[str] | None = None,
    cusum_threshold: float | None = None,
    min_segment_length: int = 20,
    variance_window: int = 50,
) -> dict[str, Any]:
    """Perform comprehensive change point detection.

    Args:
        data: Time series data with datetime index.
        variable_name: Name of the variable for labeling.
        methods: List of methods to use. Options: 'cusum', 'binary_segmentation',
                'variance', 'mean_shift'. If None, uses all methods.
        cusum_threshold: Threshold for CUSUM method.
        min_segment_length: Minimum segment length for segmentation methods.
        variance_window: Window size for variance change point detection.

    Returns:
        Dictionary containing all change point detection results.
    """
    try:
        if methods is None:
            methods = ["cusum", "binary_segmentation", "variance", "mean_shift"]

        detector = ChangePointDetection(data, variable_name)
        results = {}

        if "cusum" in methods:
            try:
                results["cusum"] = detector.cusum_change_point(cusum_threshold)
            except Exception as e:
                logger.warning("Failed to perform CUSUM change point detection: %s", e)

        if "binary_segmentation" in methods:
            try:
                results["binary_segmentation"] = detector.binary_segmentation(min_segment_length)
            except Exception as e:
                logger.warning("Failed to perform binary segmentation: %s", e)

        if "variance" in methods:
            try:
                results["variance"] = detector.variance_change_point(variance_window)
            except Exception as e:
                logger.warning("Failed to detect variance change points: %s", e)

        if "mean_shift" in methods:
            try:
                results["mean_shift"] = detector.mean_shift_detection(min_segment_length)
            except Exception as e:
                logger.warning("Failed to detect mean shifts: %s", e)

        # Summary of all detected change points
        all_change_points = []
        for method, result in results.items():
            if "change_points" in result:
                for cp in result["change_points"]:
                    cp_info = cp.copy()
                    cp_info["method"] = method
                    all_change_points.append(cp_info)

        # Sort by date
        all_change_points.sort(key=lambda x: x.get("date", ""))

        results["summary"] = {
            "all_change_points": all_change_points,
            "total_change_points": len(all_change_points),
            "methods_used": list(results.keys()),
        }

        return results

    except Exception as e:
        logger.error("Failed to perform comprehensive change point detection: %s", e)
        raise
