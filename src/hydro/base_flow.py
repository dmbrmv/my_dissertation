"""Base flow separation methods for hydrological analysis.

This module provides digital filter methods for separating base flow from
total discharge, including the Lyne-Hollick filter and related techniques.
"""

import random

import numba
import numpy as np
import pandas as pd

from ..utils.logger import setup_logger

logger = setup_logger("base_flow_separation", log_file="logs/hydrological.log")


@numba.jit(nopython=True)
def _first_pass(discharge: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """First pass of the Lyne-Hollick filter (forward direction).

    Args:
        discharge: Discharge time series array
        alpha: Filter parameter (0.9-0.98)

    Returns:
        Tuple of quick flow and base flow arrays
    """
    q_f_1 = np.full_like(discharge, np.nan, dtype=np.float64)
    q_b_1 = np.full_like(discharge, np.nan, dtype=np.float64)

    q_f_1[0] = discharge[0]

    for j in range(len(discharge) - 1):
        q_f_1[j + 1] = alpha * q_f_1[j] + 0.5 * (1 + alpha) * (discharge[j + 1] - discharge[j])

    for j in range(len(discharge)):
        if q_f_1[j] < 0:
            q_b_1[j] = discharge[j]
        else:
            q_b_1[j] = discharge[j] - q_f_1[j]

    return q_f_1, q_b_1


@numba.jit(nopython=True)
def _backward_pass(
    quick_flow: np.ndarray, base_flow: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Backward pass of the Lyne-Hollick filter.

    Args:
        quick_flow: Quick flow from previous pass
        base_flow: Base flow from previous pass
        alpha: Filter parameter

    Returns:
        Tuple of updated quick flow and base flow arrays
    """
    q_f_2 = np.full_like(base_flow, np.nan, dtype=np.float64)
    q_b_2 = np.full_like(base_flow, np.nan, dtype=np.float64)

    q_f_2[-1] = base_flow[-1]

    for j in range(len(base_flow) - 2, -1, -1):
        q_f_2[j] = alpha * q_f_2[j + 1] + 0.5 * (1 + alpha) * (base_flow[j] - base_flow[j + 1])

    for j in range(len(base_flow) - 1, -1, -1):
        if q_f_2[j] < 0:
            q_b_2[j] = base_flow[j]
        else:
            q_b_2[j] = base_flow[j] - q_f_2[j]

    return q_f_2, q_b_2


@numba.jit(nopython=True)
def _forward_pass(
    quick_flow: np.ndarray, base_flow: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass of the Lyne-Hollick filter.

    Args:
        quick_flow: Quick flow from previous pass
        base_flow: Base flow from previous pass
        alpha: Filter parameter

    Returns:
        Tuple of updated quick flow and base flow arrays
    """
    q_f_3 = np.full_like(base_flow, np.nan, dtype=np.float64)
    q_b_3 = np.full_like(base_flow, np.nan, dtype=np.float64)

    q_f_3[0] = base_flow[0]

    for j in range(len(base_flow) - 1):
        q_f_3[j + 1] = alpha * q_f_3[j] + 0.5 * (1 + alpha) * (base_flow[j + 1] - base_flow[j])

    for j in range(len(base_flow)):
        if q_f_3[j] < 0:
            q_b_3[j] = base_flow[j]
        else:
            q_b_3[j] = base_flow[j] - q_f_3[j]

    return q_f_3, q_b_3


@numba.jit(nopython=True)
def _reflect_series(discharge: np.ndarray, reflect_points: int) -> np.ndarray:
    """Reflect discharge series to handle boundary effects.

    Args:
        discharge: Original discharge array
        reflect_points: Number of points to reflect at each end

    Returns:
        Reflected discharge array
    """
    if len(discharge) - 1 <= reflect_points:
        return discharge.copy()

    q_reflect = np.full(len(discharge) + 2 * reflect_points, np.nan, dtype=np.float64)

    # Reflect beginning
    q_reflect[:reflect_points] = discharge[reflect_points:0:-1]
    # Original series
    q_reflect[reflect_points : reflect_points + len(discharge)] = discharge
    # Reflect end
    q_reflect[reflect_points + len(discharge) :] = discharge[
        len(discharge) - 2 : len(discharge) - reflect_points - 2 : -1
    ]

    return q_reflect


@numba.jit(nopython=True)
def _single_bfi_calculation(
    discharge: np.ndarray, alpha: float, passes: int, reflect_points: int
) -> tuple[float, np.ndarray]:
    """Calculate BFI for a single alpha value.

    Args:
        discharge: Discharge time series
        alpha: Filter parameter
        passes: Number of filter passes
        reflect_points: Points to reflect for boundary handling

    Returns:
        Tuple of BFI value and base flow array
    """
    q_reflect = _reflect_series(discharge, reflect_points)
    quick_flow, base_flow = _first_pass(q_reflect, alpha)

    n_pass = int(0.5 * (passes - 1))

    for _ in range(n_pass):
        quick_flow, base_flow = _backward_pass(quick_flow, base_flow, alpha)
        quick_flow, base_flow = _forward_pass(quick_flow, base_flow, alpha)

    # Extract original length and ensure non-negative values
    if len(discharge) - 1 > reflect_points:
        base_flow_final = base_flow[reflect_points : len(base_flow) - reflect_points]
    else:
        base_flow_final = base_flow

    base_flow_final = np.maximum(base_flow_final, 0.0)

    # Calculate BFI
    mean_discharge = np.mean(discharge)
    if mean_discharge == 0:
        bfi = 0.0
    else:
        bfi = np.mean(base_flow_final) / mean_discharge

    return bfi, base_flow_final


@numba.jit(nopython=True)
def _bfi_ensemble(
    discharge: np.ndarray, passes: int, reflect_points: int, n_samples: int = 1000
) -> tuple[float, np.ndarray]:
    """Calculate BFI using ensemble of alpha values.

    Args:
        discharge: Discharge time series
        passes: Number of filter passes
        reflect_points: Points to reflect
        n_samples: Number of alpha samples to use

    Returns:
        Tuple of mean BFI and mean base flow array
    """
    # Set seed for reproducibility
    random.seed(1996)
    alpha_values = [random.uniform(0.9, 0.98) for _ in range(n_samples)]

    bfi_values = np.empty(n_samples, dtype=np.float64)
    base_flow_arrays = []

    for i, alpha in enumerate(alpha_values):
        bfi, base_flow = _single_bfi_calculation(discharge, alpha, passes, reflect_points)
        bfi_values[i] = bfi
        base_flow_arrays.append(base_flow)

    # Calculate ensemble means
    bfi_mean = np.mean(bfi_values)

    # Mean base flow hydrograph
    base_flow_mean = np.zeros(len(base_flow_arrays[0]), dtype=np.float64)
    for base_flow in base_flow_arrays:
        base_flow_mean += base_flow
    base_flow_mean /= len(base_flow_arrays)

    return bfi_mean, base_flow_mean


class BaseFlowSeparation:
    """Digital filter base flow separation using Lyne-Hollick method.

    This class provides methods for separating base flow from total discharge
    using digital filtering techniques with various parameter options.
    """

    def __init__(
        self,
        alpha: float | None = None,
        passes: int = 3,
        reflect_points: int = 30,
        ensemble_size: int = 1000,
    ):
        """Initialize base flow separation parameters.

        Args:
            alpha: Filter parameter (0.9-0.98). If None, uses ensemble approach
            passes: Number of filter passes (typically 1 or 3)
            reflect_points: Number of points to reflect at boundaries
            ensemble_size: Number of alpha values for ensemble calculation
        """
        self.alpha = alpha
        self.passes = passes
        self.reflect_points = reflect_points
        self.ensemble_size = ensemble_size

        if alpha is not None and not (0.9 <= alpha <= 0.98):
            raise ValueError("Alpha parameter should be between 0.9 and 0.98")

    def separate(self, discharge: pd.Series) -> tuple[float, pd.Series]:
        """Perform base flow separation on discharge series.

        Args:
            discharge: Discharge time series with datetime index

        Returns:
            Tuple of (BFI value, base flow series)
        """
        discharge_values = discharge.dropna().values

        if len(discharge_values) == 0:
            raise ValueError("Discharge series contains no valid data")

        if self.alpha is not None:
            # Single alpha calculation
            bfi, base_flow = _single_bfi_calculation(
                discharge_values, self.alpha, self.passes, self.reflect_points
            )
        else:
            # Ensemble calculation
            bfi, base_flow = _bfi_ensemble(
                discharge_values, self.passes, self.reflect_points, self.ensemble_size
            )

        # Create pandas Series with original index (excluding NaN values)
        valid_index = discharge.dropna().index
        base_flow_series = pd.Series(base_flow, index=valid_index)

        return bfi, base_flow_series

    def calculate_baseflow_stats(self, discharge: pd.Series) -> dict[str, float]:
        """Calculate comprehensive base flow statistics.

        Args:
            discharge: Discharge time series

        Returns:
            Dictionary of base flow statistics
        """
        bfi, base_flow = self.separate(discharge)

        stats = {
            "bfi": bfi,
            "baseflow_mean": float(np.mean(base_flow)),
            "baseflow_median": float(np.median(base_flow)),
            "baseflow_std": float(np.std(base_flow)),
            "baseflow_cv": float(np.std(base_flow) / np.mean(base_flow)),
            "baseflow_min": float(np.min(base_flow)),
            "baseflow_max": float(np.max(base_flow)),
            "quickflow_ratio": 1.0 - bfi,
        }

        return stats


def calculate_bfi(
    discharge: pd.Series,
    alpha: float | None = None,
    passes: int = 3,
    reflect_points: int = 30,
) -> float:
    """Calculate Base Flow Index using digital filtering.

    This is a convenience function for quick BFI calculation.

    Args:
        discharge: Discharge time series
        alpha: Filter parameter. If None, uses ensemble approach
        passes: Number of filter passes
        reflect_points: Number of boundary reflection points

    Returns:
        Base Flow Index value
    """
    separator = BaseFlowSeparation(alpha=alpha, passes=passes, reflect_points=reflect_points)
    bfi, _ = separator.separate(discharge)
    return bfi
