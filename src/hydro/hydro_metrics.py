"""Backward compatibility layer for legacy hydrological metrics.

**DEPRECATED MODULE**: This module exists only for backward compatibility.

Modern implementations are available in:
- `src.hydro.base_flow` - Base flow separation (BFI)
- `src.hydro.flow_extremes` - High/low flow analysis
- `src.hydro.flow_indices` - Comprehensive hydrological indices
- `src.timeseries_stats.metrics` - Model performance metrics (NSE, KGE)

For new code, import from the modern modules directly.
"""

from typing import Any
import warnings

import numpy as np
import pandas as pd


def calculate_hydrological_indices(
    hydro_year: pd.Series,
    calendar_year: pd.Series | None = None,
) -> dict[str, float]:
    """Calculate comprehensive hydrological indices (modern wrapper).

    This is a convenience function that wraps modern implementations.
    For direct access to more features, use the underlying modules.

    Args:
        hydro_year: Discharge series for hydrological year (Oct-Sep)
        calendar_year: Discharge series for calendar year (Jan-Dec, optional)

    Returns:
        Dictionary with hydrological indices:
            - 'mean': Mean discharge
            - 'bfi': Base Flow Index
            - 'q5': 5th percentile flow
            - 'q95': 95th percentile flow
            - 'q1', 'q10', 'q90', 'q99': Additional quantiles
            - 'high_flow_frequency': % of time in high flows
            - 'low_flow_frequency': % of time in low flows

    Example:
        >>> from src.hydro.hydro_metrics import calculate_hydrological_indices
        >>> indices = calculate_hydrological_indices(hydro_year_series)
        >>> print(f"BFI: {indices['bfi']:.3f}")
    """
    from .base_flow import calculate_bfi
    from .flow_extremes import FlowExtremes

    # Basic statistics
    hydro_mean = float(np.nanmean(hydro_year))

    # Calculate BFI using modern Lyne-Hollick filter
    bfi_value = calculate_bfi(
        hydro_year,
        alpha=None,  # Use ensemble approach
        passes=3,
        reflect_points=30,
    )

    # Calculate flow quantiles and extremes
    extremes = FlowExtremes(hydro_year)
    quantiles = extremes.calculate_flow_quantiles(
        quantiles=[0.01, 0.05, 0.1, 0.9, 0.95, 0.99]
    )

    high_flow_metrics = extremes.analyze_high_flows(threshold_multiplier=2.0)
    low_flow_metrics = extremes.analyze_low_flows(threshold_multiplier=0.2)

    return {
        "mean": hydro_mean,
        "bfi": bfi_value,
        "q5": quantiles["q05_0"],
        "q95": quantiles["q95_0"],
        "q1": quantiles["q01_0"],
        "q10": quantiles["q10_0"],
        "q90": quantiles["q90_0"],
        "q99": quantiles["q99_0"],
        "high_flow_frequency": high_flow_metrics["high_flow_frequency"],
        "high_flow_events": high_flow_metrics["high_flow_events"],
        "low_flow_frequency": low_flow_metrics["low_flow_frequency"],
        "low_flow_events": low_flow_metrics["low_flow_events"],
    }


def hydro_job(hydro_year: pd.Series, calendar_year: pd.Series) -> dict[str, Any]:
    """Calculate basic hydrological indices (DEPRECATED).

    **DEPRECATED**: Use `calculate_hydrological_indices()` instead.

    This function maintains the old API for backward compatibility with
    archived notebooks and scripts.

    Args:
        hydro_year: Discharge series for hydrological year (Oct-Sep)
        calendar_year: Discharge series for calendar year (Jan-Dec)

    Returns:
        Dictionary with keys: 'mean', 'bfi', 'q5', 'q95'

    See Also:
        calculate_hydrological_indices: Modern replacement with more indices
    """
    warnings.warn(
        "hydro_job() is deprecated. Use calculate_hydrological_indices() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    from .base_flow import calculate_bfi
    from .flow_extremes import FlowExtremes

    hydro_mean = float(np.nanmean(hydro_year))
    bfi_value = calculate_bfi(hydro_year, alpha=None, passes=3, reflect_points=30)

    extremes = FlowExtremes(hydro_year)
    quantiles = extremes.calculate_flow_quantiles(quantiles=[0.05, 0.95])

    return {
        "mean": hydro_mean,
        "bfi": bfi_value,
        "q5": quantiles["q05_0"],
        "q95": quantiles["q95_0"],
    }


# Legacy metric functions - redirect to proper location
def nse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency (DEPRECATED).

    **DEPRECATED**: Import from `src.timeseries_stats.metrics` instead:
        from src.timeseries_stats.metrics import nash_sutcliffe_efficiency
    """
    warnings.warn(
        "nse() is deprecated. Use nash_sutcliffe_efficiency() from "
        "src.timeseries_stats.metrics instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from ..timeseries_stats.metrics import nash_sutcliffe_efficiency

    return nash_sutcliffe_efficiency(targets, predictions)


def kge(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Kling-Gupta Efficiency (DEPRECATED).

    **DEPRECATED**: Import from `src.timeseries_stats.metrics` instead:
        from src.timeseries_stats.metrics import kling_gupta_efficiency

    Returns:
        Array with [kge, r, alpha, beta] components
    """
    warnings.warn(
        "kge() is deprecated. Use kling_gupta_efficiency() from "
        "src.timeseries_stats.metrics instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from ..timeseries_stats.metrics import kling_gupta_efficiency

    # Old function returned components, new one returns only KGE value
    kge_value = kling_gupta_efficiency(targets, predictions)

    # Calculate components for backward compatibility
    sim_mean = np.mean(targets, dtype=np.float64)
    obs_mean = np.mean(predictions, dtype=np.float64)

    r_num = np.sum((targets - sim_mean) * (predictions - obs_mean), dtype=np.float64)
    r_den = np.sqrt(
        np.sum((targets - sim_mean) ** 2, dtype=np.float64)
        * np.sum((predictions - obs_mean) ** 2, dtype=np.float64)
    )
    r = r_num / r_den
    alpha = np.std(targets) / np.std(predictions, dtype=np.float64)
    beta = np.sum(targets, dtype=np.float64) / np.sum(predictions, dtype=np.float64)

    return np.array([kge_value, r, alpha, beta])


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Root Mean Square Error (DEPRECATED).

    **DEPRECATED**: Use sklearn.metrics.mean_squared_error(squared=False) instead.
    """
    warnings.warn(
        "rmse() is deprecated. Use mean_squared_error(squared=False) from "
        "sklearn.metrics instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from sklearn.metrics import mean_squared_error

    return mean_squared_error(targets, predictions, squared=False)
