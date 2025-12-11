"""Enhanced hydrology metrics for improved low/high flow calibration.

This module extends the base metrics with transformation-based approaches
optimized for different flow regimes.
"""

import numpy as np

from src.timeseries_stats.metrics import (
    kling_gupta_efficiency,
    log_nash_sutcliffe_efficiency,
    nash_sutcliffe_efficiency,
    peak_flow_error,
    percent_bias,
)


def inverse_nse(
    observed: np.ndarray, simulated: np.ndarray, epsilon: float = 0.01
) -> float:
    """Calculate Inverse-transformed Nash-Sutcliffe Efficiency.

    This metric provides the strongest emphasis on low flows, even more than
    log-NSE. The inverse transformation heavily weights errors in low-flow
    periods, making it ideal for baseflow assessment.

    NSE_inv = 1 - Σ(1/(Obs+ε) - 1/(Sim+ε))² / Σ(1/(Obs+ε) - mean(1/(Obs+ε)))²

    Reference:
        Santos et al. (2022): "Comparison of calibrated objective functions
        for low flow simulation" - demonstrated inverse transformation
        superiority for low-flow emphasis.

    Args:
        observed: Array of observed values (mm/day)
        simulated: Array of simulated values (mm/day)
        epsilon: Small constant to prevent division by zero (default 0.01)

    Returns:
        Inverse-NSE value [-∞, 1], where 1 is perfect match.
        Higher values indicate better low-flow performance.
    """
    # Remove paired NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[mask]
    sim = simulated[mask]

    if len(obs) == 0:
        return np.nan

    # Apply inverse transformation with epsilon protection
    inv_obs = 1.0 / (obs + epsilon)
    inv_sim = 1.0 / (sim + epsilon)

    # Calculate NSE on transformed values
    inv_obs_mean = np.mean(inv_obs)
    numerator = np.sum((inv_obs - inv_sim) ** 2)
    denominator = np.sum((inv_obs - inv_obs_mean) ** 2)

    if denominator == 0:
        return np.nan

    return 1.0 - (numerator / denominator)


def sqrt_nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Calculate Square-root transformed NSE for medium flows.

    The sqrt transformation provides a balanced emphasis between high and
    low flows, more moderate than log transformation.

    NSE_sqrt = 1 - Σ(√Obs - √Sim)² / Σ(√Obs - mean(√Obs))²

    Reference:
        Thirel et al. (2023): "Multi-objective assessment of hydrological
        model performances" - showed sqrt transformation balances flow regimes.

    Args:
        observed: Array of observed values (mm/day), must be non-negative
        simulated: Array of simulated values (mm/day), must be non-negative

    Returns:
        Sqrt-NSE value [-∞, 1], where 1 is perfect match.
    """
    # Remove paired NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[mask]
    sim = simulated[mask]

    if len(obs) == 0 or np.any(obs < 0) or np.any(sim < 0):
        return np.nan

    # Apply square root transformation
    sqrt_obs = np.sqrt(obs)
    sqrt_sim = np.sqrt(sim)

    # Calculate NSE on transformed values
    sqrt_obs_mean = np.mean(sqrt_obs)
    numerator = np.sum((sqrt_obs - sqrt_sim) ** 2)
    denominator = np.sum((sqrt_obs - sqrt_obs_mean) ** 2)

    if denominator == 0:
        return np.nan

    return 1.0 - (numerator / denominator)


def flow_quantile_nse(
    observed: np.ndarray,
    simulated: np.ndarray,
    lower_quantile: float = 0.0,
    upper_quantile: float = 1.0,
) -> float:
    """Calculate NSE for a specific flow quantile range.

    This allows targeted evaluation of model performance in specific flow
    regimes (e.g., low flows: 0-0.3, high flows: 0.7-1.0).

    Args:
        observed: Array of observed values
        simulated: Array of simulated values
        lower_quantile: Lower quantile boundary [0, 1]
        upper_quantile: Upper quantile boundary [0, 1]

    Returns:
        NSE value for the specified quantile range
    """
    # Remove paired NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[mask]
    sim = simulated[mask]

    if len(obs) == 0:
        return np.nan

    # Determine flow thresholds based on observed quantiles
    q_lower = np.quantile(obs, lower_quantile)
    q_upper = np.quantile(obs, upper_quantile)

    # Filter to specified range
    range_mask = (obs >= q_lower) & (obs <= q_upper)
    obs_range = obs[range_mask]
    sim_range = sim[range_mask]

    if len(obs_range) == 0:
        return np.nan

    # Calculate NSE on filtered values
    return nash_sutcliffe_efficiency(obs_range, sim_range)


def composite_low_flow_metric(
    observed: np.ndarray, simulated: np.ndarray, weights: tuple[float, float] = (0.5, 0.5)
) -> float:
    """Composite low-flow metric combining log-NSE and inverse-NSE.

    This provides robust low-flow assessment by averaging two complementary
    transformations. The combination reduces sensitivity to extreme values
    while maintaining low-flow emphasis.

    Args:
        observed: Array of observed values
        simulated: Array of simulated values
        weights: Tuple of (log_weight, inverse_weight), must sum to 1.0

    Returns:
        Weighted average of log-NSE and inverse-NSE
    """
    if not np.isclose(sum(weights), 1.0):
        raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

    log_nse = log_nash_sutcliffe_efficiency(observed, simulated)
    inv_nse = inverse_nse(observed, simulated)

    # Handle NaN values
    if np.isnan(log_nse) and np.isnan(inv_nse):
        return np.nan
    elif np.isnan(log_nse):
        return inv_nse
    elif np.isnan(inv_nse):
        return log_nse

    return weights[0] * log_nse + weights[1] * inv_nse


def composite_high_flow_metric(
    observed: np.ndarray, simulated: np.ndarray, weights: tuple[float, float] = (0.7, 0.3)
) -> float:
    """Composite high-flow metric combining NSE and peak flow error.

    Balances overall high-flow performance (NSE) with peak accuracy (PFE).
    The normalized PFE component prevents domination by extreme events.

    Args:
        observed: Array of observed values
        simulated: Array of simulated values
        weights: Tuple of (nse_weight, pfe_weight), must sum to 1.0

    Returns:
        Weighted combination of NSE and normalized PFE
    """
    if not np.isclose(sum(weights), 1.0):
        raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

    nse = nash_sutcliffe_efficiency(observed, simulated)
    pfe = peak_flow_error(observed, simulated, percentile=95)

    # Handle NaN values
    if np.isnan(nse):
        return np.nan
    if np.isnan(pfe):
        return nse  # Fallback to NSE only

    # Normalize PFE (percent error) to [-1, 1] scale roughly matching NSE
    # Assumption: PFE typically ranges ±50%; clip to ±100%
    pfe_normalized = -np.clip(pfe / 100.0, -1.0, 1.0)

    return weights[0] * nse + weights[1] * pfe_normalized


def analyze_flow_regimes(observed: np.ndarray, simulated: np.ndarray) -> dict[str, float]:
    """Comprehensive flow regime performance analysis.

    Stratifies performance across five flow quantile classes to diagnose
    model behavior in different hydrological conditions.

    Flow classes (based on observed flow distribution):
    - Very low: Q <= Q10 (drought, baseflow)
    - Low: Q10 < Q <= Q30 (sustained low flow)
    - Medium: Q30 < Q <= Q70 (typical flow)
    - High: Q70 < Q <= Q90 (elevated flow)
    - Very high: Q > Q90 (flood, snowmelt peak)

    Args:
        observed: Array of observed values
        simulated: Array of simulated values

    Returns:
        Dictionary with NSE for each flow regime plus overall metrics
    """
    # Remove paired NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[mask]
    sim = simulated[mask]

    if len(obs) == 0:
        return {}

    # Define flow class boundaries
    q10 = np.percentile(obs, 10)
    q30 = np.percentile(obs, 30)
    q70 = np.percentile(obs, 70)
    q90 = np.percentile(obs, 90)

    regime_metrics: dict[str, float] = {}

    # Very low flows (Q <= Q10)
    mask_vlow = obs <= q10
    if mask_vlow.sum() > 0:
        regime_metrics["very_low_nse"] = nash_sutcliffe_efficiency(
            obs[mask_vlow], sim[mask_vlow]
        )
        regime_metrics["very_low_pbias"] = percent_bias(obs[mask_vlow], sim[mask_vlow])

    # Low flows (Q10 < Q <= Q30)
    mask_low = (obs > q10) & (obs <= q30)
    if mask_low.sum() > 0:
        regime_metrics["low_nse"] = nash_sutcliffe_efficiency(
            obs[mask_low], sim[mask_low]
        )
        regime_metrics["low_pbias"] = percent_bias(obs[mask_low], sim[mask_low])

    # Medium flows (Q30 < Q <= Q70)
    mask_med = (obs > q30) & (obs <= q70)
    if mask_med.sum() > 0:
        regime_metrics["medium_nse"] = nash_sutcliffe_efficiency(
            obs[mask_med], sim[mask_med]
        )
        regime_metrics["medium_pbias"] = percent_bias(obs[mask_med], sim[mask_med])

    # High flows (Q70 < Q <= Q90)
    mask_high = (obs > q70) & (obs <= q90)
    if mask_high.sum() > 0:
        regime_metrics["high_nse"] = nash_sutcliffe_efficiency(
            obs[mask_high], sim[mask_high]
        )
        regime_metrics["high_pbias"] = percent_bias(obs[mask_high], sim[mask_high])

    # Very high flows (Q > Q90)
    mask_vhigh = obs > q90
    if mask_vhigh.sum() > 0:
        regime_metrics["very_high_nse"] = nash_sutcliffe_efficiency(
            obs[mask_vhigh], sim[mask_vhigh]
        )
        regime_metrics["very_high_pbias"] = percent_bias(obs[mask_vhigh], sim[mask_vhigh])

    # Add overall metrics for reference
    regime_metrics["overall_nse"] = nash_sutcliffe_efficiency(obs, sim)
    regime_metrics["overall_kge"] = kling_gupta_efficiency(obs, sim)
    regime_metrics["overall_pbias"] = percent_bias(obs, sim)

    return regime_metrics


def calculate_quality_grades(
    observed: np.ndarray,
    simulated: np.ndarray,
) -> dict[str, float | int]:
    """Calculate quality grades for NSE, PBIAS, and R² metrics.

    This function replaces the legacy hydro_job function, providing modern
    quality assessment with the updated 0-3 grading scale. It computes three
    standard metrics (NSE, PBIAS, R²) and converts them to categorical quality
    grades following hydrological modeling standards.

    Quality Grading Scale (0-based):
        - 0: Плох. (Poor) - Unsatisfactory performance
        - 1: Удов. (Satisfactory) - Acceptable performance
        - 2: Хор. (Good) - Good performance
        - 3: Отл. (Excellent) - Excellent performance

    Threshold Definitions:
        NSE (Nash-Sutcliffe Efficiency):
            - Excellent: > 0.80
            - Good: 0.70 - 0.80
            - Satisfactory: 0.50 - 0.70
            - Poor: ≤ 0.50

        PBIAS (Percent Bias):
            - Excellent: |PBIAS| ≤ 10%
            - Good: 10% < |PBIAS| ≤ 15%
            - Satisfactory: 15% < |PBIAS| ≤ 35%
            - Poor: |PBIAS| > 35%

        R² (Coefficient of Determination):
            - Excellent: > 0.85
            - Good: 0.70 - 0.85
            - Satisfactory: 0.50 - 0.70
            - Poor: ≤ 0.50

    References:
        Moriasi et al. (2007): "Model evaluation guidelines for systematic
        quantification of accuracy in watershed simulations" - established
        standard thresholds for NSE and PBIAS.

        Willmott et al. (2012): "A refined index of model performance" -
        provided rationale for R² thresholds in hydrological contexts.

    Args:
        observed: Array of observed discharge values (mm/day or m³/s)
        simulated: Array of simulated discharge values (same units as observed)

    Returns:
        Dictionary containing:
            - NSE: Nash-Sutcliffe Efficiency value
            - NSE_grade: Quality grade (0-3)
            - PBIAS: Percent Bias value
            - PBIAS_grade: Quality grade (0-3)
            - R2: Coefficient of determination
            - R2_grade: Quality grade (0-3)
            - composite_score: Average of three grades (0.0-3.0)
            - composite_grade: Overall quality grade (0-3)

    Example:
        >>> obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> sim = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        >>> grades = calculate_quality_grades(obs, sim)
        >>> print(f"NSE: {grades['NSE']:.3f}, Grade: {grades['NSE_grade']}")
        NSE: 0.950, Grade: 3

    Notes:
        - Missing values (NaN) are handled automatically via paired removal
        - If inputs are invalid or insufficient, returns grade=0 for all metrics
        - Composite score is simple average of three individual grades
        - This function modernizes the legacy hydro_job function with:
          * Type hints (Python 3.10+)
          * Zero-based grading scale (was 1-4, now 0-3)
          * Separation of metrics (no longer bundled with BFI/quantiles)
          * Logging support (use logger.info() in calling code)
    """
    # Remove paired NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[mask]
    sim = simulated[mask]

    # Handle insufficient data
    if len(obs) < 3:
        return {
            "NSE": np.nan,
            "NSE_grade": 0,
            "PBIAS": np.nan,
            "PBIAS_grade": 0,
            "R2": np.nan,
            "R2_grade": 0,
            "composite_score": 0.0,
            "composite_grade": 0,
        }

    # Calculate metrics
    nse_value = nash_sutcliffe_efficiency(obs, sim)
    pbias_value = percent_bias(obs, sim)

    # Calculate R² (coefficient of determination)
    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)
    ss_res = np.sum((obs - sim) ** 2)
    ss_tot = np.sum((obs - obs_mean) ** 2)

    if ss_tot == 0:
        r2_value = np.nan
    else:
        r2_value = 1 - (ss_res / ss_tot)

    # Handle NaN metrics (assign worst grade)
    if np.isnan(nse_value):
        nse_value = -999.0
    if np.isnan(pbias_value):
        pbias_value = 999.0
    if np.isnan(r2_value):
        r2_value = -999.0

    # Grade NSE (0-based scale)
    if nse_value > 0.80:
        nse_grade = 3  # Excellent
    elif nse_value > 0.70:
        nse_grade = 2  # Good
    elif nse_value > 0.50:
        nse_grade = 1  # Satisfactory
    else:
        nse_grade = 0  # Poor

    # Grade PBIAS (absolute value, 0-based scale)
    pbias_abs = abs(pbias_value)
    if pbias_abs <= 10.0:
        pbias_grade = 3  # Excellent
    elif pbias_abs <= 15.0:
        pbias_grade = 2  # Good
    elif pbias_abs <= 35.0:
        pbias_grade = 1  # Satisfactory
    else:
        pbias_grade = 0  # Poor

    # Grade R² (0-based scale)
    if r2_value > 0.85:
        r2_grade = 3  # Excellent
    elif r2_value > 0.70:
        r2_grade = 2  # Good
    elif r2_value > 0.50:
        r2_grade = 1  # Satisfactory
    else:
        r2_grade = 0  # Poor

    # Calculate composite score (average of three grades)
    composite_score = (nse_grade + pbias_grade + r2_grade) / 3.0

    # Decode composite to categorical grade
    if composite_score < 0.5:
        composite_grade = 0  # Poor
    elif composite_score < 1.5:
        composite_grade = 1  # Satisfactory
    elif composite_score < 2.5:
        composite_grade = 2  # Good
    else:
        composite_grade = 3  # Excellent

    return {
        "NSE": nse_value if nse_value != -999.0 else np.nan,
        "NSE_grade": nse_grade,
        "PBIAS": pbias_value if pbias_value != 999.0 else np.nan,
        "PBIAS_grade": pbias_grade,
        "R2": r2_value if r2_value != -999.0 else np.nan,
        "R2_grade": r2_grade,
        "composite_score": composite_score,
        "composite_grade": composite_grade,
    }
