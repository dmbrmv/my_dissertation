"""Hydrological signature analysis with robust error quantification.

This module provides safe calculation of percentage errors for key hydrological
signatures (Q5, Q95, BFI, Mean Flow) with proper handling of near-zero observed
values and quality grading specific to flow regime characteristics.

Physical Interpretations:
    - Mean Flow: Water balance accuracy, sensitive to precipitation/PET forcing
    - BFI (Base Flow Index): Groundwater contribution, reflects subsurface processes
    - Q5 (High Flow): Flood response, sensitive to precipitation intensity & routing
    - Q95 (Low Flow): Drought characteristics, sensitive to storage & baseflow

References:
    McMillan et al. (2017): "Hydrological signatures for diagnostic evaluation of
    catchment models", Hydrological Processes.

    Addor et al. (2018): "A ranking of hydrological signatures", WRR.
"""

import numpy as np
import pandas as pd

from ..utils.logger import setup_logger

logger = setup_logger("hydrological_signatures", log_file="logs/hydrological.log")

# ============================================================================
# EPSILON THRESHOLDS FOR NEAR-ZERO HANDLING
# ============================================================================

# Minimum observed value threshold for percentage error calculation (mm/day)
# Below this threshold, switch to absolute error to avoid division instability
EPSILON_MEAN_FLOW = 0.01  # 0.01 mm/day (essentially zero flow)
EPSILON_BFI = 0.001  # 0.001 BFI units (index is dimensionless 0-1)
EPSILON_Q5 = 0.1  # 0.1 mm/day (high flow minimum threshold)
EPSILON_Q95 = 0.01  # 0.01 mm/day (low flow - ephemeral risk!)


def safe_percentage_error(
    observed: float,
    simulated: float,
    epsilon: float,
    absolute_error_fallback: bool = True,
) -> float:
    """Calculate percentage error with safe handling of near-zero observed values.

    Percentage Error Formula: ((Sim - Obs) / Obs) * 100

    When observed value is below epsilon threshold, either:
    1. Return absolute error (Sim - Obs) if absolute_error_fallback=True
    2. Return NaN if absolute_error_fallback=False

    Args:
        observed: Observed value
        simulated: Simulated value
        epsilon: Minimum threshold for observed value (below this = near-zero)
        absolute_error_fallback: If True, return absolute error when Obs < epsilon

    Returns:
        Percentage error (%) or absolute error if near-zero observed

    Example:
        >>> safe_percentage_error(0.5, 0.6, epsilon=0.01)  # Normal case
        20.0
        >>> safe_percentage_error(0.005, 0.1, epsilon=0.01)  # Near-zero
        0.095  # Returns absolute error: 0.1 - 0.005
    """
    if np.isnan(observed) or np.isnan(simulated):
        return np.nan

    # Near-zero observed value: use absolute error
    if abs(observed) < epsilon:
        if absolute_error_fallback:
            return float(simulated - observed)  # Absolute error
        else:
            return np.nan

    # Standard percentage error calculation
    return ((simulated - observed) / observed) * 100.0


def calculate_signature_errors(
    obs_mean: float,
    sim_mean: float,
    obs_bfi: float,
    sim_bfi: float,
    obs_q5: float,
    sim_q5: float,
    obs_q95: float,
    sim_q95: float,
) -> dict[str, float]:
    """Calculate safe percentage errors for all hydrological signatures.

    This function replaces the unsafe inline calculations in c3_BlindMetrics.ipynb
    with robust error handling for near-zero observed values.

    Args:
        obs_mean: Observed mean flow (mm/day)
        sim_mean: Simulated mean flow (mm/day)
        obs_bfi: Observed base flow index (0-1)
        sim_bfi: Simulated base flow index (0-1)
        obs_q5: Observed Q5 - high flow exceeded 5% of time (mm/day)
        sim_q5: Simulated Q5 - high flow exceeded 5% of time (mm/day)
        obs_q95: Observed Q95 - low flow exceeded 95% of time (mm/day)
        sim_q95: Simulated Q95 - low flow exceeded 95% of time (mm/day)

    Returns:
        Dictionary with signed percentage errors (%) for each signature:
            - mean_error_pct: Mean flow error
            - bfi_error_pct: Base flow index error
            - q5_error_pct: High flow error
            - q95_error_pct: Low flow error
            - q95_abs_error: Absolute error for Q95 (fallback when near-zero)

    Example:
        >>> errors = calculate_signature_errors(
        ...     obs_mean=2.5,
        ...     sim_mean=2.8,
        ...     obs_bfi=0.35,
        ...     sim_bfi=0.40,
        ...     obs_q5=8.0,
        ...     sim_q5=9.5,
        ...     obs_q95=0.05,
        ...     sim_q95=0.08,  # Near-zero low flow!
        ... )
        >>> errors["mean_error_pct"]  # Normal percentage
        12.0
        >>> errors["q95_abs_error"]  # Absolute error for near-zero
        0.03
    """
    return {
        "mean_error_pct": safe_percentage_error(
            obs_mean, sim_mean, epsilon=EPSILON_MEAN_FLOW
        ),
        "bfi_error_pct": safe_percentage_error(obs_bfi, sim_bfi, epsilon=EPSILON_BFI),
        "q5_error_pct": safe_percentage_error(obs_q5, sim_q5, epsilon=EPSILON_Q5),
        "q95_error_pct": safe_percentage_error(obs_q95, sim_q95, epsilon=EPSILON_Q95),
        # Always provide absolute error for Q95 (useful for ephemeral rivers)
        "q95_abs_error": float(sim_q95 - obs_q95) if not np.isnan(obs_q95) else np.nan,
    }


# ============================================================================
# QUALITY GRADING THRESHOLDS FOR SIGNATURES (0-3 SCALE)
# ============================================================================

# Quality Grade Scale (aligned with calculate_quality_grades in metrics_enhanced.py):
# 0: Poor (Плох.) - Unsatisfactory performance
# 1: Satisfactory (Удов.) - Acceptable performance
# 2: Good (Хор.) - Good performance
# 3: Excellent (Отл.) - Excellent performance


def grade_mean_flow_error(error_pct: float) -> int:
    """Grade mean flow error (STRICTEST - water balance is fundamental).

    Thresholds (absolute percentage):
        - Excellent: ≤ 5% (tight water balance)
        - Good: 5-10% (acceptable balance)
        - Satisfactory: 10-25% (marginal balance)
        - Poor: > 25% (unacceptable bias)

    Rationale: Mean flow reflects cumulative water balance. Models should
    match this well since it integrates over long periods.

    Args:
        error_pct: Signed percentage error (%)

    Returns:
        Quality grade (0-3)
    """
    abs_error = abs(error_pct)
    if abs_error <= 5.0:
        return 3  # Excellent
    elif abs_error <= 10.0:
        return 2  # Good
    elif abs_error <= 25.0:
        return 1  # Satisfactory
    else:
        return 0  # Poor


def grade_bfi_error(error_pct: float) -> int:
    """Grade base flow index error (MODERATE - reflects subsurface process).

    Thresholds (absolute percentage):
        - Excellent: ≤ 10% (captures groundwater well)
        - Good: 10-20% (reasonable groundwater)
        - Satisfactory: 20-40% (marginal groundwater)
        - Poor: > 40% (fails groundwater representation)

    Rationale: BFI is more uncertain due to separation method assumptions
    and lack of groundwater data. Allow more tolerance than mean flow.

    Args:
        error_pct: Signed percentage error (%)

    Returns:
        Quality grade (0-3)
    """
    abs_error = abs(error_pct)
    if abs_error <= 10.0:
        return 3  # Excellent
    elif abs_error <= 20.0:
        return 2  # Good
    elif abs_error <= 40.0:
        return 1  # Satisfactory
    else:
        return 0  # Poor


def grade_q5_error(error_pct: float) -> int:
    """Grade Q5 (high flow) error (MODERATE-STRICT - flood response critical).

    Thresholds (absolute percentage):
        - Excellent: ≤ 15% (accurate flood peaks)
        - Good: 15-25% (acceptable flood response)
        - Satisfactory: 25-50% (marginal flood response)
        - Poor: > 50% (fails flood representation)

    Rationale: High flows are sensitive to precipitation forcing and routing.
    Tighter than Q95 but looser than mean due to event-based uncertainty.

    Args:
        error_pct: Signed percentage error (%)

    Returns:
        Quality grade (0-3)
    """
    abs_error = abs(error_pct)
    if abs_error <= 15.0:
        return 3  # Excellent
    elif abs_error <= 25.0:
        return 2  # Good
    elif abs_error <= 50.0:
        return 1  # Satisfactory
    else:
        return 0  # Poor


def grade_q95_error(error_pct: float) -> int:
    """Grade Q95 (low flow) error (MOST LENIENT - measurement uncertainty high).

    Thresholds (absolute percentage):
        - Excellent: ≤ 20% (accurate low flows)
        - Good: 20-35% (acceptable low flows)
        - Satisfactory: 35-60% (marginal low flows)
        - Poor: > 60% (fails low flow representation)

    Rationale: Low flows are hardest to simulate due to:
        - Measurement errors at low stages (rating curve extrapolation)
        - Deep groundwater processes not in models
        - Zero-flow ephemeral rivers (percentage error unstable)

    Allow highest tolerance among all signatures.

    Args:
        error_pct: Signed percentage error (%) or absolute error for near-zero

    Returns:
        Quality grade (0-3)
    """
    abs_error = abs(error_pct)
    if abs_error <= 20.0:
        return 3  # Excellent
    elif abs_error <= 35.0:
        return 2  # Good
    elif abs_error <= 60.0:
        return 1  # Satisfactory
    else:
        return 0  # Poor


def calculate_signature_quality_grades(
    mean_error_pct: float,
    bfi_error_pct: float,
    q5_error_pct: float,
    q95_error_pct: float,
) -> dict[str, int | float]:
    """Calculate quality grades for all hydrological signatures.

    This function applies signature-specific grading thresholds that account
    for the different uncertainty levels and physical importance of each metric.

    Args:
        mean_error_pct: Mean flow percentage error (%)
        bfi_error_pct: BFI percentage error (%)
        q5_error_pct: Q5 (high flow) percentage error (%)
        q95_error_pct: Q95 (low flow) percentage error (%)

    Returns:
        Dictionary with individual grades and composite score:
            - mean_flow_grade: Grade for mean flow (0-3)
            - bfi_grade: Grade for BFI (0-3)
            - q5_grade: Grade for Q5 (0-3)
            - q95_grade: Grade for Q95 (0-3)
            - signature_composite_score: Average of four grades (0.0-3.0)
            - signature_composite_grade: Overall categorical grade (0-3)

    Example:
        >>> grades = calculate_signature_quality_grades(
        ...     mean_error_pct=8.0,  # Good water balance
        ...     bfi_error_pct=18.0,  # Good groundwater
        ...     q5_error_pct=22.0,  # Good flood response
        ...     q95_error_pct=45.0,  # Marginal low flow
        ... )
        >>> grades["signature_composite_grade"]
        2  # Overall "Good" performance
    """
    # Calculate individual grades using signature-specific thresholds
    mean_grade = (
        grade_mean_flow_error(mean_error_pct) if not np.isnan(mean_error_pct) else 0
    )
    bfi_grade = grade_bfi_error(bfi_error_pct) if not np.isnan(bfi_error_pct) else 0
    q5_grade = grade_q5_error(q5_error_pct) if not np.isnan(q5_error_pct) else 0
    q95_grade = grade_q95_error(q95_error_pct) if not np.isnan(q95_error_pct) else 0

    # Calculate composite score (simple average)
    composite_score = (mean_grade + bfi_grade + q5_grade + q95_grade) / 4.0

    # Convert to categorical grade
    if composite_score < 0.5:
        composite_grade = 0  # Poor
    elif composite_score < 1.5:
        composite_grade = 1  # Satisfactory
    elif composite_score < 2.5:
        composite_grade = 2  # Good
    else:
        composite_grade = 3  # Excellent

    return {
        "mean_flow_grade": mean_grade,
        "bfi_grade": bfi_grade,
        "q5_grade": q5_grade,
        "q95_grade": q95_grade,
        "signature_composite_score": composite_score,
        "signature_composite_grade": composite_grade,
    }


def analyze_signature_errors_comprehensive(
    gauge_id: str,
    obs_series: pd.Series,
    sim_series: pd.Series,
) -> dict[str, float | int | str]:
    """Complete hydrological signature analysis with errors and grading.

    This is the ALL-IN-ONE function to replace the manual calculations in
    c3_BlindMetrics.ipynb. It handles:
        1. Calculation of all signatures from time series
        2. Safe percentage error calculation
        3. Quality grading with signature-specific thresholds

    Args:
        gauge_id: Gauge identifier for logging
        obs_series: Observed discharge time series (pd.Series with datetime index)
        sim_series: Simulated discharge time series (pd.Series with datetime index)

    Returns:
        Dictionary containing:
            - Observed signatures (obs_mean, obs_bfi, obs_q5, obs_q95)
            - Simulated signatures (sim_mean, sim_bfi, sim_q5, sim_q95)
            - Percentage errors (mean_error_pct, bfi_error_pct, q5_error_pct, q95_error_pct)
            - Absolute error for Q95 (q95_abs_error)
            - Quality grades (mean_flow_grade, bfi_grade, q5_grade, q95_grade)
            - Composite score and grade (signature_composite_score, signature_composite_grade)

    Example:
        >>> result = analyze_signature_errors_comprehensive("1234", obs, sim)
        >>> print(f"Mean Flow Error: {result['mean_error_pct']:.1f}%")
        >>> print(f"Overall Grade: {result['signature_composite_grade']}/3")

    Notes:
        - Uses FlowExtremes and calculate_bfi from existing codebase
        - Automatically logs warnings for near-zero observed values
        - Returns np.nan for invalid calculations (insufficient data, etc.)
    """
    from ..hydro.base_flow import calculate_bfi
    from ..hydro.flow_extremes import FlowExtremes

    # ===== CALCULATE OBSERVED SIGNATURES =====
    obs_mean = float(np.mean(obs_series))

    # BFI using ensemble Lyne-Hollick filter
    obs_bfi = calculate_bfi(obs_series, alpha=None, passes=3, reflect_points=30)

    # Flow extremes (quantiles)
    obs_extremes = FlowExtremes(obs_series)
    obs_quantiles = obs_extremes.calculate_flow_quantiles(quantiles=[0.05, 0.95])

    # IMPORTANT: Verify naming convention!
    # q05_0 = quantile(0.05) = 5th percentile = Q95 (exceeded 95% of time)
    # q95_0 = quantile(0.95) = 95th percentile = Q5 (exceeded 5% of time)
    obs_q5 = obs_quantiles["q95_0"]  # High flow (95th percentile)
    obs_q95 = obs_quantiles["q05_0"]  # Low flow (5th percentile)

    # ===== CALCULATE SIMULATED SIGNATURES =====
    sim_mean = float(np.mean(sim_series))
    sim_bfi = calculate_bfi(sim_series, alpha=None, passes=3, reflect_points=30)

    sim_extremes = FlowExtremes(sim_series)
    sim_quantiles = sim_extremes.calculate_flow_quantiles(quantiles=[0.05, 0.95])
    sim_q5 = sim_quantiles["q95_0"]  # High flow
    sim_q95 = sim_quantiles["q05_0"]  # Low flow

    # ===== CALCULATE ERRORS WITH SAFE DIVISION =====
    errors = calculate_signature_errors(
        obs_mean=obs_mean,
        sim_mean=sim_mean,
        obs_bfi=obs_bfi,
        sim_bfi=sim_bfi,
        obs_q5=obs_q5,
        sim_q5=sim_q5,
        obs_q95=obs_q95,
        sim_q95=sim_q95,
    )

    # Log warning if Q95 is near-zero (ephemeral river)
    if abs(obs_q95) < EPSILON_Q95:
        logger.warning(
            f"Gauge {gauge_id}: Q95 near-zero ({obs_q95:.4f} mm/day). "
            f"Using absolute error instead of percentage."
        )

    # ===== CALCULATE QUALITY GRADES =====
    grades = calculate_signature_quality_grades(
        mean_error_pct=errors["mean_error_pct"],
        bfi_error_pct=errors["bfi_error_pct"],
        q5_error_pct=errors["q5_error_pct"],
        q95_error_pct=errors["q95_error_pct"],
    )

    # ===== COMBINE ALL RESULTS =====
    return {
        "gauge_id": gauge_id,
        # Observed signatures
        "obs_mean": obs_mean,
        "obs_bfi": obs_bfi,
        "obs_q5": obs_q5,
        "obs_q95": obs_q95,
        # Simulated signatures
        "sim_mean": sim_mean,
        "sim_bfi": sim_bfi,
        "sim_q5": sim_q5,
        "sim_q95": sim_q95,
        # Percentage errors (signed)
        "mean_error_pct": errors["mean_error_pct"],
        "bfi_error_pct": errors["bfi_error_pct"],
        "q5_error_pct": errors["q5_error_pct"],
        "q95_error_pct": errors["q95_error_pct"],
        "q95_abs_error": errors["q95_abs_error"],
        # Quality grades
        "mean_flow_grade": grades["mean_flow_grade"],
        "bfi_grade": grades["bfi_grade"],
        "q5_grade": grades["q5_grade"],
        "q95_grade": grades["q95_grade"],
        "signature_composite_score": grades["signature_composite_score"],
        "signature_composite_grade": grades["signature_composite_grade"],
    }
