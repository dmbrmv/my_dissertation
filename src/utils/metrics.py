"""Hydrology model evaluation metrics.

This module contains functions for calculating common performance metrics
used in hydrology to evaluate models. The metrics include NSE, KGE, PBIAS,
RMSE, log-transformed NSE, r-squared, and peak-flow relative error.
"""

import numpy as np


def nash_sutcliffe_efficiency(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Calculate Nash-Sutcliffe Efficiency (NSE) between observed and simulated values.

    NSE = 1 - Σ(Obs - Sim)² / Σ(Obs - mean(Obs))²

    Args:
        observed: Array of observed values
        simulated: Array of simulated values

    Returns:
        NSE value [-∞, 1], where 1 is perfect match
    """
    # Remove any paired NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    observed = observed[mask]
    simulated = simulated[mask]

    if len(observed) == 0:
        return np.nan

    observed_mean = np.mean(observed)
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - observed_mean) ** 2)

    if denominator == 0:
        return np.nan

    return 1 - (numerator / denominator)


def kling_gupta_efficiency(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Calculate Kling-Gupta Efficiency (KGE) between observed and simulated values.

    KGE = 1 - √[(r-1)² + (α-1)² + (β-1)²]
    where:
    r = correlation coefficient
    α = std(sim) / std(obs)
    β = mean(sim) / mean(obs)

    Args:
        observed: Array of observed values
        simulated: Array of simulated values

    Returns:
        KGE value [-∞, 1], where 1 is perfect match
    """
    # Remove any paired NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    observed = observed[mask]
    simulated = simulated[mask]

    if len(observed) == 0 or np.mean(observed) == 0 or np.std(observed) == 0:
        return np.nan

    # Correlation component
    r = np.corrcoef(observed, simulated)[0, 1]

    # Variability component
    alpha = np.std(simulated) / np.std(observed)

    # Bias component
    beta = np.mean(simulated) / np.mean(observed)

    # KGE calculation
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return kge


def percent_bias(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Calculate Percent Bias (PBIAS) between observed and simulated values.

    PBIAS = 100 × Σ(Sim - Obs) / Σ(Obs)

    Args:
        observed: Array of observed values
        simulated: Array of simulated values

    Returns:
        PBIAS value [%], where 0 is no bias
    """
    # Remove any paired NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    observed = observed[mask]
    simulated = simulated[mask]

    if len(observed) == 0 or np.sum(observed) == 0:
        return np.nan

    return 100 * np.sum(simulated - observed) / np.sum(observed)


def root_mean_squared_error(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE) between observed and simulated values.

    RMSE = √[Σ(Obs - Sim)² / n]

    Args:
        observed: Array of observed values
        simulated: Array of simulated values

    Returns:
        RMSE value [0, ∞), where 0 is perfect match
    """
    # Remove any paired NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    observed = observed[mask]
    simulated = simulated[mask]

    if len(observed) == 0:
        return np.nan

    return np.sqrt(np.mean((observed - simulated) ** 2))


def log_nash_sutcliffe_efficiency(
    observed: np.ndarray, simulated: np.ndarray, epsilon: float = 0.01
) -> float:
    """Calculate Log-transformed Nash-Sutcliffe Efficiency (logNSE).

    This metric emphasizes low flow performance by applying log transformation
    before NSE calculation.

    logNSE = 1 - Σ(log(Obs+ε) - log(Sim+ε))² / Σ(log(Obs+ε) - mean(log(Obs+ε)))²

    Args:
        observed: Array of observed values
        simulated: Array of simulated values
        epsilon: Small constant to add before log transformation to handle zero values

    Returns:
        log-NSE value [-∞, 1], where 1 is perfect match
    """
    # Remove any paired NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    observed = observed[mask]
    simulated = simulated[mask]

    if len(observed) == 0:
        return np.nan

    # Add epsilon and log transform
    obs_plus_eps = observed + epsilon
    sim_plus_eps = simulated + epsilon

    # Ensure all values are strictly positive before log transformation
    if np.any(obs_plus_eps <= 0) or np.any(sim_plus_eps <= 0):
        return np.nan

    log_obs = np.log(obs_plus_eps)
    log_sim = np.log(sim_plus_eps)

    log_obs_mean = np.mean(log_obs)
    numerator = np.sum((log_obs - log_sim) ** 2)
    denominator = np.sum((log_obs - log_obs_mean) ** 2)

    if denominator == 0:
        return np.nan

    return 1 - (numerator / denominator)


def r_squared(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Calculate coefficient of determination (R²) between observed and simulated values.

    R² = [Σ((Obs - mean(Obs)) * (Sim - mean(Sim)))]² / [Σ(Obs - mean(Obs))² * Σ(Sim - mean(Sim))²]

    Args:
        observed: Array of observed values
        simulated: Array of simulated values

    Returns:
        R² value [0, 1], where 1 is perfect linear relationship
    """
    # Remove any paired NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    observed = observed[mask]
    simulated = simulated[mask]

    if len(observed) == 0:
        return np.nan

    # Get correlation coefficient
    r = np.corrcoef(observed, simulated)[0, 1]

    # Return squared value
    return r**2


def peak_flow_error(observed: np.ndarray, simulated: np.ndarray, percentile: float = 95):
    """Calculate relative error in peak flows (high flow conditions).

    This metric focuses on the high flow periods by comparing flow values
    above a specified percentile (default 95%).

    PFE = 100 * [mean(Sim_peak) - mean(Obs_peak)] / mean(Obs_peak)

    Args:
        observed: Array of observed values
        simulated: Array of simulated values
        percentile: Percentile threshold for defining peak flows (default 95%)

    Returns:
        Peak flow relative error [%], where 0 is no bias
    """
    # Remove any paired NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    observed = observed[mask]
    simulated = simulated[mask]

    if len(observed) == 0:
        return np.nan

    # Determine threshold for peak flows based on percentile
    threshold = np.percentile(observed, percentile)

    # Extract peak flow values
    peak_obs = observed[observed >= threshold]
    peak_sim = simulated[observed >= threshold]  # Using same indices as observed peaks

    if len(peak_obs) == 0 or np.mean(peak_obs) == 0:
        return np.nan

    # Calculate relative error
    return 100 * (np.mean(peak_sim) - np.mean(peak_obs)) / np.mean(peak_obs)


def evaluate_model(observed, simulated) -> dict[str, float]:
    """Evaluate model performance using multiple metrics.

    Args:
        observed: Array of observed values
        simulated: Array of simulated values

    Returns:
        Dictionary of metrics
    """
    observed_arr = np.asarray(observed, dtype=float)
    simulated_arr = np.asarray(simulated, dtype=float)
    mask = ~(np.isnan(observed_arr) | np.isnan(simulated_arr))
    observed_clean = observed_arr[mask]
    simulated_clean = simulated_arr[mask]

    # Calculate primary metrics on NaN-filtered data
    nse = nash_sutcliffe_efficiency(observed_clean, simulated_clean)
    kge = kling_gupta_efficiency(observed_clean, simulated_clean)
    pbias = percent_bias(observed_clean, simulated_clean)
    rmse = root_mean_squared_error(observed_clean, simulated_clean)
    mae = (
        np.mean(np.abs(observed_clean - simulated_clean))
        if observed_clean.size
        else np.nan
    )
    log_nse = log_nash_sutcliffe_efficiency(observed_clean, simulated_clean)
    r2 = r_squared(observed_clean, simulated_clean)
    pfe = peak_flow_error(observed_clean, simulated_clean)

    # Create metrics dictionary
    metrics = {
        "NSE": float(nse),
        "KGE": float(kge),
        "PBIAS": float(pbias),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "logNSE": float(log_nse),
        "R2": float(r2),
        "PFE": float(pfe),
    }

    return metrics
