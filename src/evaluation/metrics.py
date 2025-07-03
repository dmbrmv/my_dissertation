"""Hydrological evaluation metrics for model assessment."""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def nse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Nash-Sutcliffe Efficiency (NSE).
    
    NSE is a normalized statistic that determines the relative magnitude of the 
    residual variance compared to the measured data variance. NSE = 1 corresponds 
    to a perfect match between model and observations.
    
    Args:
        predictions: Model predictions
        targets: Observed target values
        
    Returns:
        NSE value (ranges from -∞ to 1, with 1 being perfect)
        
    Examples:
        >>> pred = np.array([1.0, 2.0, 3.0, 4.0])
        >>> obs = np.array([1.1, 1.9, 3.1, 3.9])
        >>> nse_value = nse(pred, obs)
        >>> print(f"NSE: {nse_value:.3f}")
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    # Remove NaN values
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    if np.sum(mask) == 0:
        raise ValueError("No valid data points after removing NaNs")
    
    pred_clean = predictions[mask]
    targets_clean = targets[mask]
    
    numerator = np.sum((targets_clean - pred_clean) ** 2)
    denominator = np.sum((targets_clean - np.mean(targets_clean)) ** 2)
    
    if denominator == 0:
        return float('nan')
    
    return 1 - (numerator / denominator)


def kge(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float, float, float]:
    """Calculate Kling-Gupta Efficiency (KGE) and its components.
    
    KGE is a modified version of NSE that decomposes the error into correlation,
    bias, and variability components. KGE = 1 indicates perfect model performance.
    
    Args:
        predictions: Model predictions
        targets: Observed target values
        
    Returns:
        Tuple containing (KGE, correlation, alpha, beta)
        - KGE: Kling-Gupta Efficiency
        - correlation: Pearson correlation coefficient  
        - alpha: Ratio of standard deviations (std_obs/std_sim)
        - beta: Ratio of means (mean_obs/mean_sim)
        
    Examples:
        >>> pred = np.array([1.0, 2.0, 3.0, 4.0])
        >>> obs = np.array([1.1, 1.9, 3.1, 3.9])
        >>> kge_val, r, alpha, beta = kge(pred, obs)
        >>> print(f"KGE: {kge_val:.3f}, r: {r:.3f}, α: {alpha:.3f}, β: {beta:.3f}")
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    # Remove NaN values
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    if np.sum(mask) == 0:
        raise ValueError("No valid data points after removing NaNs")
    
    pred_clean = predictions[mask]
    targets_clean = targets[mask]
    
    # Calculate means
    sim_mean = np.mean(pred_clean)
    obs_mean = np.mean(targets_clean)
    
    # Calculate correlation coefficient
    numerator = np.sum((targets_clean - obs_mean) * (pred_clean - sim_mean))
    denominator = np.sqrt(
        np.sum((targets_clean - obs_mean) ** 2) * np.sum((pred_clean - sim_mean) ** 2)
    )
    
    if denominator == 0:
        correlation = 0.0
    else:
        correlation = numerator / denominator
    
    # Calculate alpha (ratio of standard deviations)
    std_obs = np.std(targets_clean)
    std_sim = np.std(pred_clean)
    
    if std_sim == 0:
        alpha = float('inf') if std_obs > 0 else 1.0
    else:
        alpha = std_obs / std_sim
    
    # Calculate beta (ratio of means)
    if sim_mean == 0:
        beta = float('inf') if obs_mean > 0 else 1.0
    else:
        beta = obs_mean / sim_mean
    
    # Calculate KGE
    kge_value = 1 - np.sqrt(
        (correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2
    )
    
    return kge_value, correlation, alpha, beta


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Root Mean Square Error (RMSE).
    
    Args:
        predictions: Model predictions
        targets: Observed target values
        
    Returns:
        RMSE value (always >= 0, with 0 being perfect)
        
    Examples:
        >>> pred = np.array([1.0, 2.0, 3.0, 4.0])
        >>> obs = np.array([1.1, 1.9, 3.1, 3.9])
        >>> rmse_value = rmse(pred, obs)
        >>> print(f"RMSE: {rmse_value:.3f}")
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    # Remove NaN values
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    if np.sum(mask) == 0:
        raise ValueError("No valid data points after removing NaNs")
    
    pred_clean = predictions[mask]
    targets_clean = targets[mask]
    
    return float(np.sqrt(mean_squared_error(targets_clean, pred_clean)))


def relative_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate mean relative error as a percentage.
    
    Args:
        predictions: Model predictions
        targets: Observed target values
        
    Returns:
        Mean relative error as percentage
        
    Examples:
        >>> pred = np.array([1.0, 2.0, 3.0, 4.0])
        >>> obs = np.array([1.1, 1.9, 3.1, 3.9])
        >>> rel_err = relative_error(pred, obs)
        >>> print(f"Relative Error: {rel_err:.2f}%")
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    # Remove NaN values and zeros in targets (to avoid division by zero)
    mask = ~(np.isnan(predictions) | np.isnan(targets) | (targets == 0))
    if np.sum(mask) == 0:
        raise ValueError("No valid data points after removing NaNs and zeros")
    
    pred_clean = predictions[mask]
    targets_clean = targets[mask]
    
    return float(np.mean(np.abs((targets_clean - pred_clean) / targets_clean) * 100))


def create_metrics_dataframe(
    gauge_id: str, 
    predictions: np.ndarray, 
    targets: np.ndarray
) -> pd.DataFrame:
    """Create a comprehensive metrics DataFrame for a single gauge.
    
    Args:
        gauge_id: Identifier for the gauge
        predictions: Model predictions
        targets: Observed target values
        
    Returns:
        DataFrame with computed metrics
        
    Examples:
        >>> pred = np.array([1.0, 2.0, 3.0, 4.0])
        >>> obs = np.array([1.1, 1.9, 3.1, 3.9])
        >>> df = create_metrics_dataframe("gauge_001", pred, obs)
        >>> print(df.round(3))
    """
    results = pd.DataFrame(index=[gauge_id])
    
    try:
        # Calculate NSE
        results.loc[gauge_id, "NSE"] = nse(predictions, targets)
        
        # Calculate KGE and its components
        kge_value, correlation, alpha, beta = kge(predictions, targets)
        results.loc[gauge_id, "KGE"] = kge_value
        results.loc[gauge_id, "correlation"] = correlation
        results.loc[gauge_id, "alpha"] = alpha
        results.loc[gauge_id, "beta"] = beta
        
        # Calculate RMSE
        results.loc[gauge_id, "RMSE"] = rmse(predictions, targets)
        
        # Calculate relative error
        results.loc[gauge_id, "RelativeError"] = relative_error(predictions, targets)
        
        # Add data quality metrics
        results.loc[gauge_id, "n_observations"] = len(targets)
        results.loc[gauge_id, "n_valid"] = np.sum(~(np.isnan(predictions) | np.isnan(targets)))
        results.loc[gauge_id, "completeness"] = results.loc[gauge_id, "n_valid"] / len(targets)
        
    except Exception as e:
        # Fill with NaNs if calculation fails
        for col in ["NSE", "KGE", "correlation", "alpha", "beta", "RMSE", "RelativeError"]:
            results.loc[gauge_id, col] = np.nan
        results.loc[gauge_id, "error"] = str(e)
    
    return results
