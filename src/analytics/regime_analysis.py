"""DOY-based hydrological regime analysis.

This module provides functions for extracting hydrological regime characteristics
from seasonal discharge patterns using day-of-year (DOY) based analysis,
including peak timing, seasonal ratios, and regime classification.
"""

from __future__ import annotations

from collections import OrderedDict
import logging
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Seasonal DOY ranges (0-indexed for array slicing)
# Winter: Dec 1 - Feb 28/29 (DOY 335-365, 1-59) → indices 334-365, 0-58
# Spring: Mar 1 - May 31 (DOY 60-151) → indices 59-150
# Summer: Jun 1 - Aug 31 (DOY 152-243) → indices 151-242
# Autumn: Sep 1 - Nov 30 (DOY 244-334) → indices 243-333

WINTER_INDICES = list(range(334, 366)) + list(range(59))
SPRING_INDICES = list(range(59, 151))
SUMMER_INDICES = list(range(151, 243))
AUTUMN_INDICES = list(range(243, 334))


def extract_peak_timing(pattern: np.ndarray) -> tuple[int, float]:
    """Extract peak day-of-year and value from seasonal pattern.

    Args:
        pattern: Array of 366 daily values (normalized or raw).

    Returns:
        Tuple of (peak_doy, peak_value) where peak_doy is 1-indexed (1-366).
    """
    peak_idx = int(np.argmax(pattern))
    peak_doy = peak_idx + 1  # Convert to 1-indexed DOY
    peak_value = float(pattern[peak_idx])
    return peak_doy, peak_value


def calculate_seasonal_ratios(pattern: np.ndarray) -> dict[str, float]:
    """Calculate seasonal mean ratios from 366-day pattern.

    Args:
        pattern: Array of 366 daily values.

    Returns:
        Dictionary with keys: winter_ratio, spring_ratio, summer_ratio, autumn_ratio.
        Values are normalized so they sum to 1.0.
    """
    winter_mean = float(np.mean(pattern[WINTER_INDICES]))
    spring_mean = float(np.mean(pattern[SPRING_INDICES]))
    summer_mean = float(np.mean(pattern[SUMMER_INDICES]))
    autumn_mean = float(np.mean(pattern[AUTUMN_INDICES]))

    total = winter_mean + spring_mean + summer_mean + autumn_mean
    if total <= 0:
        return {
            "winter_ratio": 0.0,
            "spring_ratio": 0.0,
            "summer_ratio": 0.0,
            "autumn_ratio": 0.0,
        }

    return {
        "winter_ratio": winter_mean / total,
        "spring_ratio": spring_mean / total,
        "summer_ratio": summer_mean / total,
        "autumn_ratio": autumn_mean / total,
    }


def classify_regime_type(
    peak_doy: int,
    seasonal_ratios: dict[str, float],
    cv: float,
    language: Literal["ru", "en"] = "ru",
) -> str:
    """Classify hydrological regime type based on peak timing and variability.

    Args:
        peak_doy: Day-of-year of peak discharge (1-366).
        seasonal_ratios: Dictionary with winter/spring/summer/autumn ratios.
        cv: Coefficient of variation of the pattern.
        language: Output language ('ru' for Russian, 'en' for English).

    Returns:
        Regime type string with optional variability modifier.
    """
    spring_ratio = seasonal_ratios["spring_ratio"]
    summer_ratio = seasonal_ratios["summer_ratio"]

    # Russian regime names
    regime_names_ru = {
        "spring_flood": "Весеннее половодье",
        "spring_summer": "Весенне-летний",
        "summer_flood": "Летнее половодье",
        "summer_autumn": "Летне-осенний",
        "autumn_winter": "Осенне-зимний",
        "winter_baseflow": "Зимний (межень)",
        "low_variability": "(маловодный)",
        "high_variability": "(паводочный)",
    }

    # English regime names
    regime_names_en = {
        "spring_flood": "Spring flood-dominated",
        "spring_summer": "Spring-summer",
        "summer_flood": "Summer flood-dominated",
        "summer_autumn": "Summer-autumn",
        "autumn_winter": "Autumn-winter (rainfall)",
        "winter_baseflow": "Winter baseflow",
        "low_variability": "(low variability)",
        "high_variability": "(flashy)",
    }

    names = regime_names_ru if language == "ru" else regime_names_en

    # Classify based on peak DOY
    if 60 <= peak_doy <= 151:  # Spring peak (Mar-May)
        regime_type = (
            names["spring_flood"] if spring_ratio > 0.40 else names["spring_summer"]
        )
    elif 152 <= peak_doy <= 243:  # Summer peak (Jun-Aug)
        regime_type = (
            names["summer_flood"] if summer_ratio > 0.35 else names["summer_autumn"]
        )
    elif 244 <= peak_doy <= 334:  # Autumn peak (Sep-Nov)
        regime_type = names["autumn_winter"]
    else:  # Winter peak (Dec-Feb) - rare, indicates groundwater/regulated
        regime_type = names["winter_baseflow"]

    # Add variability modifier
    if cv < 0.3:
        regime_type += f" {names['low_variability']}"
    elif cv > 0.7:
        regime_type += f" {names['high_variability']}"

    return regime_type


def build_regime_dataframe(
    cluster_patterns: dict[str, pd.DataFrame],
    language: Literal["ru", "en"] = "ru",
) -> pd.DataFrame:
    """Build DataFrame with regime characteristics for all clusters.

    Args:
        cluster_patterns: OrderedDict mapping cluster names to DataFrames.
            Each DataFrame has rows=days (366), columns=gauge_ids.
        language: Output language for regime type names.

    Returns:
        DataFrame with columns:
            - cluster_name: Cluster identifier
            - peak_doy: Day of year of peak discharge (1-366)
            - peak_value: Peak discharge value (normalized)
            - winter_ratio, spring_ratio, summer_ratio, autumn_ratio
            - regime_type: Classified regime name
            - cv: Coefficient of variation
            - n_gauges: Number of gauges in cluster
    """
    regime_data = []

    for cluster_name, cluster_df in cluster_patterns.items():
        # Compute cluster median pattern
        cluster_median = np.asarray(cluster_df.median(axis=1).values)

        # Extract characteristics
        peak_doy, peak_value = extract_peak_timing(cluster_median)
        seasonal_ratios = calculate_seasonal_ratios(cluster_median)

        # Calculate CV
        mean_val = float(np.mean(cluster_median))
        cv = float(np.std(cluster_median) / mean_val) if mean_val > 0 else 0.0

        # Classify regime
        regime_type = classify_regime_type(peak_doy, seasonal_ratios, cv, language)

        regime_data.append(
            {
                "cluster_name": cluster_name,
                "peak_doy": peak_doy,
                "peak_value": peak_value,
                "winter_ratio": seasonal_ratios["winter_ratio"],
                "spring_ratio": seasonal_ratios["spring_ratio"],
                "summer_ratio": seasonal_ratios["summer_ratio"],
                "autumn_ratio": seasonal_ratios["autumn_ratio"],
                "regime_type": regime_type,
                "cv": cv,
                "n_gauges": len(cluster_df.columns),
            }
        )

    regime_df = pd.DataFrame(regime_data)

    log.info("Built regime DataFrame for %d clusters", len(regime_df))

    return regime_df


def group_patterns_by_cluster(
    q_df_normalized: pd.DataFrame,
    gauge_cluster_mapping: pd.Series,
    cluster_col: str = "cluster_id",
) -> OrderedDict[str, pd.DataFrame]:
    """Group normalized discharge patterns by cluster assignment.

    Args:
        q_df_normalized: Normalized patterns (rows=days, columns=gauges).
        gauge_cluster_mapping: Series or DataFrame with cluster assignments.
            Index should be gauge_id, values should be cluster labels.
        cluster_col: Column name if gauge_cluster_mapping is a DataFrame.

    Returns:
        OrderedDict mapping cluster names to DataFrames of member patterns.
    """
    if isinstance(gauge_cluster_mapping, pd.DataFrame):
        cluster_labels = gauge_cluster_mapping[cluster_col]
    else:
        cluster_labels = gauge_cluster_mapping

    # Get unique clusters sorted
    unique_clusters = sorted(cluster_labels.unique())

    cluster_patterns = OrderedDict()
    for cluster_id in unique_clusters:
        member_gauges = cluster_labels[cluster_labels == cluster_id].index.tolist()
        # Filter to gauges present in q_df
        valid_gauges = [g for g in member_gauges if g in q_df_normalized.columns]

        if valid_gauges:
            cluster_df = q_df_normalized[valid_gauges].copy()
            cluster_df.index = pd.date_range(start="2000-01-01", periods=len(cluster_df))
            cluster_patterns[f"Кластер {cluster_id}"] = cluster_df

    log.info("Grouped patterns into %d clusters", len(cluster_patterns))

    return cluster_patterns


def calculate_regime_statistics(
    pattern: np.ndarray,
) -> dict[str, float]:
    """Calculate comprehensive statistics for a discharge pattern.

    Args:
        pattern: Array of 366 daily values.

    Returns:
        Dictionary with statistical measures.
    """
    peak_doy, peak_value = extract_peak_timing(pattern)
    seasonal_ratios = calculate_seasonal_ratios(pattern)

    mean_val = float(np.mean(pattern))
    std_val = float(np.std(pattern))
    cv = std_val / mean_val if mean_val > 0 else 0.0

    # Find trough (minimum)
    trough_idx = int(np.argmin(pattern))
    trough_doy = trough_idx + 1
    trough_value = float(pattern[trough_idx])

    # Peak-to-trough ratio (flashiness indicator)
    ptr = peak_value / trough_value if trough_value > 0 else float("inf")

    return {
        "peak_doy": peak_doy,
        "peak_value": peak_value,
        "trough_doy": trough_doy,
        "trough_value": trough_value,
        "peak_to_trough_ratio": ptr,
        "mean": mean_val,
        "std": std_val,
        "cv": cv,
        **seasonal_ratios,
    }
