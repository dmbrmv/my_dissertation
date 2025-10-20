"""Numeric plots and cluster interpretation for hydrological analysis."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def interpret_cluster_from_hydroatlas(
    normalized_row: pd.Series,
    raw_row: pd.Series,
    high_threshold: float = 0.70,
    moderate_threshold: float = 0.45,
    low_threshold: float = 0.30,
) -> str:
    """Generate data-driven cluster name based on HydroATLAS characteristics.

    Prioritizes distinctive hydrological features over common land cover patterns.

    Args:
        normalized_row: 0-1 normalized centroid values
        raw_row: Raw (original scale) centroid values
        high_threshold: Threshold for high normalized values (default 0.70)
        moderate_threshold: Threshold for moderate values (default 0.45)
        low_threshold: Threshold for low normalized values (default 0.30)

    Returns:
        Descriptive cluster name. Format: "Primary / Secondary" with units.
    """
    # Check tier 1 distinctive features
    tier1_result = _check_tier1_features(normalized_row, raw_row, high_threshold)
    if tier1_result:
        return tier1_result

    # Check elevation-based classification
    elevation_result = _check_elevation_features(
        normalized_row, raw_row, high_threshold, moderate_threshold, low_threshold
    )
    if elevation_result:
        return elevation_result

    # Check tier 2 topography/hydrology features
    tier2_result = _check_tier2_features(normalized_row, raw_row, high_threshold)
    if tier2_result:
        return tier2_result

    return "Mixed / Varied characteristics"


def _check_tier1_features(
    normalized_row: pd.Series, raw_row: pd.Series, high_threshold: float
) -> str | None:
    """Check tier 1 extreme/distinctive features."""
    tier1 = {
        "prm_pc_use": "Permafrost-dominated",
        "kar_pc_use": "Karst",
        "lka_pc_use": "Lake-regulated",
        "rev_mc_usu": "Reservoir-regulated",
        "inu_pc_ult": "Wetland",
    }

    for feat, label in tier1.items():
        if feat not in normalized_row.index:
            continue
        if normalized_row[feat] <= high_threshold:
            continue

        raw_val = raw_row[feat]
        if feat.endswith(("_pc_use", "_pc_ult")):
            return f"{label} ({raw_val:.1f}%)"
        if feat.endswith("_mc_usu"):
            return f"{label} ({raw_val:.0f} Mm³)"
        return label

    return None


def _check_elevation_features(
    normalized_row: pd.Series,
    raw_row: pd.Series,
    high_threshold: float,
    moderate_threshold: float,
    low_threshold: float,
) -> str | None:
    """Check elevation-based features."""
    if "ele_mt_uav" not in normalized_row.index:
        return None

    elev_norm = normalized_row["ele_mt_uav"]
    elev_raw = raw_row["ele_mt_uav"]

    primary = _get_elevation_label(
        elev_norm, elev_raw, high_threshold, moderate_threshold, low_threshold
    )
    if not primary:
        return None

    secondary = _find_secondary_feature(normalized_row, raw_row, high_threshold)
    if secondary:
        return f"{primary} / {secondary}"
    return primary


def _get_elevation_label(
    elev_norm: float,
    elev_raw: float,
    high_threshold: float,
    moderate_threshold: float,
    low_threshold: float,
) -> str:
    """Get elevation classification label."""
    if elev_norm > high_threshold:
        return f"Highland ({elev_raw:.0f}m)"
    if elev_norm > moderate_threshold:
        return f"Mid-elevation ({elev_raw:.0f}m)"
    if elev_norm < low_threshold:
        return f"Lowland ({elev_raw:.0f}m)"
    return ""


def _check_tier2_features(
    normalized_row: pd.Series, raw_row: pd.Series, high_threshold: float
) -> str | None:
    """Check tier 2 topography/hydrology features."""
    tier2 = {
        "slp_dg_uav": "Steep",
        "sgr_dk_sav": "High-gradient",
        "gwt_cm_sav": "Deep-GW",
        "ws_area": "Large-basin",
    }

    for feat, label in tier2.items():
        if feat not in normalized_row.index:
            continue
        if normalized_row[feat] <= high_threshold:
            continue

        raw_val = raw_row[feat]
        primary_name = _format_feature_value(feat, label, raw_val)
        secondary = _find_soil_feature(normalized_row, raw_row, high_threshold)
        if secondary:
            return f"{primary_name} / {secondary}"
        return primary_name

    return None


def _find_secondary_feature(
    normalized_row: pd.Series, raw_row: pd.Series, threshold: float
) -> str | None:
    """Find secondary distinctive feature excluding elevation."""
    features = {
        "slp_dg_uav": "Steep",
        "cly_pc_uav": "Clay-rich",
        "snd_pc_uav": "Sandy",
        "gwt_cm_sav": "Deep-GW",
    }

    for feat, label in features.items():
        if feat not in normalized_row.index:
            continue
        if normalized_row[feat] <= threshold:
            continue

        raw_val = raw_row[feat]
        return _format_feature_value(feat, label, raw_val)

    return None


def _find_soil_feature(
    normalized_row: pd.Series, raw_row: pd.Series, threshold: float
) -> str | None:
    """Find dominant soil feature."""
    soils = {"cly_pc_uav": "Clay-rich", "snd_pc_uav": "Sandy", "slt_pc_uav": "Silty"}

    for feat, label in soils.items():
        if feat not in normalized_row.index:
            continue
        if normalized_row[feat] <= threshold:
            continue

        raw_val = raw_row[feat]
        return f"{label} ({raw_val:.1f}%)"

    return None


def _format_feature_value(feat: str, label: str, raw_val: float) -> str:
    """Format feature value with appropriate units."""
    if feat == "slp_dg_uav":
        return f"{label} ({raw_val:.1f}°)"
    if feat == "gwt_cm_sav":
        return f"{label} ({raw_val:.0f}cm)"
    if feat == "ws_area":
        return f"{label} ({raw_val:.0f}km²)"
    if feat.endswith(("_pc_use", "_pc_ult", "_pc_uav")):
        return f"{label} ({raw_val:.1f}%)"
    return label
