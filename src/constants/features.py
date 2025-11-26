"""Feature descriptions and metadata for HydroATLAS parameters.

This module provides utilities to load and access feature descriptions
from YAML configuration files, including Russian translations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_feature_descriptions(
    config_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load HydroATLAS feature descriptions from YAML configuration.

    Args:
        config_path: Path to feature_descriptions.yaml. If None, uses default
            location at data/config/feature_descriptions.yaml relative to repo root.

    Returns:
        Dictionary mapping feature codes to metadata dictionaries with keys:
            - name_en: English name
            - name_ru: Russian name
            - description: Full description
            - units: Measurement units
            - category: Feature category
            - hydrological_impact: Impact description
            - source: Data source

    Example:
        >>> features = load_feature_descriptions()
        >>> features["for_pc_use"]["name_ru"]
        'Лесистость'
    """
    if config_path is None:
        # Default: assume we're in src/constants/, navigate to data/config/
        repo_root = Path(__file__).parent.parent.parent
        config_path = repo_root / "data" / "config" / "feature_descriptions.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        msg = f"Feature descriptions file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data.get("features", {})


def get_russian_names(
    features: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Extract mapping of feature codes to Russian names.

    Args:
        features: Feature descriptions dictionary. If None, loads from default config.

    Returns:
        Dictionary mapping HydroATLAS codes to Russian names.

    Example:
        >>> russian = get_russian_names()
        >>> russian["prm_pc_use"]
        'Мерзлотные территории'
    """
    if features is None:
        features = load_feature_descriptions()

    return {code: meta["name_ru"] for code, meta in features.items() if "name_ru" in meta}


def get_english_names(
    features: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Extract mapping of feature codes to English names.

    Args:
        features: Feature descriptions dictionary. If None, loads from default config.

    Returns:
        Dictionary mapping HydroATLAS codes to English names.
    """
    if features is None:
        features = load_feature_descriptions()

    return {code: meta["name_en"] for code, meta in features.items() if "name_en" in meta}


def get_feature_by_code(
    code: str,
    features: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Get full metadata for a single feature by code.

    Args:
        code: HydroATLAS feature code (e.g., "for_pc_use").
        features: Feature descriptions dictionary. If None, loads from default config.

    Returns:
        Metadata dictionary for the feature.

    Raises:
        KeyError: If feature code not found.
    """
    if features is None:
        features = load_feature_descriptions()

    if code not in features:
        msg = f"Feature code '{code}' not found in descriptions"
        raise KeyError(msg)

    return features[code]


# Pre-defined list of standard HydroATLAS features used in clustering
STANDARD_FEATURES = [
    "for_pc_use",
    "crp_pc_use",
    "inu_pc_ult",
    "ire_pc_use",
    "lka_pc_use",
    "prm_pc_use",
    "pst_pc_use",
    "cly_pc_uav",
    "slt_pc_uav",
    "snd_pc_uav",
    "kar_pc_use",
    "urb_pc_use",
    "gwt_cm_sav",
    "lkv_mc_usu",
    "rev_mc_usu",
    "slp_dg_sav",
    "sgr_dk_sav",
    "ws_area",
    "ele_mt_uav",
]
