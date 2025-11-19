"""HydroATLAS cluster analysis utility functions.

Functions for catchment size categorization, feature filtering,
and cluster interpretation for CAMELS-RU dataset.
"""

import pandas as pd


def categorize_catchment_size(area: float) -> str:
    """Categorize catchment area into size classes.

    Args:
        area: Catchment area in km²

    Returns:
        Size category string with LaTeX formatting
    """
    lim_1, lim_2, lim_3, lim_4, lim_5 = 100, 2000, 10000, 50000, 200000

    if area < lim_1:
        return "a) < 100 $km^2$"
    elif (area >= lim_1) & (area <= lim_2):
        return "b) 100 $km^2$ - 2 000 $km^2$"
    elif (area > lim_2) & (area <= lim_3):
        return "c) 2 000 $km^2$ - 10 000 $km^2$"
    elif (area > lim_3) & (area <= lim_4):
        return "d) 10 000 $km^2$ - 50 000 $km^2$"
    elif (area > lim_4) & (area <= lim_5):
        return "e) 50 000 $km^2$ - 200 000 $km^2$"
    else:
        return "f) > 200 000 $km^2$"


def get_size_categories() -> list[str]:
    """Return ordered list of size category labels.

    Returns:
        List of size category strings
    """
    return [
        "a) < 100 $km^2$",
        "b) 100 $km^2$ - 2 000 $km^2$",
        "c) 2 000 $km^2$ - 10 000 $km^2$",
        "d) 10 000 $km^2$ - 50 000 $km^2$",
        "e) 50 000 $km^2$ - 200 000 $km^2$",
        "f) > 200 000 $km^2$",
    ]


def filter_hydroatlas_features(
    hydro_data: pd.DataFrame,
    gauge_index: pd.Index,
) -> tuple[pd.DataFrame, list[str]]:
    """Filter and select HydroATLAS features based on suffixes.

    Extracts features with specific HydroATLAS suffixes:
    - ult: ultimate (entire upstream watershed)
    - sse: sub-basin spatial extent
    - sav: sub-basin area-weighted average
    - use: upstream spatial extent
    - pva: point value attribute

    Args:
        hydro_data: DataFrame with HydroATLAS attributes
        gauge_index: Index of gauges to filter

    Returns:
        Tuple of (filtered_subset, selected_feature_names)
    """
    suffixes = ["yr", "lt", "av", "se", "pv"]
    prefix = ["u", "s"]

    filtered_tags = {}

    for tag in [
        i
        for i in hydro_data.columns
        if i.split("_")[-1] in [f"{p}{s}" for p in prefix for s in suffixes]
    ]:
        base_tag = "_".join(tag.split("_")[:2])
        if base_tag not in filtered_tags:
            filtered_tags[base_tag] = [tag]
        else:
            filtered_tags[base_tag].append(tag)

    selected_features = [v[0] if len(v) == 1 else v[1] for v in filtered_tags.values()]
    hydro_subset = hydro_data.loc[gauge_index, selected_features].copy()

    return hydro_subset, selected_features


def get_cluster_markers(n_clusters: int = 15) -> list[str]:
    """Return list of matplotlib marker symbols for cluster visualization.

    Args:
        n_clusters: Number of markers needed

    Returns:
        List of marker symbols
    """
    markers = ["o", "s", "^", "v", "<", ">", "D", "P", "X", "*", "h", "H", "8", "p", "d"]
    return markers[:n_clusters]


def get_cluster_colors(n_clusters: int = 15) -> list[str]:
    """Return list of color codes for cluster visualization.

    Args:
        n_clusters: Number of colors needed

    Returns:
        List of hex color codes
    """
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
    ]
    return colors[:n_clusters]


def get_marker_size_corrections() -> dict[str, float]:
    """Return marker size correction factors for visual balance.

    Different marker shapes require size adjustments to appear
    visually consistent.

    Returns:
        Dict mapping marker symbol to size correction factor
    """
    return {
        "o": 1.00,
        "^": 1.15,
        "v": 1.15,
        "<": 1.15,
        ">": 1.15,
        "d": 1.00,
        "p": 1.05,
        "h": 1.05,
        "H": 0.95,
        "8": 1.00,
        "X": 1.10,
        "*": 1.20,
        "D": 0.85,
        "P": 0.90,
        "s": 0.80,
    }


def classify_cluster_improved(
    cluster_id: int,
    cluster_centroids_detailed: pd.DataFrame,
    cluster_centroids_raw_detailed: pd.DataFrame,
    attribute_categories: dict[str, list[str]],
    threshold: float = 0.3,
) -> dict[str, str | dict | list]:
    """Classify cluster by dominant hydrological control.

    Uses standardized z-scores and category-based analysis to determine
    the dominant control type (Climate, Land Cover, Soil, etc.).

    Args:
        cluster_id: Cluster identifier (1-based)
        cluster_centroids_detailed: DataFrame with standardized centroids
        cluster_centroids_raw_detailed: DataFrame with raw centroids
        attribute_categories: Dict mapping category names to feature lists
        threshold: Z-score threshold for feature significance (default 0.3)

    Returns:
        Dict with keys: cluster_type, cluster_name, top_features, rationale
    """
    # Extract series for this cluster - type: ignore needed due to Pyright limitation
    centroid_std: pd.Series = cluster_centroids_detailed.loc[cluster_id]  # type: ignore
    centroid_raw: pd.Series = cluster_centroids_raw_detailed.loc[cluster_id]  # type: ignore

    # Find features with high absolute z-scores
    abs_values: pd.Series = centroid_std.abs()  # type: ignore
    mask = abs_values > threshold
    high_series: pd.Series = abs_values[mask].sort_values(ascending=False)  # type: ignore
    high_indices = high_series.index.tolist()

    # If we have clear dominant features
    if len(high_indices) > 0:
        top_feat = str(high_indices[0])
        z_val = float(centroid_std.loc[top_feat])
        raw_val = float(centroid_raw.loc[top_feat])

        # Determine category
        for category, features in attribute_categories.items():
            if top_feat in features:
                top_feats_dict = {f: float(centroid_std.loc[f]) for f in high_indices[:3]}
                return {
                    "cluster_type": category,
                    "cluster_name": f"{category} Controlled",
                    "top_features": top_feats_dict,
                    "rationale": (
                        f"Primary control: {top_feat} (z={z_val:.2f}, raw={raw_val:.2f})"
                    ),
                }

    # Fallback: category-based scoring
    category_scores: dict[str, float] = {}
    for category, features in attribute_categories.items():
        cat_features = [f for f in features if f in centroid_std.index]
        if cat_features:
            abs_scores: pd.Series = centroid_std.loc[cat_features].abs()  # type: ignore
            category_scores[category] = float(abs_scores.mean())

    if category_scores:
        dominant_cat = max(category_scores.keys(), key=lambda k: category_scores[k])
        top_feats_dict = (
            {f: float(centroid_std.loc[f]) for f in high_indices[:3]}
            if high_indices
            else {}
        )
        return {
            "cluster_type": dominant_cat,
            "cluster_name": f"{dominant_cat} Influenced",
            "top_features": top_feats_dict,
            "rationale": f"Category mean z-score: {category_scores[dominant_cat]:.2f}",
        }

    # Ultimate fallback
    return {
        "cluster_type": "Mixed",
        "cluster_name": "Mixed Controls",
        "top_features": {},
        "rationale": "No clear dominant pattern",
    }


def get_attribute_categories() -> dict[str, list[str]]:
    """Return mapping of attribute categories to HydroATLAS features.

    Returns:
        Dict mapping category names to lists of feature codes
    """
    return {
        "Climate": ["pre_mm_syr", "pet_mm_syr", "aet_mm_syr", "snw_pc_syr", "glc_pc_use"],
        "Land Cover": [
            "for_pc_use",
            "crp_pc_use",
            "pst_pc_use",
            "ire_pc_use",
            "gla_pc_use",
            "prm_pc_use",
            "pac_pc_use",
            "tbi_cl_smj",
        ],
        "Soil": ["cly_pc_sav", "slt_pc_sav", "snd_pc_sav"],
        "Hydrogeology": ["kar_pc_use", "glh_cl_smj"],
        "Cryosphere": ["snw_pc_syr", "glc_pc_use"],
        "Water Bodies": ["lkv_mc_usu", "rev_mc_usu"],
        "Human Impact": ["dor_pc_pva", "urb_pc_use"],
    }


def _get_feature_priority_map() -> dict[str, tuple[int, str, str]]:
    """Return feature priority mapping for cluster naming.

    Priority levels (lower number = higher priority):
    1 - Critical/Rare: Distinctive hydrological controls (permafrost, karst, urban)
    2 - Important: Significant hydrological influences (lakes, reservoirs, snow)
    3 - Moderate: Common descriptors (elevation, cropland)
    4 - Low: Ubiquitous features (forest handled separately, soil texture)

    Returns:
        Dict mapping feature code to (priority, short_name, long_name)
    """
    return {
        # Priority 1: Critical - Rare and highly distinctive features
        "prm_pc_use": (1, "Permafrost", "permafrost-affected"),
        "kar_pc_use": (1, "Karst", "karst-influenced"),
        "urb_pc_use": (1, "Urban", "urbanized"),
        "ire_pc_use": (1, "Irrigated", "irrigated"),
        "gla_pc_use": (1, "Glacial", "glacier-fed"),
        # Priority 2: Important - Significant hydrological controls
        "lka_pc_use": (2, "Lake-regulated", "lake-regulated"),
        "rev_mc_usu": (2, "Reservoir-regulated", "reservoir-regulated"),
        "lkv_mc_usu": (2, "Lake-rich", "lake-rich"),
        "snw_pc_uyr": (2, "Snow-dominated", "snow-dominated"),
        "inu_pc_ult": (2, "Inundation-prone", "wetland-influenced"),
        "gwt_cm_uav": (2, "Deep-GW", "deep groundwater"),
        "gwt_cm_sav": (2, "Deep-GW", "deep groundwater"),
        "slp_dg_uav": (2, "Steep", "steep terrain"),
        "sgr_dk_uav": (2, "High-gradient", "high stream gradient"),
        # Priority 3: Moderate - Common land use and topographic features
        "ele_mt_uav": (3, "Highland", "mountain"),
        "ws_area": (3, "Large", "large watershed"),
        "crp_pc_use": (3, "Cropland", "cropland"),
        "pst_pc_use": (3, "Pasture", "pasture"),
        "pac_pc_use": (3, "Protected", "protected areas"),
        # Priority 4: Low - Ubiquitous or gradual features
        "pre_mm_uyr": (4, "Humid", "high-precipitation"),
        "aet_mm_uyr": (4, "High-ET", "high evapotranspiration"),
        "pet_mm_uyr": (4, "High-PET", "high potential ET"),
        "cly_pc_uav": (4, "Clay-rich", "clay soils"),
        "slt_pc_uav": (4, "Silty", "silt soils"),
        "snd_pc_uav": (4, "Sandy", "sandy soils"),
        # Note: for_pc_use handled separately with extreme-value logic
    }


def _build_primary_components(
    high_features: pd.Series,
    normalized_row: pd.Series,
    raw_row: pd.Series,
    feature_priority: dict[str, tuple[int, str, str]],
) -> list[str]:
    """Build name components from high-value features.

    Args:
        high_features: Series of features above threshold
        normalized_row: Normalized centroid values
        raw_row: Raw centroid values
        feature_priority: Feature priority mapping

    Returns:
        List of name component strings
    """
    name_components: list[str] = []

    # Sort high features by priority then normalized value
    high_with_priority = []
    for feat in high_features.index:
        if feat in feature_priority:
            priority, short_name, _ = feature_priority[feat]
            norm_val: float = normalized_row[feat]  # type: ignore[assignment]
            high_with_priority.append((priority, norm_val, feat, short_name))

    # Handle forest separately - only include if extreme
    if "for_pc_use" in high_features.index:
        forest_raw: float = raw_row["for_pc_use"]  # type: ignore[assignment]
        forest_norm: float = normalized_row["for_pc_use"]  # type: ignore[assignment]
        if forest_raw > 80:
            # Very high forest
            high_with_priority.append((1, forest_norm, "for_pc_use", "Heavily Forested"))
        elif forest_raw < 10:
            # Sparse forest (distinctive in Russia)
            high_with_priority.append((2, forest_norm, "for_pc_use", "Sparsely Forested"))

    high_with_priority.sort(key=lambda x: (x[0], -x[1]))

    # Add primary and secondary characteristics
    for i, (_, _, feat, name) in enumerate(high_with_priority[:3]):
        if i == 0:
            raw_val: float = raw_row[feat]  # type: ignore[assignment]
            component = _format_primary_feature(feat, name, raw_val)
            name_components.append(component)
        elif i == 1 and len(name_components) < 2:
            name_components.append(name)

    return name_components


def _format_primary_feature(feat: str, name: str, raw_val: float) -> str:
    """Format primary feature with value and units.

    Args:
        feat: Feature code
        name: Feature display name
        raw_val: Raw feature value

    Returns:
        Formatted string with name and value
    """
    if feat == "ele_mt_uav":
        return f"{name} ({raw_val:.0f}m)"
    if feat.endswith(("_pc_use", "_pc_uyr", "_pc_ult", "_pc_uav")):
        return f"{name} ({raw_val:.1f}%)"
    if feat == "pre_mm_uyr":
        return f"{name} ({raw_val:.0f}mm/yr)"
    if feat in ("gwt_cm_sav", "gwt_cm_uav"):
        gw_label = "Deep GW" if raw_val > 200 else "Shallow GW"
        return f"{gw_label} ({raw_val:.0f}cm)"
    if feat in ("lkv_mc_usu", "rev_mc_usu"):
        return f"{name} ({raw_val:.0f}Mm³)"
    if feat == "slp_dg_uav":
        return f"{name} ({raw_val:.1f}°)"
    if feat == "sgr_dk_uav":
        return f"{name} ({raw_val:.3f})"
    if feat == "ws_area":
        return f"{name} ({raw_val:.0f}km²)"
    return name


def _build_moderate_components(
    normalized_row: pd.Series,
    raw_row: pd.Series,
) -> list[str]:
    """Build name components from moderate-value features.

    Checks second-tier distinctive features before falling back to forest.
    Only includes forest if extreme (>80% or <20%).

    Args:
        normalized_row: Normalized centroid values
        raw_row: Raw centroid values

    Returns:
        List of name component strings
    """
    # Check distinctive moderate features first
    distinctive_components = _extract_distinctive_features(raw_row)
    if len(distinctive_components) >= 2:
        return distinctive_components[:2]

    # Add topographic descriptors if needed
    if len(distinctive_components) < 2:
        topo_components = _extract_topographic_features(normalized_row, raw_row)
        distinctive_components.extend(topo_components)

    # Only add forest if nothing else found and it's extreme
    if len(distinctive_components) == 0:
        forest_component = _extract_extreme_forest(raw_row)
        if forest_component:
            distinctive_components.append(forest_component)

    return distinctive_components[:2]


def _extract_distinctive_features(raw_row: pd.Series) -> list[str]:
    """Extract distinctive moderate features from raw data.

    Args:
        raw_row: Raw centroid values

    Returns:
        List of distinctive feature descriptions
    """
    components: list[str] = []
    checks = [
        ("lka_pc_use", "Lake-influenced", 3.0),
        ("crp_pc_use", "Agricultural", 25.0),
        ("slp_dg_uav", "Hilly", 3.0),
        ("kar_pc_use", "Karst", 10.0),
        ("urb_pc_use", "Urban", 2.0),
        ("inu_pc_ult", "Wetland", 5.0),
    ]

    for feat, label, threshold in checks:
        if feat in raw_row:
            feat_val: float = raw_row[feat]  # type: ignore[assignment]
            if feat_val > threshold:
                components.append(f"{label} ({feat_val:.1f})")
                if len(components) >= 2:
                    break

    return components


def _extract_topographic_features(
    normalized_row: pd.Series,
    raw_row: pd.Series,
) -> list[str]:
    """Extract topographic feature descriptions.

    Args:
        normalized_row: Normalized centroid values
        raw_row: Raw centroid values

    Returns:
        List of topographic feature descriptions
    """
    components: list[str] = []

    elev_norm: float = normalized_row.get("ele_mt_uav", 0.5)  # type: ignore[assignment]
    if 0.35 < elev_norm < 0.65:
        elev_val: float = raw_row.get("ele_mt_uav", 0)  # type: ignore[assignment]
        components.append(_classify_elevation(elev_val))

    if len(components) < 2:
        precip_norm: float = normalized_row.get("pre_mm_uyr", 0.5)  # type: ignore[assignment]
        if 0.35 < precip_norm < 0.65:
            precip_val: float = raw_row.get("pre_mm_uyr", 0)  # type: ignore[assignment]
            precip_label = _classify_precipitation(precip_val)
            if precip_label:
                components.append(precip_label)

    return components


def _extract_extreme_forest(raw_row: pd.Series) -> str | None:
    """Extract forest descriptor only if extreme values.

    Args:
        raw_row: Raw centroid values

    Returns:
        Forest description or None
    """
    if "for_pc_use" not in raw_row:
        return None

    forest_val: float = raw_row["for_pc_use"]  # type: ignore[assignment]
    if forest_val > 80:
        return f"Heavily Forested ({forest_val:.1f}%)"
    if forest_val < 20:
        return f"Sparsely Forested ({forest_val:.1f}%)"
    return None


def _classify_elevation(elev_val: float) -> str:
    """Classify elevation into category.

    Args:
        elev_val: Elevation in meters

    Returns:
        Elevation category string
    """
    if elev_val < 200:
        return f"Lowland ({elev_val:.0f}m)"
    if elev_val < 500:
        return f"Low-elevation ({elev_val:.0f}m)"
    return f"Mid-elevation ({elev_val:.0f}m)"


def _classify_precipitation(precip_val: float) -> str:
    """Classify precipitation into category.

    Args:
        precip_val: Precipitation in mm/yr

    Returns:
        Precipitation category string or empty if moderate
    """
    if precip_val < 400:
        return f"Semi-arid ({precip_val:.0f}mm/yr)"
    if precip_val < 600:
        return f"Moderate ({precip_val:.0f}mm/yr)"
    return ""


def _build_fallback_components(raw_row: pd.Series) -> list[str]:
    """Build fallback name components from elevation and precipitation.

    Args:
        raw_row: Raw centroid values

    Returns:
        List of name component strings
    """
    name_components: list[str] = []

    elev_val: float = raw_row.get("ele_mt_uav", 0)  # type: ignore[assignment]
    precip_val: float = raw_row.get("pre_mm_uyr", 0)  # type: ignore[assignment]

    # Elevation classification
    if elev_val > 1000:
        name_components.append(f"Highland ({elev_val:.0f}m)")
    elif elev_val > 500:
        name_components.append(f"Upland ({elev_val:.0f}m)")
    else:
        name_components.append(f"Lowland ({elev_val:.0f}m)")

    # Precipitation classification
    if precip_val > 700:
        name_components.append(f"Humid ({precip_val:.0f}mm/yr)")
    elif precip_val < 400:
        name_components.append(f"Semi-arid ({precip_val:.0f}mm/yr)")
    else:
        name_components.append(f"Moderate ({precip_val:.0f}mm/yr)")

    return name_components


def interpret_cluster_from_data(
    normalized_row: pd.Series,
    raw_row: pd.Series,
    feature_descriptions: dict[str, dict[str, str]],
    high_threshold: float = 0.65,
    low_threshold: float = 0.35,
) -> str:
    """Generate data-driven cluster name based on dominant characteristics.

    Uses feature_descriptions to provide contextually accurate naming aligned
    with hydrological impact and category definitions.

    Args:
        normalized_row: 0-1 normalized centroid values
        raw_row: Raw (original scale) centroid values
        feature_descriptions: Dict of feature metadata with hydrological context
        high_threshold: Threshold for high normalized values (default 0.65)
        low_threshold: Threshold for low normalized values (default 0.35)

    Returns:
        Descriptive cluster name derived from actual data patterns
    """
    # Identify dominant high features
    high_features = normalized_row[normalized_row > high_threshold].sort_values(
        ascending=False
    )

    # Feature interpretation map with prioritization
    feature_priority = _get_feature_priority_map()

    # Try primary naming strategy
    name_components = _build_primary_components(
        high_features, normalized_row, raw_row, feature_priority
    )

    # Fallback to moderate features
    if len(name_components) == 0:
        name_components = _build_moderate_components(normalized_row, raw_row)

    # Final fallback: elevation + precipitation
    if len(name_components) == 0:
        name_components = _build_fallback_components(raw_row)

    return " / ".join(name_components[:2])
