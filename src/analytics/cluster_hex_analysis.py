"""Hybrid clustering analysis with hexagonal spatial aggregation.

This module provides tools for two-stage hybrid clustering workflows:
1. Aggregate watershed clusters to hexagonal grids
2. Map secondary cluster characteristics onto primary cluster regions
3. Calculate cross-cluster contingency tables for regionalization analysis
"""

from __future__ import annotations

from typing import Literal

# Third-party imports (sorted)
import geopandas as gpd
import numpy as np
import pandas as pd


def _ensure_combo_column(
    hexes_with_combos: gpd.GeoDataFrame, combo_col: str
) -> gpd.GeoDataFrame:
    """Ensure the hex dataframe has the requested combo column.

    If missing, try to construct it from dominant geo and hydro columns.
    """
    if combo_col in hexes_with_combos.columns:
        return hexes_with_combos

    # Try to infer likely column names for geo and hydro
    geo_candidate: str | None = None
    hydro_candidate: str | None = None
    # More permissive detection: look for any 'dominant' + 'hydro' pattern
    for col in hexes_with_combos.columns:
        low = str(col).lower()
        if geo_candidate is None and ("dominant_cluster" in low or "geo" in low):
            geo_candidate = col
        if hydro_candidate is None and ("dominant_hydro" in low or "hydro" in low):
            hydro_candidate = col

    if geo_candidate and hydro_candidate:
        out = hexes_with_combos.copy()
        out[combo_col] = (
            out[geo_candidate].astype(str) + "-" + out[hydro_candidate].astype(str)
        )
        return out

    available = ", ".join(map(str, hexes_with_combos.columns))
    raise KeyError(
        f"Column '{combo_col}' not found and could not infer components. "
        f"Available columns: {available}"
    )


def aggregate_clusters_to_hex(
    watersheds: gpd.GeoDataFrame,
    hexes: gpd.GeoDataFrame,
    cluster_col: str,
    method: Literal["dominant", "diversity", "counts"] = "dominant",
    min_watersheds: int = 1,
) -> gpd.GeoDataFrame:
    """Aggregate watershed cluster labels to hex cells.

    Args:
        watersheds: GeoDataFrame with watershed geometries and cluster labels.
        hexes: GeoDataFrame with hexagon geometries.
        cluster_col: Column name containing cluster labels (e.g., "geo_cluster").
        method: Aggregation method:
            - "dominant": Assign the most common cluster to each hex.
            - "diversity": Calculate Shannon diversity index of clusters per hex.
            - "counts": Return count of each cluster type per hex.
        min_watersheds: Minimum number of watersheds required for a hex to be included.

    Returns:
        GeoDataFrame with hexagons and aggregated cluster information.
    """
    if watersheds.crs != hexes.crs:
        msg = "CRS mismatch. Reproject both to the same equal-area CRS."
        raise ValueError(msg)

    if cluster_col not in watersheds.columns:
        msg = f"Column '{cluster_col}' not found in watersheds GeoDataFrame."
        raise KeyError(msg)

    # Use centroids for spatial join
    w_cent = watersheds.copy()
    w_cent["geometry"] = w_cent.geometry.centroid

    # Spatial join watersheds to hexes
    joined = gpd.sjoin(
        w_cent[[cluster_col, "geometry"]],
        hexes.reset_index(names="hex_id"),
        how="inner",
        predicate="within",
    )

    if joined.empty:
        return hexes.assign(count=0, dominant_cluster=np.nan)

    # Count watersheds per hex
    hex_counts = joined.groupby("hex_id").size().rename("count")

    if method == "dominant":
        # Find most common cluster in each hex
        dominant = joined.groupby("hex_id")[cluster_col].agg(
            lambda x: x.mode()[0] if not x.mode().empty else np.nan
        )
        result = hexes.reset_index(names="hex_id").merge(
            dominant.rename("dominant_cluster"), on="hex_id", how="left"
        )
        result = result.merge(hex_counts, on="hex_id", how="left")
        result["count"] = result["count"].fillna(0).astype(int)
        result = result[result["count"] >= min_watersheds].copy()

    elif method == "diversity":
        # Calculate Shannon diversity index
        def shannon_diversity(series):
            counts = series.value_counts()
            proportions = counts / counts.sum()
            return -np.sum(proportions * np.log(proportions))

        diversity = joined.groupby("hex_id")[cluster_col].apply(shannon_diversity)
        result = hexes.reset_index(names="hex_id").merge(
            diversity.rename("cluster_diversity"), on="hex_id", how="left"
        )
        result = result.merge(hex_counts, on="hex_id", how="left")
        result["count"] = result["count"].fillna(0).astype(int)
        result = result[result["count"] >= min_watersheds].copy()

    elif method == "counts":
        # Pivot to get counts of each cluster per hex
        pivot = (
            joined.groupby(["hex_id", cluster_col])
            .size()
            .reset_index(name="cluster_count")
        )
        result = hexes.reset_index(names="hex_id").merge(pivot, on="hex_id", how="left")
        result = result.merge(hex_counts, on="hex_id", how="left")
        result["count"] = result["count"].fillna(0).astype(int)
        result = result[result["count"] >= min_watersheds].copy()

    else:
        msg = f"Unknown method: {method}"
        raise ValueError(msg)

    return result


def map_secondary_cluster_to_hex(
    watersheds: gpd.GeoDataFrame,
    hexes_with_primary: gpd.GeoDataFrame,
    primary_cluster_col: str,
    secondary_cluster_col: str,
    hex_id_col: str = "hex_id",
) -> gpd.GeoDataFrame:
    """Map secondary cluster distribution onto hexes grouped by primary clusters.

    This creates statistics showing how secondary clusters (e.g., hydro)
    are distributed within each primary cluster region (e.g., geo).

    Args:
        watersheds: GeoDataFrame with both cluster columns.
        hexes_with_primary: Hexes already aggregated by primary cluster.
        primary_cluster_col: Column for primary clustering (e.g., "dominant_cluster").
        secondary_cluster_col: Column for secondary clustering (e.g., "hydro_cluster").
        hex_id_col: Column name for hex identifiers.

    Returns:
        Enhanced hex GeoDataFrame with secondary cluster statistics per hex.
    """
    if watersheds.crs != hexes_with_primary.crs:
        msg = "CRS mismatch. Reproject both to the same equal-area CRS."
        raise ValueError(msg)

    # Use centroids for spatial join
    w_cent = watersheds.copy()
    w_cent["geometry"] = w_cent.geometry.centroid

    # Prepare hex dataframe with proper ID column
    hex_df = hexes_with_primary.copy()
    if hex_id_col not in hex_df.columns:
        hex_df = hex_df.reset_index(names=hex_id_col)

    # Spatial join
    joined = gpd.sjoin(
        w_cent[[secondary_cluster_col, "geometry"]],
        hex_df[[hex_id_col, "geometry"]],
        how="inner",
        predicate="within",
    )

    if joined.empty:
        return hexes_with_primary

    # Calculate dominant secondary cluster per hex
    dominant_secondary = joined.groupby(hex_id_col)[secondary_cluster_col].agg(
        lambda x: x.mode()[0] if not x.mode().empty else np.nan
    )

    # Calculate diversity of secondary clusters per hex
    def shannon_diversity(series):
        if len(series) < 2:
            return 0.0
        counts = series.value_counts()
        proportions = counts / counts.sum()
        return float(-np.sum(proportions * np.log(proportions)))

    diversity_secondary = joined.groupby(hex_id_col)[secondary_cluster_col].apply(
        shannon_diversity
    )

    # Count of secondary cluster types per hex
    n_types = joined.groupby(hex_id_col)[secondary_cluster_col].nunique()

    # Merge results
    result = hexes_with_primary.copy()
    if hex_id_col not in result.columns:
        result = result.reset_index(names=hex_id_col)

    result = result.merge(
        dominant_secondary.rename(f"dominant_{secondary_cluster_col}"),
        on=hex_id_col,
        how="left",
    )
    result = result.merge(
        diversity_secondary.rename(f"{secondary_cluster_col}_diversity"),
        on=hex_id_col,
        how="left",
    )
    result = result.merge(
        n_types.rename(f"{secondary_cluster_col}_n_types"), on=hex_id_col, how="left"
    )

    return result


def calculate_cluster_contingency(
    watersheds: gpd.GeoDataFrame,
    hexes_with_clusters: gpd.GeoDataFrame,
    primary_cluster_col: str,
    secondary_cluster_col: str,
    hex_id_col: str = "hex_id",
) -> pd.DataFrame:
    """Create a contingency table of primary vs secondary clusters at hex level.

    Args:
        watersheds: GeoDataFrame with both cluster columns.
        hexes_with_clusters: Hexes with primary cluster assignments.
        primary_cluster_col: Column for primary clustering (in hex GeoDataFrame).
        secondary_cluster_col: Column for secondary clustering (in watersheds).
        hex_id_col: Column name for hex identifiers.

    Returns:
        DataFrame with rows=primary clusters, columns=secondary clusters, values=counts.
    """
    # Use centroids
    w_cent = watersheds.copy()
    w_cent["geometry"] = w_cent.geometry.centroid

    # Prepare hex dataframe with proper ID column
    hex_df = hexes_with_clusters.copy()
    if hex_id_col not in hex_df.columns:
        hex_df = hex_df.reset_index(names=hex_id_col)

    # Spatial join
    joined = gpd.sjoin(
        w_cent[[secondary_cluster_col, "geometry"]],
        hex_df[[hex_id_col, primary_cluster_col, "geometry"]],
        how="inner",
        predicate="within",
    )

    if joined.empty:
        return pd.DataFrame()

    # Create contingency table
    contingency = pd.crosstab(
        joined[primary_cluster_col], joined[secondary_cluster_col], margins=True
    )

    return contingency


def get_cluster_colors(n_colors: int = 10, cmap_name: str = "tab10") -> list[str]:
    """Generate a list of distinct colors for cluster visualization.

    Args:
        n_colors: Number of colors to generate.
        cmap_name: Matplotlib colormap name.

    Returns:
        List of hex color codes.
    """
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i / n_colors) for i in range(n_colors)]
    return [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for r, g, b, _ in colors
    ]


def assign_gauges_to_hybrid_classes(
    gauges: gpd.GeoDataFrame,
    watersheds: gpd.GeoDataFrame,
    hexes_with_combos: gpd.GeoDataFrame,
    combo_col: str = "hybrid_combo",
    hex_id_col: str = "hex_id",
    k_neighbors: int = 5,
) -> gpd.GeoDataFrame:
    """Assign each gauge to a hybrid geo×hydro combination.

    For singleton gauges (alone in hex), blend into the most common class
    among k nearest hexes.

    Args:
        gauges: GeoDataFrame with gauge locations (points).
        watersheds: GeoDataFrame with watershed polygons (must share index with gauges).
        hexes_with_combos: Hex grid with hybrid_combo column.
        combo_col: Column name containing hybrid combination labels (e.g., "G1-H2").
        hex_id_col: Column name for hex identifiers.
        k_neighbors: Number of neighbor hexes to consider for singleton blending.

    Returns:
        GeoDataFrame (gauges) with new column "hybrid_combo" assigned to each gauge.
    """
    if watersheds.crs != hexes_with_combos.crs:
        msg = "CRS mismatch. Reproject both to the same equal-area CRS."
        raise ValueError(msg)

    # Validate presence (or construct) combo column on hex grid
    hexes_with_combos = _ensure_combo_column(hexes_with_combos, combo_col)
    # Ensure combo column now present
    if combo_col not in hexes_with_combos.columns:
        cols_str = ", ".join(map(str, hexes_with_combos.columns))
        raise KeyError(
            f"Failed to create '{combo_col}' after inference. Columns: {cols_str}"
        )

    # Use centroids for spatial join
    ws_cent = watersheds.copy()
    ws_cent["geometry"] = ws_cent.geometry.centroid
    ws_cent["_gauge_idx"] = ws_cent.index

    # Prepare hex dataframe with proper ID column
    hex_df = hexes_with_combos.copy()
    if hex_id_col not in hex_df.columns:
        hex_df = hex_df.reset_index(names=hex_id_col)

    # Spatial join: assign each watershed (gauge) to a hex
    joined = gpd.sjoin(
        ws_cent[["_gauge_idx", "geometry"]],
        hex_df[[hex_id_col, combo_col, "geometry"]],
        how="left",
        predicate="within",
    )

    # Create result dataframe
    result = gauges.copy()
    result["_gauge_idx"] = result.index
    try:
        to_merge = joined[["_gauge_idx", hex_id_col, combo_col]]
    except KeyError as e:  # Provide helpful diagnostics
        raise KeyError(
            "Expected combo column missing after spatial join. "
            f"combo_col='{combo_col}'. Available joined columns: {list(joined.columns)}"
        ) from e

    result = result.merge(
        to_merge,
        on="_gauge_idx",
        how="left",
    )

    # Fallback: missing combo column after merge -> direct spatial join
    if combo_col not in result.columns:
        # Perform spatial join directly on gauge centroids to hex_df
        gj_cent = gauges.copy()
        gj_cent["geometry"] = gj_cent.geometry.centroid
        gj_cent["_gauge_idx"] = gj_cent.index
        direct = gpd.sjoin(
            gj_cent[["_gauge_idx", "geometry"]],
            hex_df[[hex_id_col, combo_col, "geometry"]],
            how="left",
            predicate="within",
        )
        result = result.drop(
            columns=[c for c in result.columns if c not in ["_gauge_idx", "geometry"]]
        )
        result = result.merge(
            direct[["_gauge_idx", hex_id_col, combo_col]],
            on="_gauge_idx",
            how="left",
        )
        if combo_col not in result.columns:
            # Last resort: create empty combo column
            result[combo_col] = np.nan

    # Identify singletons: gauges without assigned hex or in singleton hexes
    # Count gauges per hex
    gauges_per_hex = joined.groupby(hex_id_col)["_gauge_idx"].count()
    singleton_hexes = gauges_per_hex[gauges_per_hex == 1].index.tolist()

    singletons = result[
        result[combo_col].isna() | result[hex_id_col].isin(singleton_hexes)
    ].index.tolist()

    if singletons:
        # For singletons, find nearest k hexes and use most common combo
        hex_centroids = hex_df.copy()
        hex_centroids["geometry"] = hex_centroids.geometry.centroid

        for gauge_id in singletons:
            gauge_point = result.loc[gauge_id, "geometry"]
            if pd.isna(gauge_point):
                continue

            # Calculate distances to all hex centroids
            distances = hex_centroids.geometry.apply(
                lambda x, pt=gauge_point: x.distance(pt)
            )
            nearest_hexes = distances.nsmallest(k_neighbors).index

            # Get combos from nearest hexes
            nearest_combos = hex_df.loc[nearest_hexes, combo_col].dropna()

            if not nearest_combos.empty:
                # Assign most common combo
                most_common = nearest_combos.mode()
                if not most_common.empty:
                    result.loc[gauge_id, combo_col] = most_common.iloc[0]

    # Clean up temporary column
    result = result.drop(columns=["_gauge_idx"])

    return result


def consolidate_hybrid_combinations(
    gauges_with_combos: gpd.GeoDataFrame,
    combo_col: str = "hybrid_combo",
    target_n_classes: int = 15,
    min_gauges_per_class: int = 5,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Consolidate hybrid combinations into a smaller set of meaningful classes.

    Strategy:
    1. Keep the top N most common combinations as separate classes
    2. Merge rare combinations based on geographic cluster (G1-*, G2-*, etc.)
    3. If still too many, merge based on hydrological cluster

    Args:
        gauges_with_combos: GeoDataFrame with hybrid_combo assigned to each gauge.
        combo_col: Column name containing hybrid combination labels.
        target_n_classes: Target number of final classes (~15).
        min_gauges_per_class: Minimum gauges to keep a combination as separate class.

    Returns:
        Tuple of (updated GeoDataFrame with "hybrid_class" column, mapping DataFrame).
    """
    result = gauges_with_combos.copy()

    # Count gauges per combination
    combo_counts = result[combo_col].value_counts()

    # Keep top combinations that meet threshold
    significant_combos = combo_counts[combo_counts >= min_gauges_per_class].head(
        target_n_classes
    )

    # Create mapping dictionary
    combo_mapping = {}

    # Assign significant combos directly
    for combo in significant_combos.index:
        combo_mapping[combo] = combo

    # For rare combinations, group by geo cluster
    rare_combos = combo_counts[~combo_counts.index.isin(significant_combos.index)]

    if not rare_combos.empty:
        for combo in rare_combos.index:
            if pd.isna(combo):
                combo_mapping[combo] = "Unclassified"
                continue

            geo_cluster = combo.split("-")[0]  # Extract G1, G2, etc.

            # Try to find a dominant combo with same geo cluster
            geo_matches = [
                c for c in significant_combos.index if c.startswith(geo_cluster + "-")
            ]

            if geo_matches:
                # Assign to most common geo-matched combo
                combo_mapping[combo] = geo_matches[0]
            else:
                # Group as "Other-{geo_cluster}"
                combo_mapping[combo] = f"Other-{geo_cluster}"

    # Apply mapping
    result["hybrid_class"] = result[combo_col].map(combo_mapping)

    # Handle any remaining NaN
    result["hybrid_class"] = result["hybrid_class"].fillna("Unclassified")

    # Create summary mapping table
    mapping_df = pd.DataFrame(
        [
            {
                "original_combo": k,
                "hybrid_class": v,
                "n_gauges": combo_counts.get(k, 0),
            }
            for k, v in combo_mapping.items()
        ]
    )

    # Add class statistics
    class_counts = result["hybrid_class"].value_counts()
    mapping_df["class_total_gauges"] = mapping_df["hybrid_class"].map(class_counts)

    return result, mapping_df
