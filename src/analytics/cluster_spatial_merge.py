"""Spatial consolidation of cluster combinations.

Merges isolated hexagons with their dominant neighbors to reduce
the number of unique cluster combinations while preserving spatial coherence.
"""

from __future__ import annotations

from collections import Counter
import logging
from typing import Any

import geopandas as gpd

logger = logging.getLogger(__name__)


def create_spatial_adjacency(
    hex_grid: gpd.GeoDataFrame,
    hex_id_col: str = "hex_id",
) -> dict[Any, set[Any]]:
    """Create spatial adjacency graph for hexagons.

    Args:
        hex_grid: GeoDataFrame with hexagon geometries.
        hex_id_col: Column name containing unique hex IDs.

    Returns:
        Dictionary mapping hex_id -> set of neighboring hex_ids.
    """
    if hex_id_col not in hex_grid.columns:
        msg = f"Column '{hex_id_col}' not found in hex_grid."
        raise KeyError(msg)

    adjacency = {}

    for _idx, row in hex_grid.iterrows():
        hex_id = row[hex_id_col]
        geom = row.geometry

        # Find neighbors using touches() predicate
        neighbors = hex_grid[hex_grid.geometry.touches(geom)]
        neighbor_ids = set(neighbors[hex_id_col].values)

        adjacency[hex_id] = neighbor_ids

    return adjacency


def _get_minority_classes(
    result: gpd.GeoDataFrame,
    new_col: str,
    min_class_size: int,
    n_classes: int,
    target_classes: int,
) -> list[Any]:
    """Identify minority classes that should be merged.

    Args:
        result: GeoDataFrame with cluster labels.
        new_col: Column containing cluster labels.
        min_class_size: Minimum hex count threshold.
        n_classes: Current number of unique classes.
        target_classes: Target number of classes.

    Returns:
        List of class labels to merge (smallest classes first).
    """
    class_counts = result[new_col].value_counts()
    minority_classes = class_counts[class_counts < min_class_size].index.tolist()

    if not minority_classes:
        # No small classes, aggressively merge smallest classes
        # Merge enough classes to make progress toward target
        n_to_merge = max(1, (n_classes - target_classes) // 2)
        smallest_classes = class_counts.nsmallest(n_to_merge).index.tolist()
        minority_classes = smallest_classes

    return minority_classes


def _reassign_minority_hex(
    result: gpd.GeoDataFrame,
    hex_id: Any,
    minority_class: Any,
    adjacency: dict[Any, set[Any]],
    hex_to_idx: dict[Any, Any],
    new_col: str,
) -> bool:
    """Reassign a single minority hex to dominant neighbor class.

    Returns:
        True if reassignment was made, False otherwise.
    """
    # Get neighbors
    neighbor_ids = adjacency.get(hex_id, set())
    if not neighbor_ids:
        return False

    # Get neighbor classes (excluding self)
    neighbor_classes = []
    for n_id in neighbor_ids:
        if n_id not in hex_to_idx:
            continue
        n_idx = hex_to_idx[n_id]
        n_class = result.at[n_idx, new_col]
        if n_class != minority_class:
            neighbor_classes.append(n_class)

    if not neighbor_classes:
        return False

    # Find most common neighbor class
    dominant_neighbor = Counter(neighbor_classes).most_common(1)[0][0]

    # Reassign hex to dominant neighbor class
    idx = hex_to_idx[hex_id]
    result.at[idx, new_col] = dominant_neighbor
    return True


def consolidate_clusters_spatially(
    hex_grid: gpd.GeoDataFrame,
    cluster_col: str,
    adjacency: dict[Any, set[Any]],
    target_classes: int,
    hex_id_col: str = "hex_id",
    min_class_size: int = 3,
    max_iterations: int = 50,
) -> gpd.GeoDataFrame:
    """Consolidate cluster combinations by merging isolated hexagons.

    Algorithm:
    1. Identify minority classes (< min_class_size hexes).
    2. For each minority hex, find dominant neighbor class.
    3. Reassign hex to dominant neighbor class.
    4. Repeat until target number of classes reached or no changes.

    Args:
        hex_grid: GeoDataFrame with hexagon geometries and cluster labels.
        cluster_col: Column name containing cluster labels to consolidate.
        adjacency: Spatial adjacency graph (from create_spatial_adjacency).
        target_classes: Target number of unique classes after consolidation.
        hex_id_col: Column name containing unique hex IDs.
        min_class_size: Minimum number of hexes per class (smaller classes merged first).
        max_iterations: Maximum consolidation iterations to prevent infinite loops.

    Returns:
        GeoDataFrame with new column "{cluster_col}_consolidated" containing
        reduced cluster labels.
    """
    if cluster_col not in hex_grid.columns:
        msg = f"Column '{cluster_col}' not found in hex_grid."
        raise KeyError(msg)

    if hex_id_col not in hex_grid.columns:
        msg = f"Column '{hex_id_col}' not found in hex_grid."
        raise KeyError(msg)

    # Create working copy
    result = hex_grid.copy()
    new_col = f"{cluster_col}_consolidated"
    result[new_col] = result[cluster_col].copy()

    # Create hex_id -> index mapping for fast lookup
    hex_to_idx = {row[hex_id_col]: idx for idx, row in result.iterrows()}

    iteration = 0
    n_classes = result[new_col].nunique()

    logger.info(f"Starting consolidation: {n_classes} classes → target {target_classes}")

    while n_classes > target_classes and iteration < max_iterations:
        iteration += 1

        # Identify minority classes
        minority_classes = _get_minority_classes(
            result, new_col, min_class_size, n_classes, target_classes
        )

        logger.info(
            f"Iteration {iteration}: {n_classes} classes, "
            f"merging {len(minority_classes)} minority classes"
        )

        # Track changes
        changes_made = 0

        # Process each minority class
        for minority_class in minority_classes:
            # Get all hexes in this class
            minority_hexes = result[result[new_col] == minority_class][hex_id_col].values

            for hex_id in minority_hexes:
                reassigned = _reassign_minority_hex(
                    result, hex_id, minority_class, adjacency, hex_to_idx, new_col
                )
                if reassigned:
                    changes_made += 1

        # Update class count
        n_classes = result[new_col].nunique()

        if changes_made == 0:
            logger.info(f"No changes made in iteration {iteration}, stopping early.")
            break

    logger.info(
        f"Consolidation complete after {iteration} iterations: "
        f"{n_classes} classes (target: {target_classes})"
    )

    # Rename classes to C1, C2, ..., CN for clarity
    final_classes = result[new_col].value_counts().sort_values(ascending=False)
    class_mapping = {old: f"C{i + 1}" for i, old in enumerate(final_classes.index)}
    result[new_col] = result[new_col].map(class_mapping)

    return result
