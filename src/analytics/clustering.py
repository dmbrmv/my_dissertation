"""Hierarchical clustering for hydrological catchment analysis.

This module provides functions for performing and validating hierarchical
clustering on physiographic and hydrological data, including preprocessing,
validation metrics, and centroid computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

from src.utils.logger import setup_logger

if TYPE_CHECKING:
    import geopandas as gpd


log = setup_logger(__name__)


def perform_hierarchical_clustering(
    data: pd.DataFrame,
    n_clusters: int,
    method: str = "ward",
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    """Perform hierarchical clustering using scipy linkage.

    Args:
        data: DataFrame with features (rows=samples, columns=features).
        n_clusters: Number of clusters to extract.
        method: Linkage method ('ward', 'complete', 'average', 'single').
        metric: Distance metric ('euclidean', 'cosine', 'manhattan').

    Returns:
        Tuple of (cluster_labels, linkage_matrix).

    Example:
        >>> labels, Z = perform_hierarchical_clustering(geo_scaled, n_clusters=9)
    """
    values = data.values if isinstance(data, pd.DataFrame) else data
    linkage_matrix = linkage(values, method=method, metric=metric)
    labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")

    log.info(
        "Hierarchical clustering: %d clusters, method=%s, metric=%s",
        n_clusters,
        method,
        metric,
    )

    return labels, linkage_matrix


def calculate_cluster_validation(
    data: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Calculate clustering validation metrics.

    Args:
        data: Feature matrix (rows=samples, columns=features).
        labels: Cluster labels for each sample.

    Returns:
        Dictionary with:
            - silhouette_avg: Average silhouette score
            - silhouette_samples: Per-sample silhouette scores
            - calinski_harabasz: Calinski-Harabasz index
            - davies_bouldin: Davies-Bouldin index
    """
    values = data.values if isinstance(data, pd.DataFrame) else data

    sil_avg = silhouette_score(values, labels)
    sil_samples = silhouette_samples(values, labels)
    ch_score = calinski_harabasz_score(values, labels)
    db_score = davies_bouldin_score(values, labels)

    log.info(
        "Validation metrics: silhouette=%.3f, CH=%.1f, DB=%.3f",
        sil_avg,
        ch_score,
        db_score,
    )

    return {
        "silhouette_avg": sil_avg,
        "silhouette_samples": sil_samples,
        "calinski_harabasz": ch_score,
        "davies_bouldin": db_score,
    }


def smooth_discharge_patterns(
    q_df: pd.DataFrame,
    median_window: int = 5,
    savgol_window: int = 11,
    savgol_polyorder: int = 3,
    spike_threshold_sigma: float = 3.0,
) -> pd.DataFrame:
    """Remove spikes and smooth seasonal discharge patterns.

    Applies median filter for spike detection, interpolates anomalies,
    then applies Savitzky-Golay filter for smoothing.

    Args:
        q_df: DataFrame with discharge patterns (rows=days, columns=gauges).
        median_window: Window size for median filter spike detection.
        savgol_window: Window size for Savitzky-Golay filter.
        savgol_polyorder: Polynomial order for Savitzky-Golay filter.
        spike_threshold_sigma: Sigma threshold for spike detection.

    Returns:
        Smoothed DataFrame with same shape as input.
    """
    q_smoothed = q_df.copy()

    for col in q_smoothed.columns:
        series = q_smoothed[col].values.copy()

        # Median filter for spike detection
        smoothed = median_filter(series, size=median_window, mode="wrap")
        residuals = np.abs(series - smoothed)
        threshold = residuals.std() * spike_threshold_sigma
        spike_mask = residuals > threshold

        # Interpolate spikes
        if spike_mask.any():
            valid_indices = np.where(~spike_mask)[0]
            spike_indices = np.where(spike_mask)[0]
            if len(valid_indices) > 1:
                series[spike_indices] = np.interp(
                    spike_indices, valid_indices, series[valid_indices]
                )

        # Savitzky-Golay smoothing
        series = savgol_filter(
            series, window_length=savgol_window, polyorder=savgol_polyorder, mode="wrap"
        )
        q_smoothed[col] = series

    log.info(
        "Smoothed %d discharge patterns: median_window=%d, savgol_window=%d",
        len(q_smoothed.columns),
        median_window,
        savgol_window,
    )

    return q_smoothed


def normalize_seasonal_patterns(q_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize each column (gauge) to [0, 1] range.

    Args:
        q_df: DataFrame with discharge patterns.

    Returns:
        Normalized DataFrame with values in [0, 1].
    """
    q_min = q_df.min()
    q_max = q_df.max()
    q_range = q_max - q_min

    # Avoid division by zero for constant series
    q_range = q_range.replace(0, 1)

    return (q_df - q_min) / q_range


def prepare_clustering_features(
    q_df_normalized: pd.DataFrame,
    gauges: gpd.GeoDataFrame,
    include_coords: bool = True,
) -> pd.DataFrame:
    """Prepare features for hydrological clustering.

    Transposes normalized patterns and optionally adds spatial coordinates.

    Args:
        q_df_normalized: Normalized patterns (rows=days, columns=gauges).
        gauges: GeoDataFrame with gauge locations.
        include_coords: Whether to add lat/lon as features.

    Returns:
        DataFrame ready for clustering (rows=gauges, columns=features).
    """
    # Transpose: rows=gauges, columns=days
    q_clust = q_df_normalized.T.copy()

    if include_coords:
        for gauge_id in q_clust.index:
            if gauge_id in gauges.index:
                geom = gauges.loc[gauge_id, "geometry"]
                q_clust.loc[gauge_id, "lat"] = geom.y
                q_clust.loc[gauge_id, "lon"] = geom.x

    # Drop rows with NaN
    q_clust = q_clust.dropna()

    log.info(
        "Prepared clustering features: %d gauges x %d features",
        len(q_clust),
        q_clust.shape[1],
    )

    return q_clust


def compute_cluster_centroids(
    data: pd.DataFrame,
    labels: np.ndarray,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Compute cluster centroids (mean feature values per cluster).

    Args:
        data: Feature DataFrame (rows=samples, columns=features).
        labels: Cluster labels for each sample.
        features: List of feature columns to include. If None, uses all numeric.

    Returns:
        DataFrame with centroids (rows=clusters, columns=features).
    """
    data_with_labels = data.copy()
    data_with_labels["_cluster"] = labels

    if features is None:
        features = data.select_dtypes(include=[np.number]).columns.tolist()

    centroids = data_with_labels.groupby("_cluster")[features].mean()
    centroids.index.name = "cluster_id"

    return centroids


def scale_to_unit_range(data: pd.DataFrame) -> pd.DataFrame:
    """Scale DataFrame columns to [0, 1] range (min-max normalization).

    Args:
        data: DataFrame with numeric features.

    Returns:
        Scaled DataFrame with same shape.
    """
    return (data - data.min()) / (data.max() - data.min())


def compute_distances_from_centroids(
    data: pd.DataFrame,
    labels: np.ndarray,
    centroids: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    """Compute Euclidean distance of each sample from its cluster centroid.

    Args:
        data: Feature DataFrame (rows=samples, columns=features).
        labels: Cluster labels for each sample.
        centroids: Centroid DataFrame (rows=clusters, columns=features).
        features: Feature columns to use for distance calculation.

    Returns:
        Array of distances (same length as data).
    """
    distances = np.zeros(len(data))

    for idx, (_row_idx, row) in enumerate(data.iterrows()):
        cluster_id = labels[idx]
        sample_features = row[features].values
        centroid_features = centroids.loc[cluster_id, features].values
        distances[idx] = np.sqrt(np.sum((sample_features - centroid_features) ** 2))

    return distances


def generate_cluster_names(
    centroids_normalized: pd.DataFrame,
    centroids_raw: pd.DataFrame,
    cluster_type: str = "geo",
    high_threshold: float = 0.70,
) -> dict[int, tuple[str, str]]:
    """Generate short and long descriptive names for clusters.

    Args:
        centroids_normalized: Normalized (0-1) centroid values.
        centroids_raw: Raw (original scale) centroid values.
        cluster_type: Type of clustering ("geo", "hydro", "hybrid").
        high_threshold: Threshold for identifying dominant features.

    Returns:
        Dictionary mapping cluster_id to (short_name, long_description).
        - short_name: e.g., "C1: Clay-rich (19.4%)"
        - long_description: e.g., "C1: Lowland (230m) / Clay-rich (19.4%) -
          European Plain..."

    Example:
        >>> names = generate_cluster_names(geo_centroids_norm, geo_centroids_raw)
        >>> names[0]
        ('C1: Clay-rich (19.4%)', 'C1: Lowland (230m) / Clay-rich (19.4%) - ...')
    """
    from src.plots.numeric_plots import interpret_cluster_from_hydroatlas

    cluster_names = {}

    for cluster_id in centroids_normalized.index:
        norm_row = centroids_normalized.loc[cluster_id]
        raw_row = centroids_raw.loc[cluster_id]

        # Get interpretation
        interpretation = interpret_cluster_from_hydroatlas(
            norm_row, raw_row, high_threshold=high_threshold
        )

        # Create short name (for legend)
        prefix = "C" if cluster_type == "geo" else "H"
        short_name = f"{prefix}{cluster_id + 1}: {interpretation.split(' / ')[0]}"

        # Create long description (for reference/export)
        long_desc = f"{prefix}{cluster_id + 1}: {interpretation}"

        # Add feature summary for long description
        high_features = norm_row[norm_row > high_threshold].sort_values(ascending=False)
        if len(high_features) > 0:
            feature_list = []
            for feat in high_features.index[:3]:  # Top 3 features
                val = raw_row[feat]
                if feat.endswith(("_pc_use", "_pc_ult", "_pc_uav")):
                    feature_list.append(f"{feat}={val:.1f}%")
                elif feat.endswith("_mt_"):
                    feature_list.append(f"{feat}={val:.0f}m")
                else:
                    feature_list.append(f"{feat}={val:.2f}")

            long_desc += f" [{', '.join(feature_list)}]"

        cluster_names[cluster_id] = (short_name, long_desc)

    return cluster_names
