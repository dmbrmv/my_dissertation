"""Visualization functions for cluster analysis.

This module provides plotting utilities for hierarchical clustering results,
including dendrograms, PCA scatter plots, radar charts, and combined map+radar
visualizations for hydrological catchment analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns
from sklearn.decomposition import PCA

from src.utils.logger import setup_logger

if TYPE_CHECKING:
    from collections import OrderedDict

log = setup_logger(__name__)


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    n_clusters: int,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (16, 8),
    title: str | None = None,
) -> Figure:
    """Plot hierarchical clustering dendrogram with cluster cutoff line.

    Args:
        linkage_matrix: Linkage matrix from scipy.cluster.hierarchy.linkage.
        n_clusters: Number of clusters to highlight with cutoff line.
        output_path: Path to save figure. If None, figure is not saved.
        figsize: Figure size (width, height) in inches.
        title: Custom title. If None, uses default title.

    Returns:
        Matplotlib figure object.

    Example:
        >>> from scipy.cluster.hierarchy import linkage
        >>> Z = linkage(data, method="ward")
        >>> plot_dendrogram(Z, n_clusters=9, output_path="dendrogram.png")
    """
    fig, ax = plt.subplots(figsize=figsize)

    _ = dendrogram(
        linkage_matrix,
        ax=ax,
        above_threshold_color="#808080",
        color_threshold=linkage_matrix[-n_clusters + 1, 2],
        no_labels=True,
        truncate_mode="level",
        p=0,
    )

    ax.set_xlabel("Catchments (unlabeled due to high count)", fontsize=12)
    ax.set_ylabel("Ward distance", fontsize=12)

    if title is None:
        title = f"Hierarchical Clustering Dendrogram ({n_clusters} clusters)"
    ax.set_title(title, fontsize=14, pad=15)

    ax.axhline(
        y=linkage_matrix[-n_clusters + 1, 2],
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Cut at {n_clusters} clusters",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        log.info("Saved dendrogram to %s", output_path)

    return fig


def plot_pca_clusters(
    data: pd.DataFrame,
    labels: np.ndarray,
    n_components: int = 10,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 10),
    colors: list[str] | None = None,
) -> plt.Figure:
    """Plot clusters in PCA space (first 2 principal components).

    Args:
        data: Feature matrix (rows=samples, columns=features).
        labels: Cluster labels for each sample.
        n_components: Number of PCA components to compute.
        output_path: Path to save figure. If None, figure is not saved.
        figsize: Figure size (width, height) in inches.
        colors: List of colors for clusters. If None, uses tab10 colormap.

    Returns:
        Matplotlib figure object.
    """
    # Perform PCA
    pca = PCA(n_components=min(n_components, data.shape[1]))
    pca_features = pca.fit_transform(data)

    variance_explained = pca.explained_variance_ratio_

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    n_clusters = len(np.unique(labels))
    if colors is None:
        cmap = plt.get_cmap("tab10", n_clusters)
        colors = [cmap(i) for i in range(n_clusters)]

    # Ensure colors is a list
    color_list = colors if isinstance(colors, list) else list(colors)

    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = labels == cluster_id
        cluster_data = pca_features[cluster_mask]

        ax.scatter(
            cluster_data[:, 0],
            cluster_data[:, 1],
            c=[color_list[cluster_id - 1]],
            label=f"Cluster {cluster_id}",
            s=60,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_xlabel(f"PC1 ({variance_explained[0] * 100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({variance_explained[1] * 100:.1f}% variance)", fontsize=12)
    ax.set_title(f"Clusters in PCA Space ({n_clusters} clusters)", fontsize=14, pad=15)
    ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color="k", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        log.info("Saved PCA plot to %s", output_path)

    return fig


def plot_cluster_heatmap(
    centroids: pd.DataFrame,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Plot heatmap of cluster centroids showing feature signatures.

    Args:
        centroids: DataFrame with centroids (rows=clusters, columns=features).
        output_path: Path to save figure. If None, figure is not saved.
        figsize: Figure size (width, height) in inches.

    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        centroids.T,
        cmap="RdYlBu_r",
        center=0.5,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Normalized value (0-1 scale)"},
        linewidths=0.5,
        ax=ax,
    )

    ax.set_xlabel("Cluster ID", fontsize=12)
    ax.set_ylabel("HydroATLAS Feature", fontsize=12)
    ax.set_title(
        f"Cluster Centroids - Feature Signatures ({len(centroids)} clusters)",
        fontsize=14,
        pad=15,
    )

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        log.info("Saved heatmap to %s", output_path)

    return fig


def plot_radar_charts(
    centroids: pd.DataFrame,
    features: list[str],
    cluster_names: dict[int, str] | None = None,
    colors: list[str] | None = None,
    n_cols: int = 3,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (25, 15),
) -> plt.Figure:
    """Plot radar charts showing normalized feature values for each cluster.

    Args:
        centroids: DataFrame with centroids (rows=clusters, columns=features).
        features: List of feature names to plot on radar chart.
        cluster_names: Dictionary mapping cluster IDs to descriptive names.
        colors: List of colors for clusters. If None, uses tab10 colormap.
        n_cols: Number of columns in subplot grid.
        output_path: Path to save figure. If None, figure is not saved.
        figsize: Figure size (width, height) in inches.

    Returns:
        Matplotlib figure object.
    """
    n_clusters = len(centroids)
    n_rows = int(np.ceil(n_clusters / n_cols))

    # Prepare angles for radar chart
    n_features = len(features)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Setup colors
    if colors is None:
        cmap = plt.get_cmap("tab10", n_clusters)
        colors = [cmap(i) for i in range(n_clusters)]

    # Create subplots
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        subplot_kw={"projection": "polar"},
    )
    axes = axes.flatten() if n_clusters > 1 else [axes]

    for idx, cluster_id in enumerate(centroids.index):
        ax = axes[idx]

        # Get centroid values
        values = centroids.loc[cluster_id, features].tolist()
        values += values[:1]  # Complete the circle

        color_idx = idx % len(colors)

        # Plot
        ax.plot(angles, values, "o-", linewidth=2, color=colors[color_idx])
        ax.fill(angles, values, alpha=0.25, color=colors[color_idx])

        # Styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, size=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"], size=8)
        ax.grid(True, alpha=0.3)

        # Title
        if cluster_names and cluster_id in cluster_names:
            title = f"Cluster {cluster_id}: {cluster_names[cluster_id]}"
        else:
            title = f"Cluster {cluster_id}"
        ax.set_title(title, size=10, weight="bold", pad=20)

        # Reference line at 0.5
        ax.axhline(y=0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    # Remove empty subplots
    for idx in range(n_clusters, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(
        "Cluster Profiles: Normalized Feature Values (0-1 scale)",
        fontsize=16,
        y=0.995,
        weight="bold",
    )
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        log.info("Saved radar charts to %s", output_path)

    return fig


def plot_hydrograph_clusters(
    cluster_patterns: OrderedDict[str, pd.DataFrame],
    n_cols: int = 5,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (20, 10),
    colors: dict[str, Any] | None = None,
) -> plt.Figure:
    """Plot hydrograph panels showing normalized seasonal discharge patterns.

    Args:
        cluster_patterns: OrderedDict mapping cluster names to DataFrames.
            Each DataFrame has rows=days (366), columns=gauge_ids.
        n_cols: Number of columns in subplot grid.
        output_path: Path to save figure. If None, figure is not saved.
        figsize: Figure size (width, height) in inches.
        colors: Dictionary with color specifications. If None, uses defaults.

    Returns:
        Matplotlib figure object.
    """
    n_clusters = len(cluster_patterns)
    n_rows = int(np.ceil(n_clusters / n_cols))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    axes_flat = axes.flatten() if n_clusters > 1 else [axes]

    for i, (clust_name, clust_data) in enumerate(cluster_patterns.items()):
        if i >= len(axes_flat):
            break
        clust_ax = axes_flat[i]

        # Calculate daily statistics
        mean_by_gauge = (
            clust_data.groupby([clust_data.index.month, clust_data.index.day])
            .median()
            .reset_index(drop=True)
        )

        clust_median = mean_by_gauge.median(axis=1).values.ravel()
        clust_p25 = mean_by_gauge.quantile(0.25, axis=1).values.ravel()
        clust_p75 = mean_by_gauge.quantile(0.75, axis=1).values.ravel()

        # Layer 1: Individual gauges (background)
        for gauge_id in mean_by_gauge.columns:
            clust_ax.plot(
                mean_by_gauge.index.values,
                mean_by_gauge[gauge_id].values,
                color="blue",
                alpha=0.12,
                linewidth=0.5,
                zorder=1,
            )

        # Layer 2: 25-75 percentile spread
        clust_ax.fill_between(
            mean_by_gauge.index.values,
            clust_p25,
            clust_p75,
            color="salmon",
            alpha=0.35,
            label="25-75% spread",
            zorder=2,
        )

        # Layer 3: Cluster median (foreground)
        clust_ax.plot(
            mean_by_gauge.index.values,
            clust_median,
            color="darkred",
            alpha=1.0,
            linewidth=2.8,
            label="Cluster median",
            zorder=3,
        )

        # Formatting
        clust_ax.grid(axis="both", which="both", alpha=0.3, zorder=0)
        clust_ax.set_xlim(0, 365)
        clust_ax.set_ylim(0, 1)
        clust_ax.set_yticks(np.arange(0, 1.25, 0.25))
        clust_ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        if i % n_cols == 0:
            clust_ax.set_ylabel("Normalized runoff [0-1]")
        if i >= (n_rows - 1) * n_cols:
            clust_ax.set_xlabel("Day of year")

        for val in np.arange(0, 1.25, 0.25):
            clust_ax.axhline(val, c="black", linestyle="--", linewidth=0.2, zorder=0)

        clust_ax.set_title(f"{clust_name} — {len(clust_data.columns)} gauges")

        # Add legend to first subplot
        if i == 0:
            clust_ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # Turn off unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        "Hydrological Regime Types: Normalized Seasonal Discharge Patterns",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        log.info("Saved hydrograph clusters to %s", output_path)

    return fig


def plot_spatial_clusters(
    watersheds: Any,
    gauges: Any,
    cluster_labels: np.ndarray,
    cluster_col: str,
    basemap: Any | None = None,
    output_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (16, 12),
    show_watersheds: bool = True,
    show_gauges: bool = True,
    cmap_name: str = "tab10",
    rus_extent: list[float] | None = None,
    cluster_names: dict[int, str] | None = None,
    markers_list: list[str] | None = None,
    color_list: list[str] | None = None,
    base_marker_size: int = 30,
    marker_size_corrections: dict[str, float] | None = None,
    marker_size_variants: list[float] | None = None,
    base_linewidth: float = 0.35,
    linewidth_variants: list[float] | None = None,
    marker_edgecolor: str = "black",
    marker_alpha: float = 0.85,
    legend_cols: int = 3,
    legend_location: str = "lower center",
    legend_fontsize: int = 14,
    legend_auto_position: bool = False,
    with_histogram: bool = False,
    histogram_col: str | None = None,
    histogram_rect: tuple[float, float, float, float] = (0.00, 0.05, 0.33, 0.24),
    histogram_bar_colors: list[str] | str | None = None,
    histogram_xticklabels: list[str] | None = None,
    histogram_label_rotation: float = 30.0,
    basemap_color: str = "grey",
    basemap_edgecolor: str = "black",
    basemap_linewidth: float = 0.6,
    basemap_alpha: float = 0.8,
) -> Figure:
    """Plot spatial distribution of clusters on Russia map with proper projection.

    Uses Albers Equal Area projection optimized for Russia territory. Now supports
    full marker cycling system, histogram insets, and unified styling parameters
    consistent with russia_plots.

    Args:
        watersheds: GeoDataFrame with watershed geometries.
        gauges: GeoDataFrame with gauge locations.
        cluster_labels: Array of cluster labels aligned with gauges.
        cluster_col: Column name to store cluster labels.
        basemap: Optional GeoDataFrame for basemap boundaries.
        output_path: Path to save figure. If None, figure is not saved.
        title: Custom title. If None, uses default title.
        figsize: Figure size (width, height) in inches.
        show_watersheds: If True, plot watersheds colored by cluster.
        show_gauges: If True, plot gauge locations as points.
        cmap_name: Matplotlib colormap name (used if color_list not provided).
        rus_extent: Map extent [lon_min, lon_max, lat_min, lat_max].
        cluster_names: Optional dict mapping cluster numbers to descriptive names.
        markers_list: Custom marker shapes. Uses default if None.
        color_list: Explicit list of colors. Uses cmap_name if None.
        base_marker_size: Base marker size before scaling.
        marker_size_corrections: Per-marker size adjustments for visual balance.
        marker_size_variants: Size multipliers for repeated markers [1.0, 1.4, 1.8].
        base_linewidth: Base edge linewidth before scaling.
        linewidth_variants: Linewidth multipliers for repeated markers [1.0, 1.6, 2.2].
        marker_edgecolor: Marker edge color.
        marker_alpha: Marker transparency (0-1).
        legend_cols: Number of columns in legend.
        legend_location: Legend location string.
        legend_fontsize: Legend text size.
        legend_auto_position: Auto-adjust legend position if histogram shown.
        with_histogram: Show histogram inset.
        histogram_col: Column for histogram (None = use cluster_col).
        histogram_rect: Position [x, y, width, height] in axes coordinates.
        histogram_bar_colors: Bar colors (None = use color_list).
        histogram_xticklabels: Custom x-tick labels.
        histogram_label_rotation: X-tick label rotation angle.
        basemap_color: Basemap polygon fill color.
        basemap_edgecolor: Basemap polygon edge color.
        basemap_linewidth: Basemap polygon edge width.
        basemap_alpha: Basemap transparency.

    Returns:
        Matplotlib figure object.

    Example:
        >>> plot_spatial_clusters(
        ...     watersheds=ws,
        ...     gauges=gauges,
        ...     cluster_labels=geo_labels,
        ...     cluster_col="geo_cluster",
        ...     title="Geographical Clusters",
        ...     with_histogram=True,
        ...     color_list=["red", "blue", "green"],
        ...     base_marker_size=30,
        ... )
    """
    import matplotlib.colors as mcolors

    from src.analytics.static_analysis import (
        get_cluster_colors,
        get_cluster_markers,
        get_marker_size_corrections as get_default_marker_corrections,
    )
    from src.plots.styling_utils import (
        add_histogram_inset,
        create_legend_handles,
        get_marker_system,
        get_russia_projection,
        get_unified_colors,
        render_basemap,
        sort_plot_categories,
    )

    # Setup projection
    aea_crs = get_russia_projection()
    aea_crs_proj4 = aea_crs.proj4_init

    if rus_extent is None:
        rus_extent = [50, 140, 32, 90]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": aea_crs})
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_extent(rus_extent)  # type: ignore

    # Plot basemap
    if basemap is not None:
        render_basemap(
            ax,
            basemap,
            aea_crs_proj4,
            color=basemap_color,
            edgecolor=basemap_edgecolor,
            linewidth=basemap_linewidth,
            alpha=basemap_alpha,
        )

    # Prepare data with cluster labels
    ws_plot = watersheds.copy()
    ws_plot[cluster_col] = cluster_labels
    ws_plot = ws_plot.to_crs(aea_crs_proj4)
    ws_plot["_area"] = ws_plot.geometry.area
    ws_plot = ws_plot.sort_values("_area", ascending=False)

    gauges_plot = gauges.copy()
    gauges_plot[cluster_col] = cluster_labels
    gauges_plot = gauges_plot.to_crs(aea_crs_proj4)

    # Get unique clusters and sort
    raw_clusters = [str(c) for c in np.unique(cluster_labels)]
    sorted_cluster_strs = sort_plot_categories(raw_clusters)
    unique_clusters = [int(c) if c.isdigit() else c for c in sorted_cluster_strs]
    n_clusters = len(unique_clusters)

    # Get colors using unified system
    if color_list is not None:
        colors = get_unified_colors(n_clusters, color_list=color_list)
    elif markers_list is None:
        # Fallback to external color function
        colors = get_cluster_colors(n_clusters)
        colors = np.array([mcolors.to_rgba(c) for c in colors])
    else:
        colors = get_unified_colors(n_clusters, cmap_name=cmap_name)

    # Get marker system
    if markers_list is None:
        markers_list = get_cluster_markers(n_clusters)

    if marker_size_corrections is None:
        marker_size_corrections = get_default_marker_corrections()

    marker_shapes, marker_sizes, marker_linewidths = get_marker_system(
        n_clusters,
        markers_list=markers_list,
        base_marker_size=base_marker_size,
        marker_size_corrections=marker_size_corrections,
        marker_size_variants=marker_size_variants,
        base_linewidth=base_linewidth,
        linewidth_variants=linewidth_variants,
    )

    # Plot watersheds by cluster
    if show_watersheds:
        for i, cluster in enumerate(unique_clusters):
            cluster_ws = ws_plot[ws_plot[cluster_col] == cluster]
            if len(cluster_ws) > 0:
                cluster_ws.plot(
                    ax=ax,
                    color=colors[i],
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.7,
                    legend=False,
                )

    # Plot gauges with marker system
    if show_gauges:
        for i, cluster in enumerate(unique_clusters):
            cluster_gauges = gauges_plot[gauges_plot[cluster_col] == cluster]
            if len(cluster_gauges) == 0:
                continue

            xs = cluster_gauges.geometry.x
            ys = cluster_gauges.geometry.y

            ax.scatter(
                xs,
                ys,
                marker=marker_shapes[i],
                s=marker_sizes[i],
                c=[colors[i]],
                edgecolor=marker_edgecolor,
                linewidth=marker_linewidths[i],
                zorder=5,
                alpha=marker_alpha,
            )

        # Create legend
        legend_handles = create_legend_handles(
            categories=[str(c) for c in unique_clusters],
            colors=colors,
            markers=marker_shapes,
            category_names=cluster_names,  # type: ignore[arg-type]
            marker_edgecolor=marker_edgecolor,
        )

        # Adjust legend location if histogram present
        if legend_auto_position and with_histogram:
            legend_loc = "lower right"
        else:
            legend_loc = legend_location

        if legend_handles:
            ax.legend(
                handles=legend_handles,
                ncol=legend_cols,
                loc=legend_loc,
                frameon=True,
                fontsize=legend_fontsize,
            )

    # Add histogram if requested
    if with_histogram:
        hist_col = histogram_col if histogram_col is not None else cluster_col
        if histogram_bar_colors is not None:
            hist_colors = histogram_bar_colors
        else:
            # Convert colors array to list of hex strings
            hist_colors = [mcolors.rgb2hex(c) for c in colors]

        add_histogram_inset(
            ax=ax,
            data=gauges_plot,
            column=hist_col,
            position=histogram_rect,
            bar_colors=hist_colors,
            xticklabels=histogram_xticklabels,
            label_rotation=histogram_label_rotation,
        )

    ax.set_title(title or "Cluster Distribution", fontsize=12)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        log.info("Saved spatial cluster map to %s", output_path)

    return fig


def plot_hybrid_spatial_clusters(
    watersheds: Any,
    gauges: Any,
    gauge_mapping: pd.DataFrame,
    hybrid_col: str = "hybrid_class",
    basemap: Any | None = None,
    output_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (18, 14),
    show_watersheds: bool = True,
    show_gauges: bool = True,
    rus_extent: list[float] | None = None,
    cmap_name: str = "tab10",
    color_list: list[str] | None = None,
    markers_list: list[str] | None = None,
    base_marker_size: int = 30,
    marker_size_corrections: dict[str, float] | None = None,
    marker_size_variants: list[float] | None = None,
    base_linewidth: float = 0.35,
    linewidth_variants: list[float] | None = None,
    marker_edgecolor: str = "black",
    marker_alpha: float = 0.85,
    legend_cols: int | None = None,
    legend_cols_auto: bool = True,
    legend_location: str = "lower right",
    legend_fontsize: int = 14,
    legend_auto_position: bool = False,
    with_histogram: bool = False,
    histogram_col: str | None = None,
    histogram_rect: tuple[float, float, float, float] = (0.00, 0.05, 0.33, 0.24),
    histogram_bar_colors: list[str] | str | None = None,
    histogram_xticklabels: list[str] | None = None,
    histogram_label_rotation: float = 30.0,
    basemap_color: str = "grey",
    basemap_edgecolor: str = "black",
    basemap_linewidth: float = 0.6,
    basemap_alpha: float = 0.8,
) -> Figure:
    """Plot spatial distribution of hybrid classification on Russia map.

    Uses Albers Equal Area projection optimized for Russia territory. Now supports
    full marker cycling system, histogram insets, and unified styling parameters.

    Args:
        watersheds: GeoDataFrame with watershed geometries.
        gauges: GeoDataFrame with gauge locations.
        gauge_mapping: DataFrame with gauge_id and hybrid classification.
        hybrid_col: Column name containing hybrid class labels.
        basemap: Optional GeoDataFrame for basemap boundaries.
        output_path: Path to save figure. If None, figure is not saved.
        title: Custom title. If None, uses default title.
        figsize: Figure size (width, height) in inches.
        show_watersheds: If True, plot watersheds colored by hybrid class.
        show_gauges: If True, plot gauge locations as points.
        rus_extent: Map extent [lon_min, lon_max, lat_min, lat_max].
        cmap_name: Matplotlib colormap name (used if color_list not provided).
        color_list: Explicit list of colors. Uses cmap_name if None.
        markers_list: Custom marker shapes. Uses default if None.
        base_marker_size: Base marker size before scaling.
        marker_size_corrections: Per-marker size adjustments for visual balance.
        marker_size_variants: Size multipliers for repeated markers [1.0, 1.4, 1.8].
        base_linewidth: Base edge linewidth before scaling.
        linewidth_variants: Linewidth multipliers for repeated markers [1.0, 1.6, 2.2].
        marker_edgecolor: Marker edge color.
        marker_alpha: Marker transparency (0-1).
        legend_cols: Number of columns in legend (None = auto).
        legend_cols_auto: If True, use conditional logic (3 if n<=15 else 4).
        legend_location: Legend location string.
        legend_fontsize: Legend text size.
        legend_auto_position: Auto-adjust legend position if histogram shown.
        with_histogram: Show histogram inset.
        histogram_col: Column for histogram (None = use hybrid_col).
        histogram_rect: Position [x, y, width, height] in axes coordinates.
        histogram_bar_colors: Bar colors (None = use color_list).
        histogram_xticklabels: Custom x-tick labels.
        histogram_label_rotation: X-tick label rotation angle.
        basemap_color: Basemap polygon fill color.
        basemap_edgecolor: Basemap polygon edge color.
        basemap_linewidth: Basemap polygon edge width.
        basemap_alpha: Basemap transparency.

    Returns:
        Matplotlib figure object.

    Example:
        >>> plot_hybrid_spatial_clusters(
        ...     watersheds=ws,
        ...     gauges=gauges,
        ...     gauge_mapping=gauge_mapping,
        ...     hybrid_col="hybrid_class",
        ...     title="Hybrid Classification",
        ...     with_histogram=True,
        ...     color_list=["red", "blue", "green"],
        ... )
    """
    import matplotlib.colors as mcolors

    from src.plots.styling_utils import (
        add_histogram_inset,
        create_legend_handles,
        get_marker_system,
        get_russia_projection,
        get_unified_colors,
        render_basemap,
        sort_plot_categories,
    )

    # Setup projection
    aea_crs = get_russia_projection()
    aea_crs_proj4 = aea_crs.proj4_init

    if rus_extent is None:
        rus_extent = [50, 140, 32, 90]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": aea_crs})
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_extent(rus_extent)  # type: ignore

    # Plot basemap
    if basemap is not None:
        render_basemap(
            ax,
            basemap,
            aea_crs_proj4,
            color=basemap_color,
            edgecolor=basemap_edgecolor,
            linewidth=basemap_linewidth,
            alpha=basemap_alpha,
        )

    # Merge gauge mapping with geometries
    gauge_ids = gauge_mapping["gauge_id"].values
    ws_plot = watersheds.loc[gauge_ids].copy()
    ws_plot[hybrid_col] = gauge_mapping[hybrid_col].values
    ws_plot = ws_plot.to_crs(aea_crs_proj4)
    ws_plot["_area"] = ws_plot.geometry.area
    ws_plot = ws_plot.sort_values("_area", ascending=False)

    gauges_plot = gauges.copy().loc[gauge_ids].copy()
    gauges_plot[hybrid_col] = gauge_mapping[hybrid_col].values
    gauges_plot = gauges_plot.to_crs(aea_crs_proj4)

    # Get unique classes and sort
    raw_classes = [str(c) for c in ws_plot[hybrid_col].unique()]
    sorted_classes = sort_plot_categories(raw_classes)
    n_classes = len(sorted_classes)

    # Get colors using unified system
    if color_list is not None:
        colors = get_unified_colors(n_classes, color_list=color_list)
    else:
        # Auto-select colormap based on n_classes (legacy behavior)
        if n_classes <= 10:
            colors = get_unified_colors(n_classes, cmap_name="tab10")
        elif n_classes <= 20:
            colors = get_unified_colors(n_classes, cmap_name="tab20")
        else:
            colors = get_unified_colors(n_classes, cmap_name="gist_ncar")

    # Get marker system (full cycling support)
    if marker_size_corrections is None:
        from src.analytics.static_analysis import get_marker_size_corrections

        marker_size_corrections = get_marker_size_corrections()

    marker_shapes, marker_sizes, marker_linewidths = get_marker_system(
        n_classes,
        markers_list=markers_list,
        base_marker_size=base_marker_size,
        marker_size_corrections=marker_size_corrections,
        marker_size_variants=marker_size_variants,
        base_linewidth=base_linewidth,
        linewidth_variants=linewidth_variants,
    )

    # Determine legend columns
    if legend_cols is None and legend_cols_auto:
        legend_cols = 3 if n_classes <= 15 else 4
    elif legend_cols is None:
        legend_cols = 3

    # Plot watersheds by class
    if show_watersheds:
        for i, hybrid_class in enumerate(sorted_classes):
            class_ws = ws_plot[ws_plot[hybrid_col].astype(str) == hybrid_class]
            if len(class_ws) > 0:
                class_ws.plot(
                    ax=ax,
                    color=colors[i],
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.7,
                    legend=False,
                )

    # Plot gauges with marker system
    if show_gauges:
        for i, hybrid_class in enumerate(sorted_classes):
            class_gauges = gauges_plot[
                gauges_plot[hybrid_col].astype(str) == hybrid_class
            ]
            if len(class_gauges) == 0:
                continue

            xs = class_gauges.geometry.x
            ys = class_gauges.geometry.y

            ax.scatter(
                xs,
                ys,
                marker=marker_shapes[i],
                s=marker_sizes[i],
                c=[colors[i]],
                edgecolor=marker_edgecolor,
                linewidth=marker_linewidths[i],
                zorder=5,
                alpha=marker_alpha,
            )

        # Create legend using custom handles
        legend_handles = create_legend_handles(
            categories=sorted_classes,
            colors=colors,
            markers=marker_shapes,
            marker_edgecolor=marker_edgecolor,
        )

        # Adjust legend location if histogram present
        if legend_auto_position and with_histogram:
            legend_loc = "lower right"
        else:
            legend_loc = legend_location

        if legend_handles:
            ax.legend(
                handles=legend_handles,
                ncol=legend_cols,
                loc=legend_loc,
                frameon=True,
                fontsize=legend_fontsize,
            )

    # Add histogram if requested
    if with_histogram:
        hist_col = histogram_col if histogram_col is not None else hybrid_col
        if histogram_bar_colors is not None:
            hist_colors = histogram_bar_colors
        else:
            # Convert colors array to list of hex strings
            hist_colors = [mcolors.rgb2hex(c) for c in colors]

        add_histogram_inset(
            ax=ax,
            data=gauges_plot,
            column=hist_col,
            position=histogram_rect,
            bar_colors=hist_colors,
            xticklabels=histogram_xticklabels,
            label_rotation=histogram_label_rotation,
        )

    default_title = f"Hybrid Classification ({n_classes} classes)"
    ax.set_title(title or default_title, fontsize=12)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        log.info("Saved hybrid spatial cluster map to %s", output_path)

    return fig
