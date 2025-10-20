"""Plotting functions for continuous (float) metrics on Russia maps.

Reworked to match russia_plots style with manual bin intervals,
improved NaN handling, and histogram/colorbar positioning.
"""

import math
from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import geopandas as gpd
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import numpy as np

from ..utils.logger import setup_logger

logger = setup_logger("continuous_maps", log_file="logs/continuous_maps.log")

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _get_aea_crs() -> ccrs.AlbersEqualArea:
    """Return Albers Equal Area CRS for Russia."""
    return ccrs.AlbersEqualArea(
        central_longitude=100,
        standard_parallels=(50, 70),
        central_latitude=56,
        false_easting=0,
        false_northing=0,
    )


def _create_bins_from_intervals(intervals: list[tuple[float, float]]) -> np.ndarray:
    """Create bin edges from list of interval tuples.

    Args:
        intervals: List of (min, max) tuples defining intervals.

    Returns:
        Array of bin edges suitable for BoundaryNorm.
    """
    edges = [intervals[0][0]]
    for _, upper in intervals:
        edges.append(upper)
    return np.array(edges)


def _add_histogram_inset(
    ax,
    gdf_data: gpd.GeoDataFrame,
    metric_col: str,
    bin_edges: np.ndarray,
    cmap,
    norm,
) -> None:
    """Add histogram inset matching russia_plots style.

    Histogram positioned at bottom-left with bars colored to match colorbar.
    No x-axis labels, only y-axis with occurrence counts.

    Args:
        ax: Matplotlib axis to add histogram to.
        gdf_data: GeoDataFrame with data.
        metric_col: Column name for metric.
        bin_edges: Bin edges for histogram.
        cmap: Colormap.
        norm: Normalization.
    """
    values = gdf_data[metric_col].dropna()
    if len(values) == 0:
        return

    # Create histogram data
    hist, _ = np.histogram(values, bins=bin_edges)

    ax_hist = ax.inset_axes([0.1, 0.07, 0.25, 0.20])
    # Create bars with matching colors from colorbar
    bar_colors = [cmap(norm((bin_edges[i] + bin_edges[i + 1]) / 2)) for i in range(len(hist))]
    extra_hist = ax_hist.bar(
        range(len(hist)),
        hist,
        width=0.7,
        color=bar_colors,
        edgecolor="black",
        linewidth=1.0,
    )

    # Add count labels on bars
    for bar in extra_hist:
        height = bar.get_height()
        if height > 0:
            ax_hist.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Style matching russia_plots
    ax_hist.set_facecolor("white")
    ax_hist.set(frame_on=False)

    # Remove x-axis labels (as requested)
    ax_hist.set_xticks([])
    ax_hist.set_xlabel("")

    # Keep y-axis with occurrence count labels
    plt.setp(ax_hist.get_yticklabels(), fontsize=8)


def _determine_bins(
    metric: str,
    gdf_proj: gpd.GeoDataFrame,
    bin_intervals: dict[str, list[tuple[float, float]]] | None,
) -> tuple[np.ndarray, int]:
    """Determine bin edges for metric.

    Args:
        metric: Metric column name.
        gdf_proj: Projected GeoDataFrame.
        bin_intervals: Optional dict of manual intervals.

    Returns:
        Tuple of (bin_edges array, number of bins).
    """
    if bin_intervals and metric in bin_intervals:
        intervals = bin_intervals[metric]
        bin_edges = _create_bins_from_intervals(intervals)
        n_bins = len(intervals)
    else:
        # Fallback: auto-generate bins from data
        values = gdf_proj[metric].dropna()
        if len(values) == 0:
            return np.array([0.0, 1.0]), 1
        vmin, vmax = float(values.min()), float(values.max())
        if vmin >= vmax:
            vmax = vmin + 0.1
        n_bins = 6
        bin_edges = np.linspace(vmin, vmax, n_bins + 1)

    return bin_edges, n_bins


def _plot_metric_points(
    ax,
    gdf_valid: gpd.GeoDataFrame,
    gdf_nan: gpd.GeoDataFrame,
    metric: str,
    cmap,
    norm,
    marker_size: int,
) -> None:
    """Plot valid and NaN points for a metric.

    Args:
        ax: Matplotlib axis.
        gdf_valid: GeoDataFrame with valid values.
        gdf_nan: GeoDataFrame with NaN values.
        metric: Metric column name.
        cmap: Colormap.
        norm: Normalization.
        marker_size: Point marker size.
    """
    # Plot valid points with colormap
    if not gdf_valid.empty:
        gdf_valid.plot(
            ax=ax,
            column=metric,
            cmap=cmap,
            norm=norm,
            marker="o",
            markersize=marker_size,
            edgecolor="black",
            linewidth=0.25,
            legend=True,
            legend_kwds={
                "orientation": "horizontal",
                "shrink": 0.50,
                "pad": -0.08,
                "aspect": 35,
                "anchor": (0.70, 1.0),
                "panchor": (0.70, 0.0),
                "label": "",
            },
        )

    # Plot NaN values as tiny black dots
    if not gdf_nan.empty:
        ax.scatter(
            gdf_nan.geometry.x,
            gdf_nan.geometry.y,
            s=4,  # Tiny size
            c="black",
            marker="o",
            edgecolor="none",
            zorder=2,
            alpha=0.8,
        )


def _format_colorbar(fig, idx: int, bin_edges: np.ndarray) -> None:
    """Format colorbar ticks and labels.

    Args:
        fig: Matplotlib figure.
        idx: Current subplot index.
        bin_edges: Bin edge values.
    """
    if len(fig.axes) <= idx + 1:
        return

    cbar_ax = fig.axes[-1]
    if not hasattr(cbar_ax, "get_xlim"):
        return
    try:
        cbar_ax.set_xticks(bin_edges)
        tick_labels = [f"{val:.3g}" for val in bin_edges]
        cbar_ax.set_xticklabels(tick_labels, fontsize=7)
        cbar_ax.tick_params(labelsize=7, length=3, width=0.5)
    except Exception:
        logger.exception("Failed to format colorbar ticks and labels")
        # Skip if colorbar formatting fails


def russia_continuous_multiplot(
    gdf_to_plot: gpd.GeoDataFrame,
    basemap_data: gpd.GeoDataFrame,
    metrics: list[str],
    titles: list[str] | None = None,
    main_title: str = "",
    rus_extent: list | None = None,
    bin_intervals: dict[str, list[tuple[float, float]]] | None = None,
    cmap_name: str = "RdYlGn",
    ncols: int = 3,
    subplot_size: tuple = (6.0, 4.5),
    with_histogram: bool = True,
    marker_size: int = 12,
) -> "Figure":
    """Create multipanel plot with continuous metrics.

    Completely reworked to match russia_plots style with manual bin intervals.

    Args:
        gdf_to_plot: GeoDataFrame with points to plot.
        basemap_data: GeoDataFrame with basemap polygons.
        metrics: List of column names to plot.
        titles: Optional list of titles for each subplot.
        main_title: Overall figure title.
        rus_extent: Extent for Russia map.
        bin_intervals: Dict mapping metric name to list of (min, max) tuples
            defining custom intervals. Example:
            {"mean_discharge": [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0)]}
        cmap_name: Matplotlib colormap name.
        ncols: Number of columns in subplot grid.
        subplot_size: Size (width, height) of each subplot.
        with_histogram: Whether to add histogram inset (default True).
        marker_size: Size of point markers (default 12).

    Returns:
        Matplotlib figure object.
    """
    from matplotlib.figure import Figure

    n_metrics = len(metrics)
    if n_metrics == 0:
        raise ValueError("No metrics provided")

    # Calculate subplot grid
    nrows = math.ceil(n_metrics / ncols)

    # Use default titles if not provided
    titles_to_use = titles if titles is not None else metrics

    if len(titles_to_use) != n_metrics:
        raise ValueError("Number of titles must match number of metrics")

    # Setup CRS
    aea_crs = _get_aea_crs()
    aea_crs_proj4 = aea_crs.proj4_init

    extent = rus_extent if rus_extent is not None else [19.5, 180, 41.5, 82]

    # Convert to projected CRS once
    gdf_proj = gdf_to_plot.to_crs(aea_crs_proj4)
    basemap_proj = basemap_data.to_crs(aea_crs_proj4)

    # Create figure with subplots
    fig_width = subplot_size[0] * ncols
    fig_height = subplot_size[1] * nrows
    fig = Figure(figsize=(fig_width, fig_height))

    for idx, (metric, title) in enumerate(zip(metrics, titles_to_use, strict=False)):
        # Create subplot with projection
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection=aea_crs)
        ax.set_aspect("auto")
        ax.axis("off")
        ax.set_extent(extent, crs=ccrs.PlateCarree())  # type: ignore

        # Plot basemap
        basemap_proj.plot(
            ax=ax, color="#E5E5E5", edgecolor="#404040", linewidth=0.5, alpha=1.0, legend=False
        )

        # Determine bins for this metric
        bin_edges, n_bins = _determine_bins(metric, gdf_proj, bin_intervals)

        # Create colormap and normalization
        norm = BoundaryNorm(bin_edges, n_bins)
        cmap = cm.get_cmap(cmap_name, n_bins)

        # Separate NaN and valid values
        gdf_valid = gdf_proj[gdf_proj[metric].notna()].copy()
        gdf_nan = gdf_proj[gdf_proj[metric].isna()].copy()

        # Plot points
        _plot_metric_points(ax, gdf_valid, gdf_nan, metric, cmap, norm, marker_size)

        # Format colorbar
        _format_colorbar(fig, idx, bin_edges)

        # Add histogram if requested
        if with_histogram and not gdf_valid.empty:
            _add_histogram_inset(ax, gdf_valid, metric, bin_edges, cmap, norm)

        ax.set_title(title, fontdict={"size": 11, "weight": "normal"}, pad=8)

    # Add main title
    if main_title:
        fig.suptitle(main_title, fontsize=15, y=0.99, weight="bold")

    # Adjust layout
    rect = (0.0, 0.01, 1.0, 0.97) if main_title else (0.0, 0.01, 1.0, 1.0)
    fig.tight_layout(rect=rect)

    return fig
