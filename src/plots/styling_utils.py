"""Unified styling utilities for spatial plotting functions.

This module provides shared styling infrastructure for consistent visualization
across russia_plots, plot_spatial_clusters, plot_hybrid_spatial_clusters, and
hex_model_distribution_plot functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

import cartopy.crs as ccrs
import geopandas as gpd
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class PlotStyleConfig:
    """Unified configuration for spatial plot styling.

    This dataclass encapsulates all standardized styling parameters that can be
    shared across different plotting functions.

    Attributes:
        # Projection & Extent
        rus_extent: Map extent [lon_min, lon_max, lat_min, lat_max]
        aea_projection: Custom Albers Equal Area projection (if None, uses default)

        # Figure Layout
        figsize: Figure size (width, height) in inches
        title: Plot title
        title_fontsize: Title font size

        # Color Management
        cmap_name: Colormap name for discrete/continuous coloring
        cmap_lims: Min/max values for continuous colormap normalization
        cmap_bins: Number of discrete bins for colormap
        color_list: Explicit list of colors to use instead of colormap
        skip_colors: Number of colors to skip when sampling colormap

        # Marker System (for point-based plots)
        markers_list: Custom marker shapes
        default_markers: Default marker shape list
        base_marker_size: Base size for markers on map
        marker_size_corrections: Per-marker size adjustments for visual balance
        marker_size_variants: Size multipliers for repeated markers
        legend_marker_size: Marker size in legend (None = auto-calculate)
        base_linewidth: Base edge linewidth for markers
        linewidth_variants: Linewidth multipliers for repeated markers
        marker_edgecolor: Marker edge color
        marker_alpha: Marker transparency

        # Legend Configuration
        legend_show: Whether to show legend
        legend_cols: Number of columns in legend
        legend_location: Legend location string
        legend_auto_position: Auto-adjust position based on histogram presence
        legend_frameon: Show legend frame
        legend_fontsize: Legend text size
        legend_title: Legend title (None = use column name)
        legend_markerscale: Marker scale factor for legend

        # Histogram Configuration
        with_histogram: Show histogram inset
        histogram_rect: Position [x, y, width, height] in axes coordinates
        histogram_col: Column for histogram data (None = use distinction_col)
        histogram_bins: Bin edges for continuous data
        histogram_bar_colors: Bar colors (None = use color_list)
        histogram_bar_edgecolor: Bar edge color
        histogram_bar_linewidth: Bar edge width
        histogram_bar_width: Bar width (0-1)
        histogram_background: Histogram background color
        histogram_frame: Show histogram frame
        histogram_grid: Show histogram grid
        histogram_xlabel: X-axis label
        histogram_xticklabels: Custom x-tick labels
        histogram_label_rotation: X-tick label rotation angle
        histogram_label_fontsize: Label font size
        histogram_show_counts: Show bar count labels
        histogram_count_format: Format string for count labels
        histogram_count_fontsize: Count label font size

        # Basemap Rendering
        basemap_color: Basemap polygon fill color
        basemap_edgecolor: Basemap polygon edge color
        basemap_linewidth: Basemap polygon edge width
        basemap_alpha: Basemap transparency

        # Output Configuration
        output_path: Save path (None = don't save)
        dpi: Save resolution
        bbox_inches: Save bounding box
    """

    # Projection & Extent
    rus_extent: tuple[float, float, float, float] = (50, 140, 32, 90)
    aea_projection: ccrs.AlbersEqualArea | None = None

    # Figure Layout
    figsize: tuple[float, float] = (4.88189, 3.34646)
    title: str = ""
    title_fontsize: int = 12

    # Color Management
    cmap_name: str = "RdYlGn"
    cmap_lims: tuple[float, float] = (0.0, 1.0)
    cmap_bins: int = 18
    color_list: list[str] | None = None
    skip_colors: int = 0

    # Marker System
    markers_list: list[str] | None = None
    default_markers: list[str] = field(
        default_factory=lambda: [
            "o",
            "s",
            "^",
            "v",
            "<",
            ">",
            "P",
            "X",
            "D",
            "*",
            "h",
            "H",
        ]
    )
    base_marker_size: int = 30
    marker_size_corrections: dict[str, float] | None = None
    marker_size_variants: list[float] = field(default_factory=lambda: [1.0, 1.4, 1.8])
    legend_marker_size: int | None = None
    base_linewidth: float = 0.35
    linewidth_variants: list[float] = field(default_factory=lambda: [1.0, 1.6, 2.2])
    marker_edgecolor: str = "black"
    marker_alpha: float = 1.0

    # Legend Configuration
    legend_show: bool = True
    legend_cols: int = 3
    legend_location: str = "lower center"
    legend_auto_position: bool = True
    legend_frameon: bool = True
    legend_fontsize: int = 12
    legend_title: str | None = None
    legend_markerscale: float | None = None

    # Histogram Configuration
    with_histogram: bool = False
    histogram_rect: tuple[float, float, float, float] = (0.00, 0.05, 0.33, 0.24)
    histogram_col: str | None = None
    histogram_bins: list[float] | None = None
    histogram_bar_colors: list[str] | str | None = None
    histogram_bar_edgecolor: str = "black"
    histogram_bar_linewidth: float = 1.0
    histogram_bar_width: float = 1.0
    histogram_background: str = "white"
    histogram_frame: bool = False
    histogram_grid: bool = False
    histogram_xlabel: str | None = None
    histogram_xticklabels: list[str] | None = None
    histogram_label_rotation: float = 30.0
    histogram_label_fontsize: int = 8
    histogram_show_counts: bool = True
    histogram_count_format: str = "%.0f"
    histogram_count_fontsize: int = 8

    # Basemap Rendering
    basemap_color: str = "grey"
    basemap_edgecolor: str = "black"
    basemap_linewidth: float = 0.3
    basemap_alpha: float = 0.8

    # Output Configuration
    output_path: str | None = None
    dpi: int = 300
    bbox_inches: str = "tight"


def get_russia_projection() -> ccrs.AlbersEqualArea:
    """Return Albers Equal Area CRS optimized for Russia territory.

    Returns:
        Cartopy Albers Equal Area projection with parameters:
        - Central longitude: 100°E
        - Standard parallels: 50°N, 70°N
        - Central latitude: 56°N
        - False easting/northing: 0

    Example:
        >>> aea_crs = get_russia_projection()
        >>> fig, ax = plt.subplots(subplot_kw={"projection": aea_crs})
    """
    return ccrs.AlbersEqualArea(
        central_longitude=100,
        standard_parallels=(50, 70),
        central_latitude=56,
        false_easting=0,
        false_northing=0,
    )


def sort_plot_categories(raw_cats: list[str]) -> list[str]:
    """Sort categories by letter prefix, numeric suffix, or alphabetically.

    This function implements the canonical sorting logic from russia_plots:
    1. If categories have letter prefixes (e.g., "a)", "b)"), sort by prefix
    2. If categories have numeric suffixes (e.g., "Cluster 1", "C10"), sort numerically
    3. Otherwise, sort alphabetically

    Args:
        raw_cats: List of category strings to sort.

    Returns:
        Sorted list of categories.

    Example:
        >>> sort_plot_categories(["Cluster 10", "Cluster 2", "Cluster 1"])
        ['Cluster 1', 'Cluster 2', 'Cluster 10']
        >>> sort_plot_categories(["c) Large", "a) Small", "b) Medium"])
        ['a) Small', 'b) Medium', 'c) Large']
    """

    def _extract_letter_prefix(s: str) -> str | None:
        m = re.match(r"^([a-z])\)", s)
        return m.group(1) if m else None

    def _extract_first_int(s: str) -> int | None:
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else None

    # Try letter prefix sorting
    cats_with_prefix = [(s, _extract_letter_prefix(s)) for s in raw_cats]
    if any(prefix is not None for _, prefix in cats_with_prefix):
        return [
            s
            for s, _ in sorted(cats_with_prefix, key=lambda x: (x[1] is None, x[1] or ""))
        ]

    # Try numeric sorting
    cats_with_nums = [(s, _extract_first_int(s)) for s in raw_cats]
    if any(n is not None for _, n in cats_with_nums):
        return [
            s
            for s, _ in sorted(
                cats_with_nums,
                key=lambda x: (x[1] is None, x[1] if x[1] is not None else x[0]),
            )
        ]

    # Fallback to alphabetical
    return sorted(raw_cats)


def get_unified_colors(
    n_colors: int,
    color_list: list[str] | None = None,
    cmap_name: str = "tab10",
    skip_colors: int = 0,
) -> np.ndarray:
    """Generate unified color array for plotting.

    Implements the canonical color generation logic:
    - If color_list provided, convert to RGBA and cycle if needed
    - Otherwise, sample from colormap with optional skip

    Args:
        n_colors: Number of colors needed.
        color_list: Explicit list of colors (hex, named, or RGB tuples).
        cmap_name: Matplotlib colormap name to sample from.
        skip_colors: Number of colors to skip at start of colormap.

    Returns:
        RGBA color array of shape (n_colors, 4).

    Example:
        >>> colors = get_unified_colors(5, color_list=["red", "blue", "green"])
        >>> colors.shape
        (5, 4)  # RGBA values, cycles if needed
    """
    if color_list and len(color_list) > 0:
        # Convert explicit colors to RGBA
        colors = np.array([mcolors.to_rgba(c) for c in color_list])
        # Cycle if color_list shorter than n_colors
        if len(colors) < n_colors:
            colors = np.tile(colors, (n_colors // len(colors) + 1, 1))[:n_colors]
        return colors[:n_colors]

    # Sample from colormap
    cmap = cm.get_cmap(cmap_name)
    total_samples = n_colors + skip_colors
    colors = np.array(cmap(np.linspace(0.0, 1.0, max(total_samples, 1))))

    if skip_colors > 0:
        colors = colors[skip_colors:]

    return colors[:n_colors]


def get_marker_system(
    n_categories: int,
    markers_list: list[str] | None = None,
    default_markers: list[str] | None = None,
    base_marker_size: int = 30,
    marker_size_corrections: dict[str, float] | None = None,
    marker_size_variants: list[float] | None = None,
    base_linewidth: float = 0.35,
    linewidth_variants: list[float] | None = None,
) -> tuple[list[str], list[int], list[float]]:
    """Generate marker shapes, sizes, and linewidths for categories.

    Implements the canonical marker cycling system from russia_plots:
    - Cycles through marker shapes when categories exceed available markers
    - Applies size/linewidth variations to distinguish repeated shapes
    - Applies per-marker size corrections for visual balance

    Args:
        n_categories: Number of categories to generate markers for.
        markers_list: Custom marker shapes (uses default if None).
        default_markers: Default marker list.
        base_marker_size: Base marker size before scaling.
        marker_size_corrections: Per-marker size adjustments (e.g., {"s": 0.8}).
        marker_size_variants: Size multipliers for repeated markers.
        base_linewidth: Base edge linewidth before scaling.
        linewidth_variants: Linewidth multipliers for repeated markers.

    Returns:
        Tuple of (marker_shapes, marker_sizes, marker_linewidths).

    Example:
        >>> shapes, sizes, lws = get_marker_system(15)
        >>> len(shapes), len(sizes), len(lws)
        (15, 15, 15)
        >>> shapes[0], shapes[12], shapes[13]  # Shows cycling
        ('o', 'o', 's')  # First marker repeats at position 12
    """
    if default_markers is None:
        default_markers = ["o", "s", "^", "v", "<", ">", "P", "X", "D", "*", "h", "H"]

    markers = markers_list if markers_list and len(markers_list) > 0 else default_markers
    n_markers = len(markers)

    if marker_size_variants is None:
        marker_size_variants = [1.0, 1.4, 1.8]

    if linewidth_variants is None:
        linewidth_variants = [1.0, 1.6, 2.2]

    if marker_size_corrections is None:
        marker_size_corrections = {}

    shapes = []
    sizes = []
    linewidths = []

    for i in range(n_categories):
        # Select marker (cycle if needed)
        marker = markers[i % n_markers]
        shapes.append(marker)

        # Calculate repetition index
        repeat_idx = i // n_markers

        # Apply size/linewidth variations
        size_factor = marker_size_variants[repeat_idx % len(marker_size_variants)]
        lw_factor = linewidth_variants[repeat_idx % len(linewidth_variants)]

        # Apply marker-specific size correction
        if marker in marker_size_corrections:
            size_factor *= marker_size_corrections[marker]

        sizes.append(int(base_marker_size * size_factor))
        linewidths.append(float(base_linewidth * lw_factor))

    return shapes, sizes, linewidths


def create_legend_handles(
    categories: list[str],
    colors: np.ndarray,
    markers: list[str] | None = None,
    marker_sizes: list[int] | None = None,
    category_names: dict[str | int, str] | None = None,
    legend_marker_size: int | None = None,
    marker_edgecolor: str = "black",
) -> list[Line2D]:
    """Create custom legend handles for categorical data.

    Generates Line2D handles that display marker+color combinations with
    optional custom naming and sizing.

    Args:
        categories: List of category values.
        colors: RGBA color array (n_categories, 4).
        markers: Marker shapes (uses 'o' if None).
        marker_sizes: Marker sizes (ignored in legend).
        category_names: Dict mapping category values to display names.
        legend_marker_size: Marker size in legend (None = auto-calculate).
        marker_edgecolor: Marker edge color.

    Returns:
        List of Line2D handles for ax.legend().

    Example:
        >>> handles = create_legend_handles(
        ...     categories=["A", "B"],
        ...     colors=np.array([[1, 0, 0, 1], [0, 0, 1, 1]]),
        ...     category_names={"A": "Category A", "B": "Category B"},
        ... )
        >>> ax.legend(handles=handles)
    """
    if legend_marker_size is None:
        legend_marker_size = max(4, int(0.45 * 20))  # russia_plots default

    handles = []
    for i, cat in enumerate(categories):
        marker = markers[i] if markers else "o"
        color = colors[i % len(colors)]

        # Get display label
        if category_names and cat in category_names:
            label = category_names[cat]
        else:
            label = str(cat)

        handle = Line2D(
            [],
            [],
            marker=marker,
            color="black",
            markerfacecolor=color,
            markeredgecolor=marker_edgecolor,
            linestyle="None",
            markersize=legend_marker_size,
            label=label,
        )
        handles.append(handle)

    return handles


def render_basemap(
    ax: Any,
    basemap_gdf: gpd.GeoDataFrame,
    crs: str,
    color: str = "grey",
    edgecolor: str = "black",
    linewidth: float = 0.3,
    alpha: float = 0.8,
) -> None:
    """Render basemap polygons with unified styling.

    Args:
        ax: Matplotlib axes to plot on.
        basemap_gdf: GeoDataFrame with basemap polygons.
        crs: Target CRS (proj4 string).
        color: Fill color.
        edgecolor: Edge color.
        linewidth: Edge width.
        alpha: Transparency.

    Example:
        >>> aea_crs = get_russia_projection()
        >>> fig, ax = plt.subplots(subplot_kw={"projection": aea_crs})
        >>> render_basemap(ax, basemap_data, aea_crs.proj4_init)
    """
    basemap_gdf.to_crs(crs).plot(
        ax=ax,
        color=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        legend=False,
        alpha=alpha,
    )


def add_histogram_inset(
    ax: Any,
    data: pd.Series | gpd.GeoDataFrame,
    column: str,
    bins: list[float] | None = None,
    position: tuple[float, float, float, float] = (0.00, 0.05, 0.33, 0.24),
    bar_colors: list | str | None = None,
    xticklabels: list[str] | None = None,
    bar_edgecolor: str = "black",
    bar_linewidth: float = 1.0,
    bar_width: float = 1.0,
    background: str = "white",
    frame_on: bool = False,
    grid: bool = False,
    xlabel: str | None = None,
    label_rotation: float = 30.0,
    label_fontsize: int = 8,
    show_counts: bool = True,
    count_format: str = "%.0f",
    count_fontsize: int = 8,
) -> None:
    """Add histogram inset to plot with full customization.

    Implements the canonical histogram rendering from russia_plots with support
    for both categorical and continuous data.

    Args:
        ax: Main axes to add histogram to.
        data: Data source (Series or GeoDataFrame).
        column: Column name for histogram data.
        bins: Bin edges for continuous data (None = categorical).
        position: [x, y, width, height] in axes coordinates.
        bar_colors: Bar colors (None = use default).
        xticklabels: Custom x-tick labels.
        bar_edgecolor: Bar edge color.
        bar_linewidth: Bar edge width.
        bar_width: Bar width (0-1).
        background: Background color.
        frame_on: Show frame.
        grid: Show grid.
        xlabel: X-axis label.
        label_rotation: X-tick label rotation.
        label_fontsize: Label font size.
        show_counts: Show bar count labels.
        count_format: Format string for counts.
        count_fontsize: Count label font size.

    Example:
        >>> add_histogram_inset(
        ...     ax,
        ...     gdf,
        ...     "cluster",
        ...     position=(0.0, 0.05, 0.33, 0.24),
        ...     bar_colors=["red", "blue", "green"],
        ... )
    """
    # Create inset axes
    ax_hist = ax.inset_axes(position)

    # Build histogram data
    if bins is None:
        # Categorical histogram
        if isinstance(data, gpd.GeoDataFrame):
            # Get sorted categories
            raw_cats = list(data[column].astype(str).unique())
            sorted_cats = sort_plot_categories(raw_cats)

            # Build histogram with sorted categories
            hist_df = pd.DataFrame()
            for qual in sorted_cats:
                idx = data[data[column].astype(str) == qual].index
                hist_df.loc[0, qual] = len(idx)
        else:
            # Series data
            hist_df = pd.DataFrame([data.value_counts()])
    else:
        # Continuous histogram with bins
        hist_df = pd.crosstab(
            data[column],
            pd.cut(data[column], bins, include_lowest=False),
        )
        hist_df = hist_df.reset_index(drop=True)

    # Determine bar colors
    if bar_colors is None:
        bar_colors = "red"
    elif isinstance(bar_colors, list) and len(bar_colors) < len(hist_df.columns):
        # Cycle colors if too few
        bar_colors = bar_colors * (len(hist_df.columns) // len(bar_colors) + 1)
        bar_colors = bar_colors[: len(hist_df.columns)]

    # Plot bars
    extra_hist = hist_df.sum(axis=0).plot.bar(
        ax=ax_hist,
        rot=label_rotation,
        width=bar_width,
        grid=grid,
        color=bar_colors,
        edgecolor=bar_edgecolor,
        lw=bar_linewidth,
    )

    # Add count labels
    if show_counts:
        extra_hist.bar_label(
            extra_hist.containers[0],  # type: ignore[arg-type]
            fmt=count_format,
            fontsize=count_fontsize if bins is not None else count_fontsize,
        )

    # Styling
    extra_hist.set_facecolor(background)

    if xlabel:
        extra_hist.set_xlabel(xlabel, fontdict={"fontsize": label_fontsize}, loc="right")

    # X-tick labels
    if xticklabels is not None:
        xlbl = xticklabels
    elif bins is not None and isinstance(hist_df.columns, pd.IntervalIndex):
        # Format bin edges
        def _fmt_edge(val: float) -> str:
            val = max(val, bins[0])
            if np.isclose(val, 0.0, atol=1e-6):
                val = 0.0
            return f"{val:.1f}"

        xlbl = [
            f"{_fmt_edge(left)}-{_fmt_edge(right)}"
            for left, right in zip(bins[:-1], bins[1:], strict=False)
        ]
    else:
        xlbl = [str(col).replace(", ", "-") for col in hist_df.columns]

    ax_hist.set(frame_on=frame_on)
    extra_hist.set_xticklabels(xlbl)
    plt.setp(ax_hist.get_xticklabels(), fontsize=label_fontsize)
    plt.setp(ax_hist.get_yticklabels(), fontsize=label_fontsize)


__all__ = [
    "PlotStyleConfig",
    "get_russia_projection",
    "sort_plot_categories",
    "get_unified_colors",
    "get_marker_system",
    "create_legend_handles",
    "render_basemap",
    "add_histogram_inset",
]
