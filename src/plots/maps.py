import re

import cartopy.crs as ccrs
import geopandas as gpd
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _get_aea_crs() -> ccrs.AlbersEqualArea:
    """Return Albers Equal Area CRS for Russia."""
    return ccrs.AlbersEqualArea(
        central_longitude=100,
        standard_parallels=(50, 70),
        central_latitude=56,
        false_easting=0,
        false_northing=0,
    )


def _sort_categories(raw_cats: list[str]) -> list[str]:
    """Sort categories by letter prefix, numeric suffix, or alphabetically.

    Args:
        raw_cats: List of category strings to sort.

    Returns:
        Sorted list of categories.
    """

    def _extract_letter_prefix(s: str) -> str | None:
        m = re.match(r"^([a-z])\)", s)
        return m.group(1) if m else None

    def _extract_first_int(s: str) -> int | None:
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else None

    cats_with_prefix = [(s, _extract_letter_prefix(s)) for s in raw_cats]
    if any(prefix is not None for _, prefix in cats_with_prefix):
        return [s for s, _ in sorted(cats_with_prefix, key=lambda x: (x[1] is None, x[1] or ""))]

    cats_with_nums = [(s, _extract_first_int(s)) for s in raw_cats]
    if any(n is not None for _, n in cats_with_nums):
        return [
            s
            for s, _ in sorted(
                cats_with_nums,
                key=lambda x: (x[1] is None, x[1] if x[1] is not None else x[0]),
            )
        ]

    return sorted(raw_cats)


def _plot_basemap(ax, basemap_data, aea_crs_proj4):
    """Plot basemap polygons."""
    basemap_data.to_crs(aea_crs_proj4).plot(
        ax=ax, color="grey", edgecolor="black", legend=False, alpha=0.8
    )


def _plot_points(
    ax,
    gdf_to_plot,
    distinction_col,
    cmap,
    legend_cols,
    markers_list: list | None = None,
    color_list: list | None = None,
    base_marker_size: int = 30,
    base_linewidth: float = 0.35,
    legend_loc: str = "lower center",
    marker_size_corrections: dict | None = None,
):
    """Plot points with distinction column using unique marker+color combos.

    Supports passing an explicit `markers_list` or `color_list`. If the number
    of categories exceeds the number of available marker shapes, the function
    varies marker size and edge linewidth deterministically so repeated shapes
    remain distinguishable.

    Args:
        ax: Matplotlib axis to plot on.
        gdf_to_plot: GeoDataFrame with points to plot.
        distinction_col: Column name for categorizing points.
        cmap: Matplotlib colormap for default colors.
        legend_cols: Number of columns in legend.
        markers_list: Optional list of marker symbols to use.
        color_list: Optional list of colors to use.
        base_marker_size: Base size for markers before scaling.
        base_linewidth: Base edge linewidth for markers.
        legend_loc: Legend location string (e.g., "lower center").
        marker_size_corrections: Optional dict mapping marker symbols to size
            correction factors (e.g., {"s": 0.8, "^": 1.15}) to compensate for
            visual size differences between marker types.
    """
    # Ensure distinction column is string-like for consistent grouping
    raw_cats = list(gdf_to_plot[distinction_col].astype(str).unique())

    # Sort categories using common logic
    cats = _sort_categories(raw_cats)
    n = len(cats)

    # Markers: use provided list or fall back to a conservative set of filled markers
    default_markers = ["o", "s", "^", "v", "<", ">", "P", "X", "D", "*", "h", "H"]
    markers = markers_list if markers_list and len(markers_list) > 0 else default_markers
    n_markers = len(markers)

    # Colors: prefer explicit color_list, otherwise sample the provided colormap
    if color_list and len(color_list) > 0:
        # convert to RGBA array
        colors = np.array([mcolors.to_rgba(c) for c in color_list])
    else:
        colors = np.array(cmap(np.linspace(0.0, 1.0, max(n, 1))))

    # Variation tiers applied when markers must repeat; deterministic and cyclic
    size_variants = [1.0, 1.4, 1.8]
    linewidth_variants = [1.0, 1.6, 2.2]

    handles = []
    # Plot each category separately so we can control marker shape and color
    for i, cat in enumerate(cats):
        sub = gdf_to_plot[gdf_to_plot[distinction_col].astype(str) == cat]
        if sub.empty:
            continue
        xs = sub.geometry.x
        ys = sub.geometry.y

        # Choose marker and compute repetition index for deterministic variations
        marker = markers[i % n_markers]
        repeat_idx = i // n_markers
        size_factor = size_variants[repeat_idx % len(size_variants)]
        lw_factor = linewidth_variants[repeat_idx % len(linewidth_variants)]

        # Apply marker-specific size correction if provided
        if marker_size_corrections and marker in marker_size_corrections:
            size_factor *= marker_size_corrections[marker]

        # Choose color; cycle if color list shorter than n categories
        col = colors[i % len(colors)]

        size = int(base_marker_size * size_factor)
        linewidth = float(base_linewidth * lw_factor)

        ax.scatter(
            xs,
            ys,
            marker=marker,
            s=size,
            c=[col],
            edgecolor="black",
            linewidth=linewidth,
            zorder=3,
        )

        # Create a legend handle that shows marker+color and smaller size than map markers
        legend_size = max(4, int(0.45 * 20))
        handles.append(
            Line2D(
                [],
                [],
                marker=marker,
                color="black",
                markerfacecolor=col,
                markeredgecolor="black",
                linestyle="None",
                markersize=legend_size,
                label=str(cat),
            )
        )

    # Draw legend using our custom handles
    if handles:
        ax.legend(
            handles=handles,
            title=distinction_col,
            ncol=legend_cols,
            loc=legend_loc,
            frameon=True,
        )

    return ax


def _plot_polygons(ax, gdf_to_plot, metric_col, cmap, norm_cmap):
    """Plot polygons with metric column."""
    return gdf_to_plot.plot(
        ax=ax,
        column=metric_col,
        cmap=cmap,
        norm=norm_cmap,
        marker="o",
        markersize=24,
        edgecolor="black",
        linewidth=0.2,
        legend=True,
        legend_kwds={
            "orientation": "horizontal",
            "shrink": 0.35,
            "pad": -0.075,
            "anchor": (0.6, 0.5),
            "drawedges": True,
        },
    )


def _plot_ugms(ax, ugms_gdf, metric_col, cmap, norm_cmap, aea_crs_proj4):
    """Plot UGMS polygons."""
    ugms_gdf.to_crs(aea_crs_proj4).plot(
        ax=ax,
        column=metric_col,
        cmap=cmap,
        norm=norm_cmap,
        legend=False,
        edgecolor="black",
        linewidth=0.6,
        missing_kwds={"color": "#DF60DF00"},
    )


def _add_histogram(
    ax, gdf_to_plot, distinction_col, metric_col, list_of_limits, specific_xlabel, color_list=None
):
    """Add histogram inset to plot.

    Args:
        ax: Matplotlib axis to add histogram to.
        gdf_to_plot: GeoDataFrame with data to plot.
        distinction_col: Column name for categorizing data.
        metric_col: Metric column for non-categorical histograms.
        list_of_limits: Bin limits for continuous data.
        specific_xlabel: Custom x-axis labels.
        color_list: Optional list of colors matching the categories in distinction_col.
    """
    if distinction_col:
        # Get sorted categories using the same logic as _plot_points
        raw_cats = list(gdf_to_plot[distinction_col].astype(str).unique())
        sorted_cats = _sort_categories(raw_cats)

        # Build histogram with sorted categories
        hist_df = pd.DataFrame()
        for qual in sorted_cats:
            idx = gdf_to_plot[gdf_to_plot[distinction_col].astype(str) == qual].index
            hist_df.loc[0, qual] = len(idx)
    else:
        hist_df = pd.crosstab(
            gdf_to_plot[metric_col],
            pd.cut(gdf_to_plot[metric_col], list_of_limits, include_lowest=False),
        )
        hist_df = hist_df.reset_index(drop=True)

    ax_hist = ax.inset_axes([0.00, 0.05, 0.33, 0.24])

    # Use provided colors if available, otherwise default to red
    bar_colors = color_list if color_list and len(color_list) >= len(hist_df.columns) else "red"

    extra_hist = hist_df.sum(axis=0).plot.bar(
        ax=ax_hist,
        rot=30,
        width=1,
        grid=False,
        color=bar_colors,
        edgecolor="black",
        lw=1,
    )
    if distinction_col:
        extra_hist.bar_label(extra_hist.containers[0], fmt="%.0f")  # type: ignore[arg-type]
    else:
        extra_hist.bar_label(extra_hist.containers[0], fmt="%.0f", fontsize=8)  # type: ignore[arg-type]
    extra_hist.set_facecolor("white")
    extra_hist.set_xlabel(f"{metric_col}", fontdict={"fontsize": 8}, loc="right")

    if not specific_xlabel:
        if isinstance(hist_df.columns, pd.IntervalIndex):
            edges = list_of_limits

            def _fmt_edge(val: float) -> str:
                val = max(val, edges[0])
                if np.isclose(val, 0.0, atol=1e-6):
                    val = 0.0
                return f"{val:.1f}"

            xlbl = [
                f"{_fmt_edge(left)}-{_fmt_edge(right)}"
                for left, right in zip(edges[:-1], edges[1:], strict=False)
            ]
        else:
            xlbl = [str(col).replace(", ", "-") for col in hist_df.columns]
    else:
        xlbl = specific_xlabel

    ax_hist.set(frame_on=False)
    extra_hist.set_xticklabels(xlbl)
    plt.setp(ax_hist.get_xticklabels(), fontsize=8)
    plt.setp(ax_hist.get_yticklabels(), fontsize=8)


def russia_plots(
    gdf_to_plot: gpd.GeoDataFrame,
    basemap_data: gpd.GeoDataFrame,
    distinction_col: str,
    specific_xlabel: list | None = None,
    title_text: str = "",
    rus_extent: list | None = None,
    list_of_limits: list | None = None,
    cmap_lims: tuple = (0, 1),
    cmap_name: str = "RdYlGn",
    metric_col: str = "",
    figsize: tuple = (4.88189, 3.34646),
    just_points: bool = False,
    legend_cols: int = 3,
    with_histogram: bool = False,
    ugms: bool = False,
    ugms_gdf: gpd.GeoDataFrame | None = None,
    markers_list: list | None = None,
    color_list: list | None = None,
    base_marker_size: int = 30,
    base_linewidth: float = 0.35,
    marker_size_corrections: dict | None = None,
):
    """Plot Russia map with points or polygons and optional histogram."""
    specific_xlabel = specific_xlabel or []
    aea_crs = _get_aea_crs()
    aea_crs_proj4 = aea_crs.proj4_init
    if rus_extent is None:
        rus_extent = [50, 140, 32, 90]
    if list_of_limits is None:
        list_of_limits = [0.0, 0.4, 0.6, 0.8, 1.0]
    if ugms_gdf is None:
        ugms_gdf = gpd.GeoDataFrame()

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": aea_crs})
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_extent(rus_extent)  # type: ignore

    if not ugms:
        _plot_basemap(ax, basemap_data, aea_crs_proj4)
    gdf_to_plot = gdf_to_plot.to_crs(aea_crs_proj4)  # type: ignore

    if just_points:
        cmap = cm.get_cmap(cmap_name, 18)
        # Determine legend location based on histogram presence
        legend_loc = "lower right" if with_histogram else "lower center"
        # forward optional marker/color customizations to the low-level plot
        scatter_plot = _plot_points(
            ax,
            gdf_to_plot,
            distinction_col,
            cmap,
            legend_cols,
            markers_list=markers_list,
            color_list=color_list,
            base_marker_size=base_marker_size,
            base_linewidth=base_linewidth,
            legend_loc=legend_loc,
            marker_size_corrections=marker_size_corrections,
        )
    else:
        cmap = cm.get_cmap(cmap_name, 5)
        vmin, vmax = cmap_lims
        norm_cmap = mcolors.Normalize(vmin=vmin, vmax=vmax)
        if ugms:
            _plot_ugms(ax, ugms_gdf, metric_col, cmap, norm_cmap, aea_crs_proj4)
        scatter_plot = _plot_polygons(ax, gdf_to_plot, metric_col, cmap, norm_cmap)

    my_fig = scatter_plot.figure
    if not just_points:
        cb_ax = my_fig.axes[1]
        cb_ax.tick_params(labelsize=8)

    if with_histogram:
        _add_histogram(
            ax,
            gdf_to_plot,
            distinction_col,
            metric_col,
            list_of_limits,
            specific_xlabel,
            color_list=color_list,
        )

    ax.set_title(title_text, fontdict={"size": 12})
    plt.tight_layout()
    return fig
