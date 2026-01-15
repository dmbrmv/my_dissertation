"""Plotting helpers for hexagon-based aggregations over Russia."""

from __future__ import annotations

from typing import Literal

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib as mpl
from matplotlib import cm, colors
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from shapely.geometry import box

from .hex_utils import (
    aggregate_nse_to_hex,
    build_hex_grid,
    suggest_hex_radius,
    summarize_hex_coverage,
    to_equal_area,
)


def hexes_plots_n(
    watersheds: gpd.GeoDataFrame,
    basemap_data: gpd.GeoDataFrame,
    metric_cols: list[str],
    label_list: list[str],
    nrows: int,
    ncols: int,
    r_km: float | None = None,
    target_ws_per_hex: float | None = None,
    quantile: float = 0.5,
    min_r_km: float = 40.0,
    max_r_km: float = 120.0,
    agg: Literal["median", "mean", "max", "min"] = "median",
    area_weighted: bool = False,
    min_overlap_share: float = 0.1,
    negative_threshold: float | None = None,
    rus_extent: tuple = (50, 140, 32, 90),
    list_of_limits: list[float] | None = None,
    cmap_lims: tuple = (0.0, 1.0),
    cmap_name: str = "RdYlGn",
    figsize: tuple = (9.76378, 6.69291),
    basemap_alpha: float = 0.8,
    annotate_counts: bool = False,
    with_histogram: bool = False,
    title_text: list[str] | None = None,
    cb_label: list[str] | None = None,
) -> tuple[plt.Figure, dict[str, gpd.GeoDataFrame], float, dict[str, dict[str, float]]]:
    """Plot multiple hex-aggregated metrics in a grid similar to ``russia_plots_n``."""
    if not metric_cols:
        raise ValueError("metric_cols must contain at least one column name.")
    if len(label_list) < len(metric_cols):
        raise ValueError("label_list must provide at least one label per metric.")

    if r_km is None:
        if target_ws_per_hex is not None:
            aea_proj = ccrs.AlbersEqualArea(
                central_longitude=100,
                standard_parallels=(50, 70),
                central_latitude=56,
                false_easting=0,
                false_northing=0,
            ).proj4_init
            r_km = suggest_hex_radius(
                watersheds,
                target_ws_per_hex=target_ws_per_hex,
                quantile=quantile,
                min_r_km=min_r_km,
                max_r_km=max_r_km,
                crs=aea_proj,
            )
        else:
            r_km = 75.0

    aea_crs = ccrs.AlbersEqualArea(
        central_longitude=100,
        standard_parallels=(50, 70),
        central_latitude=56,
        false_easting=0,
        false_northing=0,
    )
    aea_proj4 = aea_crs.proj4_init

    watersheds_eq = to_equal_area(watersheds, crs=aea_proj4)
    watersheds_eq.loc[:, "orig_area"] = watersheds_eq.geometry.area
    extent_poly = box(*watersheds_eq.total_bounds)
    hex_grid = build_hex_grid(extent_poly, r_km=r_km, crs=aea_proj4)

    fig, axs = plt.subplots(
        figsize=figsize, ncols=ncols, nrows=nrows, subplot_kw={"projection": aea_crs}
    )
    axs_flat = np.array(axs).ravel()

    metric_hexes: dict[str, gpd.GeoDataFrame] = {}
    coverage_stats: dict[str, dict[str, float]] = {}

    if list_of_limits:
        cmap = cm.get_cmap(cmap_name, len(list_of_limits) - 1)
        # Replace inf values with finite bounds for BoundaryNorm
        finite_limits = (
            list_of_limits.copy()
            if isinstance(list_of_limits, list)
            else list(list_of_limits)
        )
        for i, val in enumerate(finite_limits):
            if np.isinf(val) and val < 0:
                finite_limits[i] = -1e10
            elif np.isinf(val) and val > 0:
                finite_limits[i] = 1e10
        norm = mpl.colors.BoundaryNorm(finite_limits, len(finite_limits) - 1)
    else:
        cmap = cm.get_cmap(cmap_name)
        vmin, vmax = cmap_lims
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    if cb_label is not None:
        if list_of_limits is None:
            raise ValueError(
                "cb_label requires list_of_limits to define colorbar segments."
            )
        if len(cb_label) != len(list_of_limits) - 1:
            raise ValueError(
                "cb_label length must match the number of colorbar segments."
            )

    basemap_proj = basemap_data.to_crs(aea_proj4)

    for idx, metric in enumerate(metric_cols):
        if idx >= len(axs_flat):
            break
        ax = axs_flat[idx]

        ws_metric = watersheds_eq.dropna(subset=[metric])
        if ws_metric.empty:
            raise ValueError(f"Metric column '{metric}' contains only NaN values.")

        aggregated = aggregate_nse_to_hex(
            ws_metric[[metric, "geometry", "orig_area"]],
            hex_grid,
            nse_col=metric,
            agg=agg,
            area_weighted=area_weighted,
            min_overlap_share=min_overlap_share,
            negative_threshold=negative_threshold,
        )
        metric_label = f"{agg}_{metric}"

        metric_hexes[metric] = aggregated.to_crs(aea_proj4)
        coverage_stats[metric] = summarize_hex_coverage(metric_hexes[metric])

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_extent(rus_extent)  # type: ignore
        basemap_proj.plot(
            ax=ax, color="grey", edgecolor="black", alpha=basemap_alpha, linewidth=0.4
        )

        metric_hexes[metric].plot(
            ax=ax,
            column=metric_label,
            cmap=cmap,
            norm=norm,
            edgecolor="black",
            linewidth=0.2,
            legend=False,
            missing_kwds={"color": "#DF60DF00"},
        )

        if annotate_counts:
            centroids = metric_hexes[metric].geometry.centroid
            for x, y, count in zip(
                centroids.x,
                centroids.y,
                metric_hexes[metric]["count"].values,
                strict=False,
            ):
                ax.text(x, y, str(int(count)), fontsize=6, ha="center", va="center")

        cax = inset_axes(
            ax,
            width="55%",
            height="4%",
            loc="lower right",
            bbox_to_anchor=(0.0, 0.05, 0.95, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        color_mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        color_mapper.set_array([])
        cbar = fig.colorbar(
            color_mapper, cax=cax, orientation="horizontal", drawedges=True
        )
        cbar.ax.tick_params(labelsize=8)

        # Fix colorbar tick labels: replace large negative values with -∞ symbol
        if list_of_limits is not None:
            current_ticks = cbar.get_ticks()
            new_labels = []
            for t in current_ticks:
                if t < -1e9:  # Very negative = represents -∞
                    new_labels.append("-∞")
                elif t > 1e9:  # Very positive = represents ∞
                    new_labels.append("∞")
                elif t == int(t):
                    new_labels.append(f"{int(t)}")
                else:
                    new_labels.append(f"{t:.1f}")
            cbar.ax.set_xticklabels(new_labels)

        if cb_label is not None and list_of_limits is not None:
            n_segments = len(cb_label)
            for label_idx, label_text in enumerate(cb_label):
                pos = (label_idx + 0.5) / max(n_segments, 1)
                cbar.ax.text(
                    pos,
                    1.12,
                    label_text,
                    ha="center",
                    va="bottom",
                    transform=cbar.ax.transAxes,
                    fontsize=9,
                    clip_on=False,
                )

        if with_histogram:
            if list_of_limits is None:
                raise ValueError(
                    "list_of_limits must be provided when with_histogram=True."
                )

            values = ws_metric[metric].dropna()
            if not values.empty:
                bins = list_of_limits
                categories = pd.cut(values, bins=bins, include_lowest=True, right=True)
                hist_counts = categories.value_counts(sort=False)

                ax_hist = ax.inset_axes([0.05, 0.05, 0.30, 0.24])

                edges = list_of_limits
                # Use finite limits for midpoint calculation (color mapping)
                finite_edges = [
                    -1e10
                    if (np.isinf(e) and e < 0)
                    else (1e10 if (np.isinf(e) and e > 0) else e)
                    for e in edges
                ]
                bin_midpoints = [
                    (left + right) / 2
                    for left, right in zip(
                        finite_edges[:-1], finite_edges[1:], strict=False
                    )
                ]
                colors = [color_mapper.to_rgba(midpoint) for midpoint in bin_midpoints]

                bars = ax_hist.bar(
                    np.arange(len(hist_counts)),
                    hist_counts.values,
                    width=1,
                    color=colors,
                    edgecolor="black",
                    linewidth=1,
                )
                ax_hist.bar_label(bars, fmt="%.0f")
                ax_hist.set_facecolor("white")
                ax_hist.tick_params(width=1)
                ax_hist.grid(False)

                # if hist_name is not None and idx < len(hist_name):
                #     ax_hist.set_xlabel(hist_name[idx], fontdict={"fontsize": 8}, loc="center")

                def _fmt_edge(val: float) -> str:
                    if np.isinf(val) and val < 0:
                        return "-∞"
                    if np.isinf(val) and val > 0:
                        return "∞"
                    val = max(val, edges[0])
                    if np.isclose(val, 0.0, atol=1e-6):
                        val = 0.0
                    return f"{val:.1f}"

                xlbl = [
                    f"{_fmt_edge(left)}-{_fmt_edge(right)}"
                    for left, right in zip(edges[:-1], edges[1:], strict=False)
                ]

                ax_hist.set_xticks(np.arange(len(xlbl)))
                ax_hist.set_xticklabels(xlbl)
                ax_hist.set(frame_on=False)

                plt.setp(ax_hist.get_xticklabels(), fontsize=8, rotation=30)
                plt.setp(ax_hist.get_yticklabels(), fontsize=8)

        if title_text is not None and idx < len(title_text):
            ax.set_title(title_text[idx], fontsize=12)

        if idx < len(label_list):
            ax.text(
                0,
                1,
                label_list[idx],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=14,
            )

    for idx in range(len(metric_cols), len(axs_flat)):
        axs_flat[idx].set_visible(False)

    # Increase vertical spacing when multiple rows of maps are drawn so inset
    hspace = 0.05 if nrows == 1 else 0.25
    fig.subplots_adjust(wspace=0.05, hspace=hspace)
    return fig, metric_hexes, r_km, coverage_stats


def hex_model_distribution_plot(
    watersheds: gpd.GeoDataFrame,
    basemap_data: gpd.GeoDataFrame,
    model_col: str,
    model_dict: dict[str, float],
    *,
    min_overlap_share: float = 0.15,
    dominant_threshold: float = 0.33,
    ambiguous_label: str = "Неоднозначно",
    r_km: float | None = None,
    target_ws_per_hex: float | None = 6.0,
    quantile: float = 0.5,
    min_r_km: float = 40.0,
    max_r_km: float = 150.0,
    cmap_name: str = "turbo",
    color_list: list[str] | None = None,
    skip_colors: int = 0,
    rus_extent: tuple = (50, 140, 32, 90),
    figsize: tuple = (10.0, 5.0),
    basemap_color: str = "grey",
    basemap_edgecolor: str = "black",
    basemap_linewidth: float = 0.4,
    basemap_alpha: float = 0.8,
    legend_show: bool = True,
    legend_cols: int = 2,
    legend_kwargs: dict | None = None,
    with_histogram: bool = True,
    histogram_col: str | None = None,
    histogram_rect: tuple[float, float, float, float] = (0.05, 0.05, 0.30, 0.24),
    histogram_xticklabels: list[str] | None = None,
    histogram_label_rotation: float = 0.0,
    histogram_count_format: str = "%d",
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes, gpd.GeoDataFrame, pd.Series]:
    """Visualize the dominant model per hexagon along with category counts.

    Now aligned with unified plotting infrastructure: uses consistent parameter
    naming (with_histogram, legend_cols) and supports explicit color lists.

    Args:
        watersheds: GeoDataFrame with watershed geometries.
        basemap_data: GeoDataFrame for basemap boundaries.
        model_col: Column name containing model classifications.
        model_dict: Dict mapping model labels to numeric codes.
        min_overlap_share: Minimum watershed overlap fraction to include.
        dominant_threshold: Minimum frequency for non-ambiguous classification.
        ambiguous_label: Label for hexes below dominant threshold.
        r_km: Hex radius in km (None = auto-calculate).
        target_ws_per_hex: Target watersheds per hex for radius calculation.
        quantile: Quantile for radius calculation.
        min_r_km: Minimum hex radius.
        max_r_km: Maximum hex radius.
        cmap_name: Colormap name (used if color_list not provided).
        color_list: Explicit list of colors (overrides cmap_name).
        skip_colors: Number of colors to skip when sampling colormap.
        rus_extent: Map extent [lon_min, lon_max, lat_min, lat_max].
        figsize: Figure size (width, height) in inches.
        basemap_color: Basemap polygon fill color.
        basemap_edgecolor: Basemap polygon edge color.
        basemap_linewidth: Basemap polygon edge width.
        basemap_alpha: Basemap transparency.
        legend_show: Whether to show legend.
        legend_cols: Number of columns in legend.
        legend_kwargs: Additional legend kwargs (overrides defaults).
        with_histogram: Show histogram inset.
        histogram_col: Column for histogram (None = use model_col).
        histogram_rect: Position [x, y, width, height] in axes coordinates.
        histogram_xticklabels: Custom x-tick labels.
        histogram_label_rotation: X-tick label rotation angle.
        histogram_count_format: Format string for bar count labels.
        title: Plot title.

    Returns:
        Tuple of (figure, axes, hex_grid_gdf, category_counts_series).
    """
    from src.plots.styling_utils import get_russia_projection, get_unified_colors

    if model_col not in watersheds.columns:
        raise KeyError(f"Column '{model_col}' not found in watersheds GeoDataFrame.")
    if watersheds.empty:
        raise ValueError("Watersheds GeoDataFrame is empty.")

    # Setup projection using shared helper
    aea_crs = get_russia_projection()
    aea_proj4 = aea_crs.proj4_init

    if r_km is None:
        if target_ws_per_hex is not None:
            r_km = suggest_hex_radius(
                watersheds,
                target_ws_per_hex=target_ws_per_hex,
                quantile=quantile,
                min_r_km=min_r_km,
                max_r_km=max_r_km,
                crs=aea_proj4,
            )
        else:
            r_km = 75.0

    watersheds_eq = to_equal_area(watersheds, crs=aea_proj4).copy()
    watersheds_eq.loc[:, "orig_area"] = watersheds_eq.geometry.area
    watersheds_eq.loc[:, "watershed_idx"] = watersheds_eq.index

    ws_valid = watersheds_eq.dropna(subset=[model_col])
    if ws_valid.empty:
        raise ValueError(f"'{model_col}' column contains only NaN values.")

    extent_poly = box(*ws_valid.total_bounds)
    hex_grid = build_hex_grid(extent_poly, r_km=r_km, crs=aea_proj4).copy()
    hex_grid.loc[:, "hex_idx"] = hex_grid.index

    intersections = gpd.overlay(
        ws_valid[["watershed_idx", model_col, "orig_area", "geometry"]],
        hex_grid[["hex_idx", "geometry"]],
        how="intersection",
    )
    if intersections.empty:
        raise ValueError("No watershed overlaps found for the constructed hex grid.")

    intersections.loc[:, "intersect_area"] = intersections.geometry.area
    intersections = intersections[
        intersections["intersect_area"] >= min_overlap_share * intersections["orig_area"]
    ]
    if intersections.empty:
        raise ValueError(
            "No watershed overlaps exceed the minimum overlap share. Consider lowering the threshold."
        )

    counts = (
        intersections.groupby(["hex_idx", model_col]).size().reset_index(name="count")
    )
    totals = (
        counts.groupby("hex_idx", as_index=False)["count"]
        .sum()
        .rename(columns={"count": "total"})
    )
    counts = counts.merge(totals, on="hex_idx")
    counts.loc[:, "freq"] = counts["count"] / counts["total"]

    top = counts.sort_values(
        ["hex_idx", "freq"], ascending=[True, False]
    ).drop_duplicates("hex_idx")
    top_idx = top.set_index("hex_idx")

    hex_grid.loc[:, "dominant_model"] = hex_grid["hex_idx"].map(top_idx[model_col])
    hex_grid.loc[:, "dominant_share"] = hex_grid["hex_idx"].map(top_idx["freq"])

    if ambiguous_label is not None:
        mask = hex_grid["dominant_share"].notna() & (
            hex_grid["dominant_share"] < dominant_threshold
        )
        hex_grid.loc[mask, "dominant_model"] = ambiguous_label

    hex_grid.loc[hex_grid["dominant_share"].isna(), "dominant_model"] = np.nan

    label_order = [label for label in model_dict.keys() if isinstance(label, str)]
    if ambiguous_label and ambiguous_label not in label_order:
        label_order.append(ambiguous_label)

    present_labels = [
        lbl
        for lbl in hex_grid["dominant_model"].dropna().unique()
        if isinstance(lbl, str)
    ]
    for lbl in present_labels:
        if lbl not in label_order:
            label_order.append(lbl)
    label_order = list(dict.fromkeys(label_order))

    if not label_order:
        raise ValueError("No categorical labels available for plotting.")

    label_to_idx = {label: idx for idx, label in enumerate(label_order)}
    hex_grid.loc[:, "model_idx"] = hex_grid["dominant_model"].map(label_to_idx)
    hex_grid.loc[:, "model_code"] = hex_grid["dominant_model"].map(model_dict)

    # Use unified color generation
    required_colors = len(label_order)
    sampled = get_unified_colors(
        required_colors,
        color_list=color_list,
        cmap_name=cmap_name,
        skip_colors=skip_colors,
    )
    cmap = colors.ListedColormap(sampled, name=f"{cmap_name}_models")
    bounds = np.arange(required_colors + 1) - 0.5
    norm = mpl.colors.BoundaryNorm(bounds, required_colors)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": aea_crs})
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_extent(rus_extent)  # type: ignore

    basemap_data.to_crs(aea_proj4).plot(
        ax=ax,
        color=basemap_color,
        edgecolor=basemap_edgecolor,
        alpha=basemap_alpha,
        linewidth=basemap_linewidth,
    )

    hexes_to_plot = hex_grid.to_crs(aea_proj4)
    valid_hexes = hexes_to_plot.loc[hexes_to_plot["model_idx"].notna()].copy()
    if valid_hexes.empty:
        raise ValueError("No hex cells remain after applying dominance filtering.")

    valid_hexes.plot(
        ax=ax,
        column="model_idx",
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.2,
        legend=False,
        missing_kwds={"color": "#DF60DF00"},
    )

    if legend_show:
        legend_handles = [
            Line2D(
                [],
                [],
                marker="h",
                linestyle="",
                markerfacecolor=sampled[label_to_idx[label]],
                markeredgecolor="black",
                markeredgewidth=0.8,
                markersize=11,
                label=label,
            )
            for label in label_order
        ]
        legend_defaults = {
            "loc": "center left",
            "bbox_to_anchor": (1.02, 0.5),
            "bbox_transform": ax.transAxes,
            "frameon": False,
            "borderpad": 0.3,
            "ncol": legend_cols,
            "columnspacing": 0.8,
            "handletextpad": 0.35,
        }
        if legend_kwargs:
            legend_defaults.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_defaults)

    category_counts = (
        valid_hexes["dominant_model"]
        .value_counts()
        .reindex(label_order, fill_value=0)
        .rename("count")
    )

    if with_histogram:
        ax_hist = ax.inset_axes(histogram_rect)
        bars = ax_hist.bar(
            np.arange(required_colors),
            category_counts.values,
            width=0.9,
            color=sampled,
            edgecolor="black",
            linewidth=1,
        )
        ax_hist.bar_label(bars, fmt=histogram_count_format)
        ax_hist.set_facecolor("white")
        ax_hist.tick_params(width=1)
        ax_hist.grid(False)
        ax_hist.set_xticks(np.arange(required_colors))

        # Use custom labels if provided, otherwise use label_order
        xticklabels = histogram_xticklabels if histogram_xticklabels else label_order
        ax_hist.set_xticklabels(
            xticklabels,
            rotation=histogram_label_rotation,
            ha="right" if histogram_label_rotation else "center",
        )
        ax_hist.set(frame_on=False)
        plt.setp(ax_hist.get_xticklabels(), fontsize=8)
        plt.setp(ax_hist.get_yticklabels(), fontsize=8)

    if title:
        ax.set_title(title, fontsize=12)

    fig.subplots_adjust(right=0.78)
    return fig, ax, hexes_to_plot, category_counts


def hex_model_distribution_plots_n(
    watersheds: gpd.GeoDataFrame,
    basemap_data: gpd.GeoDataFrame,
    model_cols: list[str],
    model_dict: dict[str, float] | list[dict[str, float]],
    *,
    label_list: list[str] | None = None,
    nrows: int = 2,
    ncols: int = 2,
    min_overlap_share: float = 0.15,
    dominant_threshold: float = 0.33,
    ambiguous_label: str = "Неоднозначно",
    r_km: float | None = None,
    target_ws_per_hex: float | None = 6.0,
    quantile: float = 0.5,
    min_r_km: float = 40.0,
    max_r_km: float = 150.0,
    color_list: list[str] | list[list[str]] | None = None,
    cmap_name: str = "turbo",
    rus_extent: tuple = (50, 140, 32, 90),
    figsize: tuple = (18, 14),
    basemap_color: str = "grey",
    basemap_edgecolor: str = "black",
    basemap_linewidth: float = 0.4,
    basemap_alpha: float = 0.8,
    title_text: list[str] | None = None,
    with_histogram: bool = True,
    histogram_rect: list[float] | None = None,
    histogram_label_rotation: float = 20,
    legend_show: bool = True,
    legend_kwargs: dict | None = None,
) -> tuple[plt.Figure, np.ndarray, dict[str, gpd.GeoDataFrame], dict[str, pd.Series]]:
    """Create multi-panel hex model distribution plots in a grid layout.

    Each subplot includes its own histogram and legend, matching the style
    of hex_model_distribution_plot.

    Args:
        watersheds: GeoDataFrame with watershed geometries and model columns
        basemap_data: GeoDataFrame for basemap boundaries
        model_cols: List of column names containing model classifications
        model_dict: Dict mapping model labels to numeric codes, OR a list of dicts
            (one per column) for heterogeneous categories across subplots
        label_list: Panel labels (e.g., ["а)", "б)", "в)", "г)"])
        nrows: Number of rows in subplot grid
        ncols: Number of columns in subplot grid
        min_overlap_share: Minimum watershed overlap fraction
        dominant_threshold: Minimum frequency for non-ambiguous classification
        ambiguous_label: Label for ambiguous hexes
        r_km: Hex radius in km (None = auto-calculate)
        target_ws_per_hex: Target watersheds per hex
        quantile: Quantile for radius calculation
        min_r_km: Minimum hex radius
        max_r_km: Maximum hex radius
        color_list: Explicit list of colors, OR a list of color lists (one per column)
            for heterogeneous categories across subplots
        cmap_name: Colormap name (used if color_list not provided)
        rus_extent: Map extent [lon_min, lon_max, lat_min, lat_max]
        figsize: Figure size (width, height)
        basemap_color: Basemap fill color
        basemap_edgecolor: Basemap edge color
        basemap_linewidth: Basemap line width
        basemap_alpha: Basemap transparency
        title_text: List of titles for each subplot
        with_histogram: Whether to show histogram on each subplot
        histogram_rect: Position [x, y, width, height] for histogram inset
        histogram_label_rotation: Rotation angle for histogram x-tick labels
        legend_show: Whether to show legend on each subplot
        legend_kwargs: Additional kwargs for legend

    Returns:
        Tuple of (figure, axes_array, dict_of_hex_gdfs, dict_of_category_counts)
    """
    from shapely.geometry import box

    from .hex_utils import build_hex_grid, suggest_hex_radius, to_equal_area
    from .styling_utils import get_russia_projection, get_unified_colors

    if len(model_cols) > nrows * ncols:
        raise ValueError(
            f"Too many model columns ({len(model_cols)}) for grid ({nrows}x{ncols})"
        )

    if histogram_rect is None:
        histogram_rect = [0.02, 0.02, 0.25, 0.35]

    # Normalize model_dict to list format
    if isinstance(model_dict, dict):
        model_dicts = [model_dict] * len(model_cols)
    else:
        if len(model_dict) != len(model_cols):
            raise ValueError(
                f"model_dict list length ({len(model_dict)}) must match "
                f"model_cols length ({len(model_cols)})"
            )
        model_dicts = model_dict

    # Normalize color_list to list-of-lists format
    if color_list is None:
        color_lists = [None] * len(model_cols)
    elif isinstance(color_list[0], str):
        # Single color list for all columns
        color_lists = [color_list] * len(model_cols)  # type: ignore
    else:
        # List of color lists
        if len(color_list) != len(model_cols):
            raise ValueError(
                f"color_list length ({len(color_list)}) must match "
                f"model_cols length ({len(model_cols)})"
            )
        color_lists = color_list  # type: ignore

    # Setup projection
    aea_crs = get_russia_projection()
    aea_proj4 = aea_crs.proj4_init

    # Create figure with subplots
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, subplot_kw={"projection": aea_crs}
    )
    axes_flat = np.array(axes).flatten()

    # Prepare data once
    ws_eq = to_equal_area(watersheds, crs=aea_proj4).copy()
    ws_eq.loc[:, "orig_area"] = ws_eq.geometry.area
    extent_poly = box(*ws_eq.total_bounds)

    if r_km is None:
        r_km = suggest_hex_radius(
            watersheds,
            target_ws_per_hex=target_ws_per_hex,
            quantile=quantile,
            min_r_km=min_r_km,
            max_r_km=max_r_km,
            crs=aea_proj4,
        )

    hex_grid = build_hex_grid(extent_poly, r_km=r_km, crs=aea_proj4)
    basemap_proj = basemap_data.to_crs(aea_proj4)

    # Store results
    hex_results = {}
    category_counts_dict = {}

    # Plot each model column
    for idx, model_col in enumerate(model_cols):
        if idx >= len(axes_flat):
            break

        ax = axes_flat[idx]
        ws_data = ws_eq[[model_col, "geometry", "orig_area"]].copy()

        # Get per-column dict and colors
        curr_model_dict = model_dicts[idx]
        curr_color_list = color_lists[idx]

        # Setup colors for this subplot
        label_order = [lbl for lbl in curr_model_dict.keys() if isinstance(lbl, str)]
        if ambiguous_label and ambiguous_label not in label_order:
            label_order.append(ambiguous_label)

        n_colors = len(label_order)
        if curr_color_list:
            colors_to_use = curr_color_list[:n_colors]
        else:
            colors_to_use = get_unified_colors(n_colors, cmap_name=cmap_name)

        cmap = colors.ListedColormap(colors_to_use, name=f"cat_{idx}")
        bounds = np.arange(n_colors + 1) - 0.5
        norm = mpl.colors.BoundaryNorm(bounds, n_colors)
        label_to_idx = {lbl: i for i, lbl in enumerate(label_order)}

        # Aggregate to hexagons
        intersections = gpd.overlay(
            ws_data,
            hex_grid.reset_index().rename(columns={"index": "hex_idx"})[
                ["hex_idx", "geometry"]
            ],
            how="intersection",
        )
        intersections.loc[:, "intersect_area"] = intersections.geometry.area
        intersections = intersections[
            intersections["intersect_area"]
            >= min_overlap_share * intersections["orig_area"]
        ]

        # Find dominant model per hex
        counts = (
            intersections.groupby(["hex_idx", model_col]).size().reset_index(name="count")
        )
        totals = counts.groupby("hex_idx")["count"].sum().reset_index(name="total")
        counts = counts.merge(totals, on="hex_idx")
        counts.loc[:, "freq"] = counts["count"] / counts["total"]

        top = counts.sort_values(
            ["hex_idx", "freq"], ascending=[True, False]
        ).drop_duplicates("hex_idx")

        hex_plot = hex_grid.copy()
        hex_plot.loc[:, "dominant"] = hex_plot.index.map(
            top.set_index("hex_idx")[model_col]
        )
        hex_plot.loc[:, "dominant_share"] = hex_plot.index.map(
            top.set_index("hex_idx")["freq"]
        )

        # Mark ambiguous
        if ambiguous_label:
            mask = hex_plot["dominant_share"].notna() & (
                hex_plot["dominant_share"] < dominant_threshold
            )
            hex_plot.loc[mask, "dominant"] = ambiguous_label

        # Map to indices
        hex_plot.loc[:, "model_idx"] = hex_plot["dominant"].map(label_to_idx)

        # Store results
        hex_results[model_col] = hex_plot.to_crs(aea_proj4)

        # Plot
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_extent(rus_extent)  # type: ignore

        basemap_proj.plot(
            ax=ax,
            color=basemap_color,
            edgecolor=basemap_edgecolor,
            alpha=basemap_alpha,
            linewidth=basemap_linewidth,
        )

        valid = hex_plot.loc[hex_plot["model_idx"].notna()].to_crs(aea_proj4)
        if not valid.empty:
            valid.plot(
                ax=ax,
                column="model_idx",
                cmap=cmap,
                norm=norm,
                edgecolor="black",
                linewidth=0.2,
                legend=False,
            )

        # Calculate category counts for this subplot
        category_counts = (
            valid["dominant"]
            .value_counts()
            .reindex(label_order, fill_value=0)
            .rename("count")
        )
        category_counts_dict[model_col] = category_counts

        # Add histogram inset
        if with_histogram:
            ax_hist = ax.inset_axes(histogram_rect)
            bars = ax_hist.bar(
                np.arange(n_colors),
                category_counts.values,
                width=0.9,
                color=colors_to_use,
                edgecolor="black",
                linewidth=1,
            )
            ax_hist.bar_label(bars, fmt="%d")
            ax_hist.set_facecolor("white")
            ax_hist.tick_params(width=1)
            ax_hist.grid(False)
            ax_hist.set_xticks(np.arange(n_colors))
            ax_hist.set_xticklabels(
                label_order,
                rotation=histogram_label_rotation,
                ha="right" if histogram_label_rotation else "center",
            )
            ax_hist.set(frame_on=False)
            plt.setp(ax_hist.get_xticklabels(), fontsize=8)
            plt.setp(ax_hist.get_yticklabels(), fontsize=8)

        # Add per-subplot legend
        if legend_show:
            legend_handles = [
                Line2D(
                    [],
                    [],
                    marker="h",
                    linestyle="",
                    markerfacecolor=colors_to_use[i],
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    markersize=11,
                    label=label,
                )
                for i, label in enumerate(label_order)
            ]
            legend_defaults = {
                "loc": "lower right",
                "bbox_to_anchor": (1.0, 0.0),
                "frameon": False,
                "ncol": 2,
                "columnspacing": 0.8,
                "handletextpad": 0.35,
                "fontsize": 9,
            }
            if legend_kwargs:
                legend_defaults.update(legend_kwargs)
            ax.legend(handles=legend_handles, **legend_defaults)

        # Add title
        if title_text and idx < len(title_text):
            ax.set_title(title_text[idx], fontsize=12)

        # Add label
        if label_list and idx < len(label_list):
            ax.text(
                0,
                1,
                label_list[idx],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=14,
            )

    # Hide unused subplots
    for idx in range(len(model_cols), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()

    return fig, axes, hex_results, category_counts_dict


__all__ = [
    "hexes_plots_n",
    "hex_model_distribution_plot",
    "hex_model_distribution_plots_n",
]
