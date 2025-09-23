"""Plotting helpers for hexagon-based aggregations over Russia."""

from __future__ import annotations

from typing import Literal

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib as mpl
from matplotlib import cm
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


def hexes_plot(
    watersheds: gpd.GeoDataFrame,
    basemap_data: gpd.GeoDataFrame,
    metric_col: str,
    r_km: float = 75.0,
    agg: Literal["median", "mean", "max", "min"] = "median",
    area_weighted: bool = False,
    rus_extent: list = [50, 140, 32, 90],
    cmap_name: str = "RdYlGn",
    list_of_limits: list[float] | None = None,
    cmap_lims: tuple = (0.0, 1.0),
    figsize: tuple = (4.88189, 3.34646),
    title_text: str = "",
    legend_shrink: float = 0.35,
    basemap_alpha: float = 0.8,
    annotate_counts: bool = False,
):
    """Plot watershed metrics aggregated to equal-area hexagons over Russia."""
    if metric_col not in watersheds.columns:
        raise KeyError(f"Column '{metric_col}' not found in watersheds GeoDataFrame.")

    watersheds_filtered = watersheds.dropna(subset=[metric_col])
    if watersheds_filtered.empty:
        raise ValueError(f"'{metric_col}' column contains only NaN values.")

    aea_crs = ccrs.AlbersEqualArea(
        central_longitude=100,
        standard_parallels=(50, 70),
        central_latitude=56,
        false_easting=0,
        false_northing=0,
    )
    aea_proj4 = aea_crs.proj4_init

    watersheds_eq = to_equal_area(watersheds_filtered, crs=aea_proj4)
    extent_poly = box(*watersheds_eq.total_bounds)
    hexes = build_hex_grid(extent_poly, r_km=r_km, crs=aea_proj4)
    aggregated_hexes = aggregate_nse_to_hex(
        watersheds_eq[[metric_col, "geometry"]],
        hexes,
        nse_col=metric_col,
        agg=agg,
        area_weighted=area_weighted,
    )

    metric_label = f"{agg}_{metric_col}"
    if aggregated_hexes.empty or aggregated_hexes[metric_label].isna().all():
        raise ValueError("No watershed geometries intersect the generated hex grid.")

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": aea_crs})
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_extent(rus_extent)  # type: ignore

    basemap_data.to_crs(aea_proj4).plot(
        ax=ax, color="grey", edgecolor="black", alpha=basemap_alpha, linewidth=0.4
    )

    if list_of_limits:
        cmap = cm.get_cmap(cmap_name, len(list_of_limits) - 1)
        norm = mpl.colors.BoundaryNorm(list_of_limits, len(list_of_limits) - 1)
    else:
        cmap = cm.get_cmap(cmap_name)
        vmin, vmax = cmap_lims
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    hexes_to_plot = aggregated_hexes.to_crs(aea_proj4)
    plot = hexes_to_plot.plot(
        ax=ax,
        column=metric_label,
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.2,
        legend=True,
        legend_kwds={
            "orientation": "horizontal",
            "shrink": legend_shrink,
            "pad": -0.075,
            "anchor": (0.6, 0.5),
            "drawedges": True,
        },
    )

    if annotate_counts:
        centroids = hexes_to_plot.geometry.centroid
        for x, y, count in zip(centroids.x, centroids.y, hexes_to_plot["count"].values, strict=False):
            ax.text(x, y, str(count), fontsize=6, ha="center", va="center")

    plot.figure.axes[-1].tick_params(labelsize=8)
    ax.set_title(title_text, fontsize=12)

    return plot.figure, aggregated_hexes


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
    rus_extent: list = [50, 140, 32, 90],
    list_of_limits: list[float] | None = None,
    cmap_lims: tuple = (0.0, 1.0),
    cmap_name: str = "RdYlGn",
    figsize: tuple = (9.76378, 6.69291),
    basemap_alpha: float = 0.8,
    annotate_counts: bool = False,
    with_histogram: bool = False,
    hist_name: list[str] | None = None,
    title_text: list[str] | None = None,
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
        norm = mpl.colors.BoundaryNorm(list_of_limits, len(list_of_limits) - 1)
    else:
        cmap = cm.get_cmap(cmap_name)
        vmin, vmax = cmap_lims
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    basemap_proj = basemap_data.to_crs(aea_proj4)

    for idx, metric in enumerate(metric_cols):
        if idx >= len(axs_flat):
            break
        ax = axs_flat[idx]

        ws_metric = watersheds_eq.dropna(subset=[metric])
        if ws_metric.empty:
            raise ValueError(f"Metric column '{metric}' contains only NaN values.")

        aggregated = aggregate_nse_to_hex(
            ws_metric[[metric, "geometry"]],
            hex_grid,
            nse_col=metric,
            agg=agg,
            area_weighted=area_weighted,
        )
        metric_label = f"{agg}_{metric}"

        metric_hexes[metric] = aggregated.to_crs(aea_proj4)
        coverage_stats[metric] = summarize_hex_coverage(metric_hexes[metric])

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_extent(rus_extent)  # type: ignore
        basemap_proj.plot(ax=ax, color="grey", edgecolor="black", alpha=basemap_alpha, linewidth=0.4)

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
                centroids.x, centroids.y, metric_hexes[metric]["count"].values, strict=False
            ):
                ax.text(x, y, str(int(count)), fontsize=6, ha="center", va="center")

        cax = inset_axes(
            ax,
            width="55%",
            height="4%",
            loc="lower center",
            bbox_to_anchor=(0.0, -0.12, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8)

        if with_histogram:
            if list_of_limits is None:
                raise ValueError("list_of_limits must be provided when with_histogram=True.")

            values = ws_metric[metric].dropna().clip(lower=0)
            if not values.empty:
                bins = list_of_limits
                categories = pd.cut(values, bins=bins, include_lowest=True, right=True)
                hist_counts = categories.value_counts(sort=False)

                ax_hist = ax.inset_axes([0.00, 0.05, 0.33, 0.24])
                bar_ax = hist_counts.plot.bar(
                    ax=ax_hist,
                    rot=0,
                    width=1,
                    facecolor="red",
                    edgecolor="black",
                    lw=1,
                )
                if bar_ax.containers:
                    bar_ax.bar_label(bar_ax.containers[0], fmt="%.0f")
                bar_ax.set_facecolor("white")
                bar_ax.tick_params(width=1)
                bar_ax.grid(False)

                if hist_name is not None and idx < len(hist_name):
                    bar_ax.set_xlabel(hist_name[idx], fontdict={"fontsize": 8}, loc="right")

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
                if len(xlbl) > 4:
                    xlbl = [lbl if not (lbl_idx % 2) else "" for lbl_idx, lbl in enumerate(xlbl)]

                bar_ax.set_xticklabels(xlbl)
                ax_hist.set(frame_on=False)

                plt.setp(ax_hist.get_xticklabels(), fontsize=8)
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

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    return fig, metric_hexes, r_km, coverage_stats


__all__ = ["hexes_plot", "hexes_plots_n"]
