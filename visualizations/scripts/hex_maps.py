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
        norm = mpl.colors.BoundaryNorm(list_of_limits, len(list_of_limits) - 1)
    else:
        cmap = cm.get_cmap(cmap_name)
        vmin, vmax = cmap_lims
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    if cb_label is not None:
        if list_of_limits is None:
            raise ValueError("cb_label requires list_of_limits to define colorbar segments.")
        if len(cb_label) != len(list_of_limits) - 1:
            raise ValueError("cb_label length must match the number of colorbar segments.")

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
            loc="lower right",
            bbox_to_anchor=(0.0, 0.05, 0.95, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        color_mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        color_mapper.set_array([])
        cbar = fig.colorbar(color_mapper, cax=cax, orientation="horizontal", drawedges=True)
        cbar.ax.tick_params(labelsize=8)

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
                raise ValueError("list_of_limits must be provided when with_histogram=True.")

            values = ws_metric[metric].dropna().clip(lower=0)
            if not values.empty:
                bins = list_of_limits
                categories = pd.cut(values, bins=bins, include_lowest=True, right=True)
                hist_counts = categories.value_counts(sort=False)

                ax_hist = ax.inset_axes([0.05, 0.05, 0.30, 0.24])

                edges = list_of_limits
                bin_midpoints = [
                    (left + right) / 2 for left, right in zip(edges[:-1], edges[1:], strict=False)
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

                ax_hist.set_xticks(np.arange(len(xlbl)))
                ax_hist.set_xticklabels(xlbl)
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
    skip_colors: int = 0,
    rus_extent: tuple = (50, 140, 32, 90),
    figsize: tuple = (10.0, 5.0),
    basemap_alpha: float = 0.8,
    legend: bool = True,
    legend_columns: int = 2,
    legend_kwargs: dict | None = None,
    histogram: bool = True,
    histogram_rect: tuple[float, float, float, float] = (0.05, 0.05, 0.30, 0.24),
    histogram_label_rotation: float = 0.0,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes, gpd.GeoDataFrame, pd.Series]:
    """Visualise the dominant model per hexagon along with category counts."""
    if model_col not in watersheds.columns:
        raise KeyError(f"Column '{model_col}' not found in watersheds GeoDataFrame.")
    if watersheds.empty:
        raise ValueError("Watersheds GeoDataFrame is empty.")

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

    counts = intersections.groupby(["hex_idx", model_col]).size().reset_index(name="count")
    totals = counts.groupby("hex_idx", as_index=False)["count"].sum().rename(columns={"count": "total"})
    counts = counts.merge(totals, on="hex_idx")
    counts.loc[:, "freq"] = counts["count"] / counts["total"]

    top = counts.sort_values(["hex_idx", "freq"], ascending=[True, False]).drop_duplicates("hex_idx")
    top_idx = top.set_index("hex_idx")

    hex_grid.loc[:, "dominant_model"] = hex_grid["hex_idx"].map(top_idx[model_col])
    hex_grid.loc[:, "dominant_share"] = hex_grid["hex_idx"].map(top_idx["freq"])

    if ambiguous_label is not None:
        mask = hex_grid["dominant_share"].notna() & (hex_grid["dominant_share"] < dominant_threshold)
        hex_grid.loc[mask, "dominant_model"] = ambiguous_label

    hex_grid.loc[hex_grid["dominant_share"].isna(), "dominant_model"] = np.nan

    label_order = [label for label in model_dict.keys() if isinstance(label, str)]
    if ambiguous_label and ambiguous_label not in label_order:
        label_order.append(ambiguous_label)

    present_labels = [
        lbl for lbl in hex_grid["dominant_model"].dropna().unique() if isinstance(lbl, str)
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

    base_cmap = mpl.colormaps[cmap_name]
    required_colors = len(label_order)
    sampled = base_cmap(np.linspace(0, 1, required_colors + skip_colors))
    if skip_colors:
        sampled = sampled[skip_colors:]
    if len(sampled) < required_colors:
        raise ValueError("Not enough colours sampled from colormap. Reduce skip_colors.")
    sampled = sampled[:required_colors]
    cmap = colors.ListedColormap(sampled, name=f"{cmap_name}_models")
    bounds = np.arange(required_colors + 1) - 0.5
    norm = mpl.colors.BoundaryNorm(bounds, required_colors)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": aea_crs})
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_extent(rus_extent)  # type: ignore

    basemap_data.to_crs(aea_proj4).plot(
        ax=ax, color="grey", edgecolor="black", alpha=basemap_alpha, linewidth=0.4
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

    if legend:
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
            "ncol": legend_columns,
            "columnspacing": 0.8,
            "handletextpad": 0.35,
        }
        if legend_kwargs:
            legend_defaults.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_defaults)

    category_counts = (
        valid_hexes["dominant_model"].value_counts().reindex(label_order, fill_value=0).rename("count")
    )

    if histogram:
        ax_hist = ax.inset_axes(histogram_rect)
        bars = ax_hist.bar(
            np.arange(required_colors),
            category_counts.values,
            width=0.9,
            color=sampled,
            edgecolor="black",
            linewidth=1,
        )
        ax_hist.bar_label(bars, fmt="%d")
        ax_hist.set_facecolor("white")
        ax_hist.tick_params(width=1)
        ax_hist.grid(False)
        ax_hist.set_xticks(np.arange(required_colors))
        ax_hist.set_xticklabels(
            label_order,
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


__all__ = ["hexes_plot", "hexes_plots_n", "hex_model_distribution_plot"]
