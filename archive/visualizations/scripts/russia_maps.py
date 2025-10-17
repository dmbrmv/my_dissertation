"""Plotting helpers for Russia-wide visualisations."""

from __future__ import annotations

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def russia_plots(
    gdf_to_plot: gpd.GeoDataFrame,
    basemap_data: gpd.GeoDataFrame,
    distinction_col: str,
    specific_xlabel: list | None = None,
    title_text: str = "",
    rus_extent: list = [50, 140, 32, 90],
    list_of_limits: list = [0.0, 0.4, 0.6, 0.8, 1.0],
    cmap_lims: tuple = (0, 1),
    cmap_name: str = "RdYlGn",
    metric_col: str = "",
    figsize: tuple = (4.88189, 3.34646),
    just_points: bool = False,
    legend_cols: int = 3,
    with_histogram: bool = False,
    ugms: bool = False,
    ugms_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(),
):
    specific_xlabel = specific_xlabel or []

    aea_crs = ccrs.AlbersEqualArea(
        central_longitude=100,
        standard_parallels=(50, 70),
        central_latitude=56,
        false_easting=0,
        false_northing=0,
    )
    aea_crs_proj4 = aea_crs.proj4_init

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": aea_crs})
    if not just_points:
        cmap = cm.get_cmap(cmap_name, 5)
        vmin, vmax = cmap_lims
        norm_cmap = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        cmap = cm.get_cmap(cmap_name, 18)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_extent(rus_extent)  # type: ignore
    if not ugms:
        basemap_data.to_crs(aea_crs_proj4).plot(
            ax=ax, color="grey", edgecolor="black", legend=False, alpha=0.8
        )
    gdf_to_plot = gdf_to_plot.to_crs(aea_crs_proj4)  # type: ignore

    if just_points:
        scatter_plot = gdf_to_plot.plot(
            ax=ax,
            column=distinction_col,
            cmap=cmap,
            marker="o",
            markersize=15,
            legend=True,
            legend_kwds={
                "ncol": legend_cols,
                "loc": "lower right",
                "fmt": "{:.0f}",
                "fontsize": 14,
                "markerscale": 0.5,
                "frameon": True,
            },
        )
    else:
        if ugms:
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
        scatter_plot = gdf_to_plot.plot(
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

    my_fig = scatter_plot.figure
    if not just_points:
        cb_ax = my_fig.axes[1]
        cb_ax.tick_params(labelsize=8)

    if with_histogram:
        if distinction_col:
            hist_df = pd.DataFrame()
            for qual, idx in gdf_to_plot.groupby(f"{distinction_col}").groups.items():
                hist_df.loc[0, f"{qual}"] = len(idx)
        else:
            hist_df = pd.crosstab(
                gdf_to_plot[metric_col],
                pd.cut(gdf_to_plot[metric_col], list_of_limits, include_lowest=False),
            )
            hist_df = hist_df.reset_index(drop=True)

        ax_hist = ax.inset_axes([0.00, 0.05, 0.33, 0.24])

        extra_hist = hist_df.sum(axis=0).plot.bar(
            ax=ax_hist,
            rot=15,
            width=1,
            grid=False,
            facecolor="red",
            edgecolor="black",
            lw=1,
        )
        if distinction_col:
            extra_hist.bar_label(extra_hist.containers[0], fmt="%.0f")
        else:
            extra_hist.bar_label(extra_hist.containers[0], fmt="%.0f", fontsize=8)
        extra_hist.set_facecolor("white")

        extra_hist.set_xlabel(f"{metric_col}", fontdict={"fontsize": 8}, loc="right")
        if len(specific_xlabel) == 0:
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

    ax.set_title(title_text, fontdict={"size": 12})
    plt.tight_layout()

    return fig


def russia_plots_n(
    gdf_to_plot: gpd.GeoDataFrame,
    basemap_data: gpd.GeoDataFrame,
    columns_from_gdf: list,
    label_list: list,
    nrows: int,
    ncols: int,
    title_text: list = [""],
    hist_name: list = [""],
    rus_extent: list = [50, 140, 32, 90],
    list_of_limits: list = [0.0, 0.5, 0.7, 0.8, 1.0],
    cmap_lims: tuple = (0, 1),
    cmap_name: str = "RdYlGn",
    figsize: tuple = (4.88189, 3.34646),
    just_points: bool = False,
    with_histogram: bool = False,
    ugms: bool = False,
    ugms_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(),
):
    aea_crs = ccrs.AlbersEqualArea(
        central_longitude=100,
        standard_parallels=(50, 70),
        central_latitude=56,
        false_easting=0,
        false_northing=0,
    )
    aea_crs_proj4 = aea_crs.proj4_init

    fig, axs = plt.subplots(
        figsize=figsize, ncols=ncols, nrows=nrows, subplot_kw={"projection": aea_crs}
    )

    for i, ax in enumerate(np.ravel(axs)):
        cmap = cm.get_cmap(cmap_name, len(list_of_limits) - 1)
        vmin, vmax = cmap_lims
        norm_cmap = mpl.colors.BoundaryNorm(list_of_limits, len(list_of_limits) - 1)
        if i >= len(columns_from_gdf):
            ax.set_visible(False)
            break

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_extent(rus_extent)  # type: ignore
        if not ugms:
            basemap_data.to_crs(aea_crs_proj4).plot(
                ax=ax,
                color="grey",
                edgecolor="black",
                legend=False,
                alpha=0.8,
                linewidth=0.6,
            )
        gdf_proj = gdf_to_plot.to_crs(aea_crs_proj4)  # type: ignore

        if just_points:
            scatter_plot = gdf_proj.plot(
                ax=ax,
                column=columns_from_gdf[i],
                cmap=cmap,
                norm=norm_cmap,
                marker="o",
                markersize=10,
                legend=True,
                legend_kwds={
                    "loc": "lower center",
                    "ncol": 2,
                    "fmt": "{:.0f}",
                    "fontsize": 8,
                },
            )
        else:
            if ugms:
                ugms_gdf.to_crs(aea_crs_proj4).plot(
                    ax=ax,
                    column=columns_from_gdf[i],
                    cmap=cmap,
                    norm=norm_cmap,
                    legend=False,
                    edgecolor="black",
                    linewidth=0.6,
                    missing_kwds={"color": "#DF60DF00"},
                )
            scatter_plot = gdf_proj.plot(
                ax=ax,
                column=columns_from_gdf[i],
                cmap=cmap,
                norm=norm_cmap,
                marker="o",
                markersize=14,
                edgecolor="black",
                linewidth=0.4,
                legend=True,
                legend_kwds={
                    "orientation": "horizontal",
                    "shrink": 0.3,
                    "pad": -0.05,
                    "anchor": (0.6, 0.5),
                    "drawedges": True,
                },
            )
        ax.text(
            0,
            1,
            label_list[i],
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=14,
        )
        my_fig = scatter_plot.figure
        if not just_points:
            cb_ax = my_fig.axes[nrows * ncols + i]
            cb_ax.tick_params(labelsize=10)

        if with_histogram:
            if just_points:
                hist_df = pd.DataFrame()
                for qual, idx in gdf_proj.groupby(
                    f"{columns_from_gdf[i]}"
                ).groups.items():
                    hist_df.loc[0, f"{qual}"] = len(idx)
            else:
                hist_df = pd.crosstab(
                    gdf_proj[columns_from_gdf[i]],
                    pd.cut(
                        gdf_proj[columns_from_gdf[i]],
                        list_of_limits,
                        include_lowest=False,
                    ),
                )

            hist_df = hist_df.reset_index(drop=True)
            hist_df.columns.name = hist_name[i]
            ax_hist = ax.inset_axes([0.00, 0.05, 0.33, 0.24])
            extra_hist = hist_df.sum(axis=0).plot.bar(
                ax=ax_hist, rot=0, width=1, facecolor="red", edgecolor="black", lw=1
            )
            extra_hist.bar_label(extra_hist.containers[0], fmt="%.0f")
            extra_hist.set_facecolor("white")
            extra_hist.tick_params(width=1)
            extra_hist.grid(False)

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
                if just_points:
                    xlbl = [str(col).replace(", ", "-") for col in hist_df.columns]
                else:
                    xlbl = [str(col)[1:-1].replace(", ", "-") for col in hist_df.columns]
            if len(xlbl) > 4:
                xlbl = [lbl if not (idx % 2) else "" for idx, lbl in enumerate(xlbl)]

            ax_hist.set(frame_on=False)
            extra_hist.set_xticklabels(xlbl)

            plt.setp(ax_hist.get_xticklabels(), fontsize=8)
            plt.setp(ax_hist.get_yticklabels(), fontsize=8)
        ax.set_title(f"{title_text[i]}", fontdict={"size": 12})
        plt.tight_layout()

    return fig


__all__ = ["russia_plots", "russia_plots_n"]
