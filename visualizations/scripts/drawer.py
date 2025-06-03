import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


def create_nse_boxplots(df, nse_columns, figsize=(14, 8), color="lightblue", save_path=None, max_cols=3):
    """Create boxplots for NSE columns in a dataframe.

    Args:
        df: DataFrame containing NSE columns
        nse_columns: List of column names containing NSE values
        figsize: Tuple of figure size (width, height)
        color: Color for the boxplots
        save_path: Optional path to save the figure as PNG (e.g., 'output/nse_boxplots.png')
        max_cols: Maximum number of columns in the subplot grid (default: 3)

    Returns:
        matplotlib.figure.Figure: The created figure

    """
    n_cols = len(nse_columns)
    ncols = min(n_cols, max_cols)  # Maximum 3 columns
    nrows = (n_cols + ncols - 1) // ncols  # Calculate required rows

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Ensure axes is always a list for consistent indexing
    if n_cols == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Russian alphabet letters for labeling
    russian_labels = ["а)", "б)", "в)", "г)", "д)", "е)", "ж)", "з)", "и)", "к)"]

    for i, col in enumerate(nse_columns):
        ax = axes[i]

        # Calculate statistics
        median_val = df[col].median()
        total_gauges = len(df[col].dropna())
        satisfactory = (df[col] >= 0.5).sum()
        satisfactory_pct = satisfactory / total_gauges * 100 if total_gauges > 0 else 0

        # Create boxplot
        df[col].plot.box(
            ax=ax,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker="o", markerfacecolor="red", markersize=4, alpha=0.5),
        )

        # Customize the plot
        title = f"Медиана: {median_val:.2f}, NSE≥0.5: {satisfactory} ({satisfactory_pct:.1f}%)"
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_ylabel("NSE", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Add Russian alphabet label in top right corner
        if i < len(russian_labels):
            ax.text(
                0.95,
                0.95,
                russian_labels[i],
                transform=ax.transAxes,
                fontsize=16,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        # Limit NSE axis to 0-1.0 for better visibility
        ax.set_ylim(0, 1.0)

        # Add horizontal line at NSE = 0.5
        ax.axhline(
            y=0.5,
            color="orange",
            linestyle="--",
            alpha=0.8,
            linewidth=2,
        )

    # Hide unused subplots
    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    return fig


def russia_plots(
    gdf_to_plot: gpd.GeoDataFrame,
    basemap_data: gpd.GeoDataFrame,
    distinction_col: str,
    specific_xlabel: list = [],
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
    # RUSSIA
    aea_crs = ccrs.AlbersEqualArea(
        central_longitude=100,
        standard_parallels=(50, 70),
        central_latitude=56,
        false_easting=0,
        false_northing=0,
    )
    # to proj4 form
    aea_crs_proj4 = aea_crs.proj4_init

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": aea_crs})
    if not just_points:
        cmap = cm.get_cmap(cmap_name, 5)
        vmin, vmax = cmap_lims
        norm_cmap = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        cmap = cm.get_cmap(cmap_name, 18)
    # plot settings
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_extent(rus_extent)  # type: ignore
    if not ugms:
        # plot russia basemap
        basemap_data.to_crs(aea_crs_proj4).plot(  # type: ignore
            ax=ax, color="grey", edgecolor="black", legend=False, alpha=0.8
        )
    gdf_to_plot = gdf_to_plot.to_crs(aea_crs_proj4)  # type: ignore
    # plot variable
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
        # cb_ax.xaxis.set_major_locator(
        #     mticker.FixedLocator([i for i in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]))
        # cb_ax.set_xticklabels([f'{i}'
        #                        for i in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])

    if with_histogram:
        if distinction_col:
            hist_df = pd.DataFrame()
            for qual, idx in gdf_to_plot.groupby(f"{distinction_col}").groups.items():
                hist_df.loc[0, f"{qual}"] = len(idx)
        else:
            # list_of_limits_hist = [0, 250, 500, 1000, 2000]
            hist_df = pd.crosstab(
                gdf_to_plot[metric_col],
                pd.cut(gdf_to_plot[metric_col], list_of_limits, include_lowest=False),
            )

            hist_df = hist_df.reset_index(drop=True)

        # x of borders, y of borders, weight, height
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
            # xlbl = [str(col)[1:-1].replace(', ', '-')
            #         for col in hist_df.columns]
            xlbl = [str(col).replace(", ", "-") for col in hist_df.columns]

            # def str_to_float(x):
            #     return list(map(float, x.split('-')))

            # # def round_1_digit(x):
            # #     return str(int(x))

            # def str_joiner(x):
            #     return '-'.join(x)
            # xlbl = [str_joiner([y for y in str_to_float(x)]) for x in xlbl]
        else:
            xlbl = specific_xlabel

        ax_hist.set(frame_on=False)
        extra_hist.set_xticklabels(xlbl)
        extra_hist.tick_params(width=1)

        plt.setp(ax_hist.get_xticklabels(), fontsize=8)
        plt.setp(ax_hist.get_yticklabels(), fontsize=8)

    plt.title(f"{title_text}", fontdict={"size": 12})

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
    # RUSSIA
    aea_crs = ccrs.AlbersEqualArea(
        central_longitude=100,
        standard_parallels=(50, 70),
        central_latitude=56,
        false_easting=0,
        false_northing=0,
    )
    # to proj4 form
    aea_crs_proj4 = aea_crs.proj4_init

    fig, axs = plt.subplots(
        figsize=figsize, ncols=ncols, nrows=nrows, subplot_kw={"projection": aea_crs}
    )

    for i, ax in enumerate(np.ravel(axs)):
        cmap = cm.get_cmap(cmap_name, len(list_of_limits) - 1)
        vmin, vmax = cmap_lims
        # [-100, -75, -50, -25, 0, 25, 50, 75, 100]
        # [0.0, 0.50, 0.70, 0.80, 1.00]
        # bounds = [-100, -75, -50, -25, 0, 25, 50, 75, 100]

        norm_cmap = mpl.colors.BoundaryNorm(list_of_limits, len(list_of_limits) - 1)
        # norm_cmap = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        if i >= len(columns_from_gdf):
            ax.set_visible(False)
            break
        # plot settings
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_extent(rus_extent)  # type: ignore
        if not ugms:
            # plot russia basemap
            basemap_data.to_crs(aea_crs_proj4).plot(  # type: ignore
                ax=ax,
                color="grey",
                edgecolor="black",
                legend=False,
                alpha=0.8,
                linewidth=0.6,
            )
        gdf_to_plot = gdf_to_plot.to_crs(aea_crs_proj4)  # type: ignore
        # plot variable
        if just_points:
            scatter_plot = gdf_to_plot.plot(
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
                    # missing_kwds=dict(color="black", aspect="equal"),
                    missing_kwds={"color": "#DF60DF00"},
                )
            scatter_plot = gdf_to_plot.plot(
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
            # [-100, -75, -50, -25, 0,
            # 25, 50, 75, 100]
            # [0.00, 0.25, .50, .75, 1.00]
            # cb_ax.xaxis.set_major_locator(
            #     mticker.FixedLocator([i for i
            #                           in list_of_limits]))
            # cb_ax.set_xticklabels([f'{i:.0f}'
            #                        for i in [0.00, 0.25, .50, .75, 1.00]])

        if with_histogram:
            if just_points:
                hist_df = pd.DataFrame()
                for qual, idx in gdf_to_plot.groupby(f"{columns_from_gdf[i]}").groups.items():
                    hist_df.loc[0, f"{qual}"] = len(idx)
            else:
                hist_df = pd.crosstab(
                    gdf_to_plot[columns_from_gdf[i]],  # type: ignore
                    pd.cut(
                        gdf_to_plot[columns_from_gdf[i]],  # type: ignore
                        list_of_limits,
                        include_lowest=False,
                    ),
                )

            hist_df = hist_df.reset_index(drop=True)
            hist_df.columns.name = hist_name[i]
            # x of borders, y of borders, weight, height
            ax_hist = ax.inset_axes([0.00, 0.05, 0.33, 0.24])
            extra_hist = hist_df.sum(axis=0).plot.bar(
                ax=ax_hist, rot=0, width=1, facecolor="red", edgecolor="black", lw=1
            )
            extra_hist.bar_label(extra_hist.containers[0], fmt="%.0f")
            extra_hist.set_facecolor("white")
            extra_hist.tick_params(width=1)
            # extra_hist.set_xlabel(f'{columns_from_gdf[i]}',
            #                       fontdict={'fontsize': 14}, loc='right')
            extra_hist.grid(False)

            if just_points:
                xlbl = [str(col).replace(", ", "-") for col in hist_df.columns]
            else:
                xlbl = [str(col)[1:-1].replace(", ", "-") for col in hist_df.columns]
            if len(xlbl) > 4:
                xlbl = [lbl if not (i % 2) else "" for i, lbl in enumerate(xlbl)]

            # def str_to_float(x):
            #     return list(map(float, x.split('-')))

            # def round_1_digit(x):
            #     return str(round(x, 1))

            # def str_joiner(x):
            #     return '-'.join(x)
            # xlbl = [str_joiner([round_1_digit(y)
            #                     for y in str_to_float(x)]) for x in xlbl]
            ax_hist.set(frame_on=False)
            extra_hist.set_xticklabels(xlbl)

            plt.setp(ax_hist.get_xticklabels(), fontsize=8)
            plt.setp(ax_hist.get_yticklabels(), fontsize=8)
        ax.set_title(f"{title_text[i]}", fontdict={"size": 12})
        plt.tight_layout()

    return fig


def metric_viewer(gauges_file: gpd.GeoDataFrame, metric_col: str, metric_csv: str):
    model_metric = pd.read_csv(metric_csv)
    model_metric = model_metric.rename(columns={"basin": "gauge_id", "gauge": "gauge_id"})
    model_metric["gauge_id"] = model_metric["gauge_id"].astype("str")
    model_metric = model_metric.set_index("gauge_id")
    if "gauge_id" not in gauges_file.columns:
        res_file = gauges_file.join(model_metric).dropna()
    else:
        res_file = gauges_file.set_index("gauge_id").join(model_metric).dropna()
    nse_median = res_file[metric_col].median()
    # res_file.loc[res_file[metric_col] < 0, metric_col] = 0

    return res_file, nse_median
