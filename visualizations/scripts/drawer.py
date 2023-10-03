import geopandas as gpd
import cartopy.crs as ccrs
from cartopy.mpl import geoaxes
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"


def russia_plots(gdf_to_plot: gpd.GeoDataFrame,
                 basemap_data: gpd.GeoDataFrame,
                 distinction_col: str,
                 specific_xlabel: list = [],
                 title_text: str = '',
                 rus_extent: list = [50, 140, 32, 90],
                 list_of_limits: list = [0.0, 0.4, 0.6, 0.8, 1.0],
                 cmap_lims: tuple = (0, 1),
                 cmap_name: str = 'RdYlGn',
                 metric_col: str = '',
                 figsize: tuple = (4.88189, 3.34646),
                 just_points: bool = False,
                 with_histogram: bool = False,
                 ugms: bool = False,
                 ugms_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame()):
    # RUSSIA
    aea_crs = ccrs.AlbersEqualArea(central_longitude=100,
                                   standard_parallels=(50, 70),
                                   central_latitude=56,
                                   false_easting=0,
                                   false_northing=0)
    # to proj4 form
    aea_crs_proj4 = aea_crs.proj4_init

    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={'projection': aea_crs})
    if not just_points:
        cmap = cm.get_cmap(cmap_name, 5)
        vmin, vmax = cmap_lims
        norm_cmap = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        cmap = cm.get_cmap(cmap_name, 18)
    # plot settings
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_extent(rus_extent)  # type: ignore
    if not ugms:
        # plot russia basemap
        basemap_data.to_crs(aea_crs_proj4).plot(  # type: ignore
            ax=ax, color='grey', edgecolor='black', legend=False, alpha=0.8)
    gdf_to_plot = gdf_to_plot.to_crs(aea_crs_proj4)  # type: ignore
    # plot variable
    if just_points:
        scatter_plot = gdf_to_plot.plot(
            ax=ax,
            column=distinction_col,
            cmap=cmap,
            marker='o', markersize=24,
            legend=True,
            legend_kwds={'ncol': 4,
                         "loc": "lower center",
                         "fmt": "{:.0f}", 'fontsize': 14,
                         'markerscale': 0.5, 'frameon': True})
    else:
        if ugms:
            ugms_gdf.to_crs(aea_crs_proj4).plot(
                ax=ax,
                column=metric_col,
                cmap=cmap, norm=norm_cmap,
                legend=False,
                edgecolor='black', linewidth=0.6,
                missing_kwds={'color': '#DF60DF00'})
        scatter_plot = gdf_to_plot.plot(
            ax=ax,
            column=metric_col,
            cmap=cmap, norm=norm_cmap,
            marker='o', markersize=24,
            edgecolor='black', linewidth=0.2,
            legend=True,
            legend_kwds={'orientation': 'horizontal',
                         'shrink': 0.35,
                         'pad': -0.075,
                         'anchor': (0.6, 0.5),
                         'drawedges': True})

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
            for qual, idx in gdf_to_plot.groupby(
                    f'{distinction_col}').groups.items():
                hist_df.loc[0, f'{qual}'] = len(idx)
        else:
            # list_of_limits_hist = [0, 250, 500, 1000, 2000]
            hist_df = pd.crosstab(gdf_to_plot[metric_col],
                                  pd.cut(gdf_to_plot[metric_col],
                                         list_of_limits,
                                         include_lowest=False))

            hist_df = hist_df.reset_index(drop=True)

        # x of borders, y of borders, weight, height
        ax_hist = ax.inset_axes([0.00, 0.05, 0.33, 0.24])

        extra_hist = hist_df.sum(axis=0).plot.bar(ax=ax_hist,
                                                  rot=0,
                                                  width=1,
                                                  grid=False,
                                                  facecolor='red',
                                                  edgecolor='black',
                                                  lw=1)
        if distinction_col:
            extra_hist.bar_label(extra_hist.containers[0], fmt='%.0f')
        else:
            extra_hist.bar_label(extra_hist.containers[0], fmt='%.0f',
                                 fontsize=8)
        extra_hist.set_facecolor('white')

        extra_hist.set_xlabel(f'{metric_col}',
                              fontdict={'fontsize': 8}, loc='right')
        if len(specific_xlabel) == 0:
            # xlbl = [str(col)[1:-1].replace(', ', '-')
            #         for col in hist_df.columns]
            xlbl = [str(col).replace(', ', '-')
                    for col in hist_df.columns]

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

        plt.setp(ax_hist.get_xticklabels(),
                 fontsize=8)
        plt.setp(ax_hist.get_yticklabels(),
                 fontsize=8)

    plt.title(f'{title_text}',
              fontdict={'size': 12})

    return ax


def russia_plots_n(gdf_to_plot: gpd.GeoDataFrame,
                   basemap_data: gpd.GeoDataFrame,
                   columns_from_gdf: list,
                   label_list: list,
                   nrows: int,
                   ncols: int,
                   title_text: list = [''],
                   rus_extent: list = [50, 140, 32, 90],
                   list_of_limits: list = [0.0, 0.5, 0.7, 0.8, 1.0],
                   cmap_lims: tuple = (0, 1),
                   cmap_name: str = 'RdYlGn',
                   figsize: tuple = (4.88189, 3.34646),
                   just_points: bool = False,
                   with_histogram: bool = False,
                   ugms: bool = False,
                   ugms_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame()):
    # RUSSIA
    aea_crs = ccrs.AlbersEqualArea(central_longitude=100,
                                   standard_parallels=(50, 70),
                                   central_latitude=56,
                                   false_easting=0,
                                   false_northing=0)
    # to proj4 form
    aea_crs_proj4 = aea_crs.proj4_init

    fig, axs = plt.subplots(figsize=figsize,
                            ncols=ncols,
                            nrows=nrows,
                            subplot_kw={'projection': aea_crs})

    for i, ax in enumerate(np.ravel(axs)):
        cmap = cm.get_cmap(cmap_name, 4)
        vmin, vmax = cmap_lims
        # [-100, -75, -50, -25, 0, 25, 50, 75, 100]
        # [0.0, 0.50, 0.70, 0.80, 1.00]
        # bounds = [-100, -75, -50, -25, 0, 25, 50, 75, 100]

        norm_cmap = mpl.colors.BoundaryNorm([0.0, 0.50, 0.70, 0.80, 1.00],
                                            4)
        # norm_cmap = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        if i >= len(columns_from_gdf):
            ax.set_visible(False)
            break
        # plot settings
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_extent(rus_extent)  # type: ignore
        if not ugms:
            # plot russia basemap
            basemap_data.to_crs(aea_crs_proj4).plot(  # type: ignore
                ax=ax, color='grey', edgecolor='black',
                legend=False, alpha=0.8, linewidth=0.6)
        gdf_to_plot = gdf_to_plot.to_crs(aea_crs_proj4)  # type: ignore
        # plot variable
        if just_points:
            scatter_plot = gdf_to_plot.plot(
                ax=ax,
                column=columns_from_gdf[i],
                cmap=cmap, norm=norm_cmap,
                marker='o', markersize=8,
                legend=True,
                legend_kwds={"loc": "lower center",
                             "ncol": 2,
                             "fmt": "{:.0f}", 'fontsize': 8})

        else:
            if ugms:
                ugms_gdf.to_crs(aea_crs_proj4).plot(
                    ax=ax,
                    column=columns_from_gdf[i],
                    cmap=cmap, norm=norm_cmap,
                    legend=False,
                    edgecolor='black', linewidth=0.6,
                    missing_kwds={'color': '#DF60DF00'})
            scatter_plot = gdf_to_plot.plot(
                ax=ax,
                column=columns_from_gdf[i],
                cmap=cmap, norm=norm_cmap,
                marker='o', markersize=14,
                edgecolor='black', linewidth=0.1,
                legend=True,
                legend_kwds={'orientation': 'horizontal',
                             'shrink': 0.3,
                             'pad': -0.05,
                             'anchor': (0.6, 0.5),
                             'drawedges': True})
        ax.text(0, 1, label_list[i], ha='left', va='top',
                transform=ax.transAxes, fontsize=14)
        my_fig = scatter_plot.figure
        if not just_points:
            cb_ax = my_fig.axes[nrows*ncols+i]
            cb_ax.tick_params(labelsize=10)
            # [-100, -75, -50, -25, 0,
            # 25, 50, 75, 100]
            # [0.00, 0.25, .50, .75, 1.00]
            # cb_ax.xaxis.set_major_locator(
            #     mticker.FixedLocator([i for i
            #                           in [0.00, 0.25, .50, .75, 1.00]]))
            # cb_ax.set_xticklabels([f'{i:.0f}'
            #                        for i in [0.00, 0.25, .50, .75, 1.00]])

        if with_histogram:
            if just_points:
                hist_df = pd.DataFrame()
                for qual, idx in gdf_to_plot.groupby(
                        f'{columns_from_gdf[i]}').groups.items():
                    hist_df.loc[0, f'{qual}'] = len(idx)
            else:
                hist_df = pd.crosstab(
                    gdf_to_plot[columns_from_gdf[i]],  # type: ignore
                    pd.cut(gdf_to_plot[columns_from_gdf[i]],  # type: ignore
                           list_of_limits,
                           include_lowest=False))

            hist_df = hist_df.reset_index(drop=True)
            # x of borders, y of borders, weight, height
            ax_hist = ax.inset_axes([0.00, 0.05, 0.33, 0.24])
            extra_hist = hist_df.sum(axis=0).plot.bar(ax=ax_hist,
                                                      rot=0,
                                                      width=1,
                                                      facecolor='red',
                                                      edgecolor='black',
                                                      lw=1)
            extra_hist.bar_label(extra_hist.containers[0], fmt='%.0f')
            extra_hist.set_facecolor('white')
            extra_hist.tick_params(width=1)
            # extra_hist.set_xlabel(f'{columns_from_gdf[i]}',
            #                       fontdict={'fontsize': 14}, loc='right')
            extra_hist.grid(False)

            if just_points:
                xlbl = [str(col).replace(', ', '-')
                        for col in hist_df.columns]
            else:
                xlbl = [str(col)[1:-1].replace(', ', '-')
                        for col in hist_df.columns]

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

            plt.setp(ax_hist.get_xticklabels(),
                     fontsize=8)
            plt.setp(ax_hist.get_yticklabels(),
                     fontsize=8)
        ax.set_title(f'{title_text[i]}',
                     fontdict={'size': 12})
        plt.tight_layout()

    return fig


def metric_viewer(gauges_file: gpd.GeoDataFrame,
                  metric_col: str,
                  metric_csv: str):
    model_metric = pd.read_csv(metric_csv)
    model_metric = model_metric.rename(columns={'basin': 'gauge_id',
                                                'gauge': 'gauge_id'})
    model_metric['gauge_id'] = model_metric['gauge_id'].astype('str')
    model_metric = model_metric.set_index('gauge_id')
    if 'gauge_id' not in gauges_file.columns:
        res_file = gauges_file.join(model_metric).dropna()
    else:
        res_file = gauges_file.set_index('gauge_id').join(model_metric).dropna()
    nse_median = res_file[metric_col].median()
    # res_file.loc[res_file[metric_col] < 0, metric_col] = 0

    return res_file, nse_median
