import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import pandas as pd
plt.rcParams["font.family"] = "Times New Roman"


def russia_plots(gdf_to_plot: gpd.GeoDataFrame,
                 basemap_data: gpd.GeoDataFrame,
                 distinction_col: str,
                 title_text: str = '',
                 rus_extent: list = [50, 140, 32, 90],
                 list_of_limits: list = [0.0, 0.3, 0.5, 0.7, 1.0],
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
    cmap = cm.get_cmap(cmap_name, 5)
    norm_cmap = mpl.colors.Normalize(vmin=0, vmax=1)

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
            marker='o', markersize=1,
            legend=True,
            legend_kwds={"loc": "lower left",
                         "fmt": "{:.0f}", 'fontsize': 6})

    else:
        if ugms:
            ugms_gdf.to_crs(aea_crs_proj4).plot(
                ax=ax,
                column=metric_col,
                cmap=cmap, norm=norm_cmap,
                legend=False,
                edgecolor='black', linewidth=0.6)
        scatter_plot = gdf_to_plot.plot(
            ax=ax,
            column=metric_col,
            cmap=cmap, norm=norm_cmap,
            marker='o', markersize=3,
            edgecolor='black', linewidth=0.2,
            legend=True,
            legend_kwds={'orientation': 'horizontal',
                         'shrink': 0.4,
                         'pad': -0.075,
                         'anchor': (0.6, 0.5),
                         #  'values': list_of_limits,
                         #  'ticks': [i for i in list_of_limits],
                         'drawedges': True})

    my_fig = scatter_plot.figure
    if not just_points:
        cb_ax = my_fig.axes[1]
        cb_ax.tick_params(labelsize=6)
        # cb_ax.set_xticklabels([f'{i}-{round(i+0.2, 2)}'
        #                        for i in list_of_limits])
    if with_histogram:
        # list_of_limits_hist = [0, 250, 500, 1000, 2000]
        hist_df = pd.crosstab(gdf_to_plot[metric_col],  # type: ignore
                              pd.cut(gdf_to_plot[metric_col],  # type: ignore
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
        extra_hist.set_facecolor('white')
        extra_hist.tick_params(width=1)
        extra_hist.set_xlabel(f'{metric_col}',
                              fontdict={'fontsize': 8}, loc='right')

        xlbl = [str(col)[1:-1].replace(', ', '-')
                for col in hist_df.columns]

        extra_hist.set_xticklabels(xlbl)

        plt.setp(ax_hist.get_xticklabels(),
                 fontsize=6)
        plt.setp(ax_hist.get_yticklabels(),
                 fontsize=6)

    plt.title(f'{title_text}',
              fontdict={'size': 9})

    return fig


def metric_viewer(gauges_file: gpd.GeoDataFrame,
                  metric_col: str,
                  metric_csv: str):
    model_metric = pd.read_csv(metric_csv)
    model_metric = model_metric.rename(columns={'basin': 'gauge_id'})
    model_metric['gauge_id'] = model_metric['gauge_id'].astype('str')
    model_metric = model_metric.set_index('gauge_id')

    res_file = gauges_file.set_index('gauge_id').join(model_metric).dropna()
    nse_median = res_file[metric_col].median()
    res_file.loc[res_file[metric_col] < 0, metric_col] = 0

    return res_file, nse_median
