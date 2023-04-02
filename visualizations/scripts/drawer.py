import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd


def russia_plots(gdf_to_plot: gpd.GeoDataFrame,
                 basemap_data: gpd.GeoDataFrame,
                 distinction_col: str,
                 title_text: str,
                 rus_extent: list = [50, 140, 32, 90],
                 list_of_limits: list = [0.00, 0.15, 0.30, 0.50, 0.70, 0.90],
                 cmap: str = 'RdYlGn',
                 metric_col: str = '',
                 figsize: tuple = (11.69, 8.27),
                 just_points: bool = False,
                 with_histogram: bool = False):
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

    # plot settings
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_extent(rus_extent)  # type: ignore

    # plot russia basemap
    basemap_data.to_crs(aea_crs_proj4).plot(  # type: ignore
        ax=ax, color='grey', edgecolor='black', legend=False)
    gdf_to_plot = gdf_to_plot.to_crs(aea_crs_proj4)  # type: ignore
    # plot variable
    if just_points:
        scatter_plot = gdf_to_plot.plot(
            ax=ax,
            column=distinction_col,
            cmap=cmap,
            marker='o', markersize=10,
            legend=True,
            legend_kwds={'loc': "lower left",
                         'fontsize': 8})
    else:
        scatter_plot = gdf_to_plot.plot(
            ax=ax,
            column=metric_col,
            cmap=cmap,
            marker='o', markersize=20,
            legend=True,
            legend_kwds={'label': metric_col,
                         'orientation': 'horizontal',
                         'shrink': 0.3,
                         'pad': -0.1,
                         'values': list_of_limits,
                         'ticks': [0.00, 0.20, 0.40, 0.55, 0.75, 0.90],
                         'drawedges': True})
    if with_histogram:
        # list_of_limits_hist = [0, 250, 500, 1000, 2000]
        hist_df = pd.crosstab(gdf_to_plot[metric_col],  # type: ignore
                              pd.cut(gdf_to_plot[metric_col],  # type: ignore
                                     list_of_limits,
                                     include_lowest=False))

        hist_df = hist_df.reset_index(drop=True)

        # x of borders, y of borders, weight, height
        ax_hist = ax.inset_axes([0.06, 0.07, 0.24, 0.24])

        extra_hist = hist_df.sum(axis=0).plot.bar(ax=ax_hist,
                                                  rot=0,
                                                  width=1,
                                                  facecolor='red',
                                                  edgecolor='black',
                                                  lw=1)
        extra_hist.set_facecolor('white')
        extra_hist.tick_params(width=2)

        xlbl = [str(col)[1:-1].replace(', ', '-')
                for col in hist_df.columns]

        extra_hist.set_xticklabels(xlbl)

        plt.setp(ax_hist.get_xticklabels(),
                 fontsize=9,
                 fontweight="bold")
        plt.setp(ax_hist.get_yticklabels(),
                 fontsize=9,
                 fontweight="bold")

    plt.title(f'{title_text}',
              fontdict={'size': 14})

    return fig
