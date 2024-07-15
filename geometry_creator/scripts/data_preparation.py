import numpy as np
from itertools import product
from copy import deepcopy
import glob
import geopandas as gpd
from typing import Union
from .geom_functions import round_down, round_up, min_max_xy
from pathlib import Path
import pandas as pd


def aoi_name_path(shp_generator):

    name_n_path = {str(path).split('/')[-1][:-4]: path
                   for path in shp_generator}

    return name_n_path


def tile_selector(topo_p, gauge_gdf):
    x_min, y_min, x_max, y_max = gauge_gdf[['x', 'y']].describe().loc[
        ['min', 'max'], ['x', 'y']].values.ravel()

    x_min, y_min = list(map(round_down, [x_min, y_min]))
    x_max, y_max = list(map(round_up, [x_max, y_max]))

    tile_boundaries = [f'n{lat}e0{lon}' if len(str(lon)) < 3
                       else f'n{lat}e{lon}'
                       for lat, lon in
                       product(np.arange(
                           start=y_min, stop=y_max, step=5),
        np.arange(
                           start=x_min, stop=x_max, step=5))]
    variables = ['dir', 'elv']

    tiles = {var: [item for sublist in
                   [glob.glob(f'{topo_p}/{var}/**/{boundary}_{var}.tif',
                              recursive=True)
                    for boundary in tile_boundaries]
                   for item in sublist]
             for var in variables}

    return tiles


def aoi_tiles(topo_p: Union[Path, str], aoi_shp: str) -> dict:

    test_mask = gpd.read_file(aoi_shp).loc[0, 'geometry']
    x_min, y_min, x_max, y_max = min_max_xy(test_mask)
    x_min, y_min = list(map(round_down, [x_min, y_min]))
    x_max, y_max = list(map(round_up, [x_max, y_max]))

    if (x_min < 0) & (x_max < 0):
        x_max, x_min = np.abs(x_min), np.abs(x_max)

        tile_boundaries = [f'n{lat}w0{lon}' if len(str(lon)) < 3
                           else f'n{lat}w{lon}'
                           for lat, lon in
                           product(
            np.arange(start=y_min, stop=y_max, step=5),
            np.arange(start=x_min, stop=x_max, step=5))]
    else:
        tile_boundaries = [f'n{lat}e0{lon}' if len(str(lon)) < 3
                           else f'n{lat}e{lon}'
                           for lat, lon in
                           product(
            np.arange(start=y_min, stop=y_max, step=5),
            np.arange(start=x_min, stop=x_max, step=5))]

    variables = ['dir', 'elv']

    tiles = {var: [item for sublist in
                   [glob.glob(f'{topo_p}/{var}/**/{boundary}_{var}.tif',
                              recursive=True)
                    for boundary in tile_boundaries]
                   for item in sublist]
             for var in variables}

    return tiles


def read_gauge(gdf):
    gdf['point_geom'] = deepcopy(gdf['geometry'])

    gdf['geometry'] = None
    gdf['x'] = [point_info.x for point_info in gdf.loc[:, 'point_geom']]
    gdf['y'] = [point_info.y for point_info in gdf.loc[:, 'point_geom']]
    gdf['point_geom'] = gdf['point_geom'].astype('str')

    return gdf


def tile_determinator(gauge_file,
                      path_to_save):

    all_tiles = glob.glob('./data/region_masks/*.gpkg')

    tile_dict = {}
    tile_ws_area = {}

    if 'area' not in gauge_file.columns:
        gauge_file['area'] = np.NaN

    gauge_file['area'] = pd.to_numeric(gauge_file['area'], errors='coerce')
    if 'name_ru' in gauge_file.columns:
        gauge_file = gauge_file.rename(columns={'name_ru': 'name'})

    for row, name in enumerate(gauge_file['name']):
        tile_dict[name] = {}
        tile_ws_area[name] = {}
        for tile in all_tiles:
            aoi = gpd.read_file(tile,
                                encoding='utf-8').loc[0, 'geometry']
            if aoi.contains(gauge_file.loc[row, 'geometry']):
                tile_dict[name][tile] = aoi.area
                tile_ws_area[name][tile] = gauge_file.loc[row, 'area']

    fin_tiles = {}
    for name, tile_candidates in tile_dict.items():

        if len(tile_candidates.values()) > 1:
            if max(tile_ws_area[name].values()) > 24000:
                fin_tiles[name] = max(tile_candidates, key=tile_candidates.get)
            else:
                fin_tiles[name] = min(tile_candidates, key=tile_candidates.get)

        elif len(tile_candidates.values()) == 0:
            fin_tiles[name] = np.NaN
        else:
            fin_tiles[name] = list(tile_candidates.keys())[0]

    for row, name in enumerate(gauge_file['name']):
        gauge_file.loc[row, 'target_tile'] = fin_tiles[name]

    for tile, ids in gauge_file.groupby('target_tile').groups.items():
        fname = tile.split('/')[-1][:-9]
        gauge_file.loc[ids, :].to_file(f'{path_to_save}/{fname}.gpkg',
                                       encoding='utf-8')

    return gauge_file
