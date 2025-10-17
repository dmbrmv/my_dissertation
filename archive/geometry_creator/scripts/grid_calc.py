from pathlib import Path
from copy import deepcopy
from shapely import geometry, ops
from pysheds.grid import Grid  # type: ignore
import geopandas as gpd
import gc
from tqdm import tqdm
import numpy as np
import pandas as pd

from .geom_functions import (polygon_area, find_float_len, select_big_from_MP)
from ...data_builders.geometry_scripts.gdal_processing import create_mosaic, vrt_to_geotiff


def from_dir_to_ws(region_name,
                   gauge_file,
                   result_folder,
                   tiles,
                   save_p):

    result_folder = Path(f'{result_folder}/fdir/{region_name}')
    result_folder.mkdir(exist_ok=True, parents=True)

    mosaic = create_mosaic(file_path=result_folder,
                           file_name=f'{region_name}',
                           tiles=tiles)

    dir_tiff = vrt_to_geotiff(
        vrt_path=mosaic,
        geotiff_path=f'{result_folder}/{region_name}_dir.tif')

    initial_grid = Grid.from_raster(dir_tiff, data_name='fdir_grid',
                                    nodata=0)
    fdir = initial_grid.read_raster(dir_tiff, data_name='fdir',
                                    nodata=0)
    del dir_tiff
    del mosaic
    gc.collect()

    for i in range(len(gauge_file)):
        name = gauge_file.loc[i, 'name']
        if gauge_file.loc[i, 'area'] is not None:
            area = float(gauge_file.loc[i, 'area'])
            print(f'\nCalculation for {name} -- Area, sq.km = {area:.2f}')
        else:
            area = None
            print(f'Calculation for {name} -- Area, sq.km = {area}')
        x_coord = gauge_file.loc[i, 'x']
        y_coord = gauge_file.loc[i, 'y']

        if find_float_len(x_coord) or find_float_len(y_coord):
            gauge_file.loc[i, 'geometry'] = catchment_from_dir(
                grid_copy=deepcopy(initial_grid),
                fdir_copy=fdir,
                x=x_coord,
                y=y_coord)
            new_area = polygon_area(
                lats=gauge_file.loc[i, 'geometry'].exterior.coords.xy[1],
                lons=gauge_file.loc[i, 'geometry'].exterior.coords.xy[0])
            gauge_file.loc[i, 'new_area'] = new_area
            print(f'Calculated area, sq.km {new_area:.2f}')
        else:
            print(
                f'\nOoops ! {x_coord}, {y_coord} -- Define gage better !')
            gauge_file.loc[i, 'geometry'] = None

    del initial_grid
    del fdir
    gc.collect()
    gauge_file = gpd.GeoDataFrame(gauge_file).set_crs(epsg=4326)
    gauge_file['db_dif'] = (
        gauge_file['area'] - gauge_file['new_area']) \
        / (gauge_file['area']) * 100

    gauge_file.to_file(f'{save_p}/{region_name}.gpkg',
                       encoding='utf-8')
    return gauge_file


def dir_acc_for_aoi(region_name,
                    result_folder,
                    tiles):

    fdir_folder = Path(f'{result_folder}/fdir/')
    fdir_folder.mkdir(exist_ok=True, parents=True)

    mosaic = create_mosaic(file_path=result_folder,
                           file_name=f'{region_name}',
                           tiles=tiles)

    dir_tiff = vrt_to_geotiff(
        vrt_path=mosaic,
        geotiff_path=f'{fdir_folder}/{region_name}_dir.tif')

    initial_grid = Grid.from_raster(dir_tiff, data_name='fdir_grid',
                                    nodata=0)
    fdir = initial_grid.read_raster(dir_tiff, data_name='fdir',
                                    nodata=0)
    del dir_tiff
    del mosaic
    gc.collect()

    # Determine D8 flow directions from DEM
    # ----------------------
    # Specify directional mapping
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    # Calculate flow accumulation
    # --------------------------
    acc = initial_grid.accumulation(fdir, dirmap=dirmap)

    acc_folder = Path(f'{result_folder}/acc/')
    acc_folder.mkdir(exist_ok=True, parents=True)

    initial_grid.to_raster(file_name=f'{acc_folder}/{region_name}_acc.tiff',
                           data=acc, data_name='acc')
    del fdir
    del acc
    gc.collect()

    print(f'\n{region_name} полностью обсчитан !\n')


# def catchment_from_elv(grid_copy,
#                        dem_copy,
#                        x,
#                        y):
#     ws = None
#     initial_grid = grid_copy
#     # Resolve flats in DEM
#     inflated_dem = dem_copy

#     # Determine D8 flow directions from DEM
#     # ----------------------
#     # Specify directional mapping
#     dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
#     # Compute flow directions
#     # -------------------------------------
#     fdir = initial_grid.flowdir(inflated_dem, dirmap=dirmap)

#     # Calculate flow accumulation
#     # --------------------------
#     acc = initial_grid.accumulation(fdir, dirmap=dirmap)

#     if acc is not None:
#         # Snap pour point to high accumulation cell
#         x_snap, y_snap = initial_grid.snap_to_mask(acc > 1e3, (x, y))

#         # Delineate the catchment
#         catch = initial_grid.catchment(x=x_snap, y=y_snap,
#                                        fdir=fdir,
#                                        dirmap=dirmap,
#                                        xytype='coordinate')

#         # Crop and plot the catchment
#         # ---------------------------
#         # Clip the bounding box to the catchment
#         initial_grid.clip_to(catch)
#         # clipped_catch = initial_grid.view(ws_grid)
#         if initial_grid is not None:
#             ws = initial_grid.polygonize()
#             ws = ops.unary_union([geometry.shape(shape)
#                                   for shape, value in ws])
#             ws = select_big_from_MP(ws)

#         return ws
#     else:
#         return None


def catchment_from_dir(grid_copy,
                       fdir_copy,
                       x,
                       y):
    ws = None
    initial_grid = None
    initial_grid = grid_copy

    # Determine D8 flow directions from DEM
    # ----------------------
    # Specify directional mapping
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    # Calculate flow accumulation
    # --------------------------
    acc = initial_grid.accumulation(fdir_copy, dirmap=dirmap)

    if acc is not None:
        # Snap pour point to high accumulation cell
        x_snap, y_snap = initial_grid.snap_to_mask(acc > 1e3, (x, y))
        del acc

        # Delineate the catchment
        catch = initial_grid.catchment(x=x_snap, y=y_snap,
                                       fdir=fdir_copy,
                                       dirmap=dirmap,
                                       xytype='coordinate')

        # Crop and plot the catchment
        # ---------------------------
        # Clip the bounding box to the catchment
        initial_grid.clip_to(catch)
        # clipped_catch = initial_grid.view(ws_grid)
        if initial_grid is not None:
            ws = initial_grid.polygonize()
            ws = ops.unary_union([geometry.shape(shape)
                                  for shape, _ in ws])
            ws = select_big_from_MP(ws)
        gc.collect()
        return ws
    else:
        return None


def my_catchment(grid_p: str,
                 fdir_p: str,
                 facc_p: str,
                 acc_faktor: float,
                 gauges_file,
                 save_p: str,
                 region_name: str):
    """_summary_

    Args:
        grid_p (str): _description_
        fdir_p (str): _description_
        facc_p (str): _description_
        gauges_file (_type_): _description_
        save_p (str): _description_
        region_name (str): _description_
    """
    ws = None

    grid = Grid.from_raster(grid_p, data_name='fdir_grid',
                            nodata=0)
    fdir = grid.read_raster(fdir_p, data_name='fdir',
                            nodata=0)
    acc = grid.read_raster(facc_p, data_name='acc',
                           nodata=0)
    gauges_file['area'] = pd.to_numeric(gauges_file['area'],
                                        errors='coerce')
    for i in tqdm(range(len(gauges_file))):
        grid = Grid.from_raster(grid_p, data_name='fdir_grid',
                                nodata=0)

        name = gauges_file.loc[i, 'name']
        area = gauges_file.loc[i, 'area']

        if np.isnan(area) | isinstance(area, str):
            print(f'\nCalculation for {name} -- Area, sq.km = {area}')
        else:
            print(f'\nCalculation for {name} -- Area, sq.km = {area:.2f}')

        x_coord = gauges_file.loc[i, 'x']
        y_coord = gauges_file.loc[i, 'y']
        # Determine D8 flow directions from DEM
        # ----------------------
        # Specify directional mapping
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        # Snap pour point to high accumulation cell
        x_snap, y_snap = grid.snap_to_mask(
            acc > acc_faktor, (x_coord, y_coord))          # type: ignore

        # Delineate the catchment
        catch = grid.catchment(x=x_snap, y=y_snap,
                               fdir=fdir,  # type: ignore
                               dirmap=dirmap,
                               xytype='coordinate')
        # Crop and plot the catchment
        # ---------------------------
        # Clip the bounding box to the catchment
        grid.clip_to(catch)
        # clipped_catch = initial_grid.view(ws_grid)

        ws = grid.polygonize()
        ws = ops.unary_union([geometry.shape(shape)
                              for shape, _ in ws])
        ws = select_big_from_MP(ws)
        gauges_file.loc[i, 'geometry'] = ws

        new_area = polygon_area(
            lats=gauges_file.loc[i, 'geometry'].exterior.coords.xy[1],
            lons=gauges_file.loc[i, 'geometry'].exterior.coords.xy[0])
        gauges_file.loc[i, 'new_area'] = new_area

        print(f'Calculated area, sq.km {new_area:.2f}')
        del grid
        gc.collect()

    gauges_file = gpd.GeoDataFrame(
        gauges_file).set_crs(epsg=4326)  # type: ignore

    if 'area_db' in gauges_file.columns:
        gauges_file['db_dif'] = (
            gauges_file['area_db'] - gauges_file['new_area']) \
            / (gauges_file['area_db']) * 100

        gauges_file['db_checked'] = np.where(
            np.abs(gauges_file['db_dif']) < 15, 1, 0)

    if 'ais_area' in gauges_file.columns:
        gauges_file['ais_dif'] = (
            gauges_file['ais_area'] - gauges_file['new_area']) \
            / (gauges_file['ais_area']) * 100
        gauges_file['ais_checked'] = np.where(
            np.abs(gauges_file['ais_dif']) < 15, 1, 0)

        for i in range(len(gauges_file)):
            if (gauges_file.loc[i, 'ais_checked'] == 0) \
                    & (gauges_file.loc[i, 'db_checked'] == 0):
                gauges_file.loc[i, 'checked'] = 0
            elif (gauges_file.loc[i, 'ais_checked'] == 1) \
                    & (gauges_file.loc[i, 'db_checked'] == 0):
                gauges_file.loc[i, 'checked'] = 1
            else:
                gauges_file.loc[i, 'checked'] = 1

    storage = Path(save_p)
    storage.mkdir(exist_ok=True, parents=True)
    gauges_file.to_file(f'{storage}/{region_name}.gpkg',
                        encoding='utf-8')
    return gauges_file
