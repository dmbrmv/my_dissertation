import sys
sys.path.append('/workspaces/my_dissertation')
from shapely.geometry import Polygon
from pathlib import Path
import fiona
import pandas as pd
import numpy as np
import geopandas as gpd
from .hydro_atlas_variables import (hydrology_variables,
                                    physiography_variables,
                                    climate_variables, landcover_variables,
                                    soil_and_geo_variables, urban_variables,
                                    monthes)
from meteo_grids_parser.scripts.geom_proc import (poly_from_multipoly,
                                                  area_from_gdf)


def featureXtractor(user_ws: Polygon, gdb_file_path: str):
    """
    Fucntion calculates weighted mean of variables which are occured
    in intersection of user_ws and subbasins from HydroATLAS database

    Args:
        user_ws (Polygon): User's watershed boundary
        gdb_file_path (str): path to gdb of HydroATLAS on disk

    Returns:
        geo_vector (Series): Series of variables corresponded to user_ws id
    """

    # get only biggest polygon areas from watershed
    gdf_your_ws = poly_from_multipoly(user_ws)
    # transform to geopandas for geometry operations
    gdf_your_ws = gpd.GeoSeries([gdf_your_ws])
    gdf_your_ws = gpd.GeoDataFrame({'geometry': gdf_your_ws})
    gdf_your_ws = gdf_your_ws.set_crs('EPSG:4326')
    # connect to HydroATLAS file with fiona+gpd interface
    layer_small = fiona.listlayers(gdb_file_path)[-1]
    # Read choosen geodatabase layer with geopandas
    gdf = gpd.read_file(gdb_file_path,
                        mask=user_ws,
                        layer=layer_small,
                        ignore_geometry=False)
    # keep only single geometries from HydroATLAS
    gdf['gdf_geometry'] = tuple(map(poly_from_multipoly, gdf['geometry']))
    # transform each polygon to geodataframe instance
    gdf['gdf_geometry'] = [gpd.GeoDataFrame(
        {'geometry': sample_geom},
        index=[0]).set_crs('EPSG:4326')
        for sample_geom in gdf['gdf_geometry']]
    # calculate weight of each intersection correspond to it native size
    gdf['weights'] = gdf['gdf_geometry'].apply(
        lambda x: gpd.overlay(gdf_your_ws, x)).apply(area_from_gdf) /\
        gdf['gdf_geometry'].apply(area_from_gdf)
    # calculate area with weight appliance
    gdf['weight_area'] = tuple(map(area_from_gdf,
                                   gdf['gdf_geometry'])) * gdf['weights']
    # calculate each variable weighted mean
    geo_vector = gdf[hydrology_variables +
                     physiography_variables +
                     climate_variables +
                     landcover_variables +
                     soil_and_geo_variables +
                     urban_variables].applymap(
        lambda x:
            np.sum(x * gdf['weights'])/np.sum(gdf['weights']))
    geo_vector[geo_vector < 0] = np.NaN
    geo_vector = geo_vector.mean()
    # some values in HydroATLAS was multiplied by <X>, so to bring it
    # back to original form this procedure is required
    divide_by_10 = [item for sublist
                    in [['lka_pc_use'],
                        ['dor_pc_pva'],
                        ['slp_dg_sav'],
                        ['tmp_dc_s{}'.format(i) for i in monthes],
                        ['hft_ix_s93'],
                        ['hft_ix_s09']]
                    for item in sublist]
    divide_by_100 = [item for sublist
                     in [['ari_ix_sav'],
                         ['cmi_ix_s{}'.format(i) for i in monthes]]
                     for item in sublist]

    geo_vector[divide_by_10] /= 10
    geo_vector[divide_by_100] /= 100
    # store basin area
    geo_vector['ws_area'] = area_from_gdf(gdf_your_ws)

    return geo_vector


def save_results(extracted_data: list,
                 gauge_ids: pd.Series,
                 path_to_save: str):

    Path(path_to_save).mkdir(exist_ok=True, parents=True)

    # create DataFrame to save it by categories
    df_to_disk = pd.concat(extracted_data, axis=1).T.set_index(gauge_ids)

    df_to_disk.to_csv(f'{path_to_save}/geo_vector.csv')

    save_names = {'hydro': hydrology_variables,
                  'physio': physiography_variables+['ws_area'],
                  'climate': climate_variables,
                  'landcover': landcover_variables,
                  'soil_geo': soil_and_geo_variables,
                  'urban': urban_variables}

    for key, values in save_names.items():
        df_to_disk[values].to_csv(f'{path_to_save}/{key}.csv')
