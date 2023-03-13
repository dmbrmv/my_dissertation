import geopandas as gpd
import numpy as np
import pandas as pd
import fiona
from pathlib import Path
from shapely.geometry import Polygon
from .geometry_modifications import (select_big_from_mp, get_gdf_poly,
                                     find_poly_area)
from .hydro_atlas_variables import (hydrology_variables,
                                    physiography_variables,
                                    climate_variables, landcover_variables,
                                    soil_and_geo_variables, urban_variables,
                                    monthes)


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
    gdf_your_WS = select_big_from_mp(user_ws)

    # transform to geopandas for geometry operations
    gdf_your_WS = gpd.GeoSeries([gdf_your_WS])
    gdf_your_WS = gpd.GeoDataFrame({'geometry': gdf_your_WS})
    gdf_your_WS = gdf_your_WS.set_crs('EPSG:4326')

    # connect to HydroATLAS file with fiona+gpd interface
    layer_small = fiona.listlayers(gdb_file_path)[-1]
    # Read choosen geodatabase layer with geopandas
    gdf = gpd.read_file(gdb_file_path,
                        mask=user_ws,
                        layer=layer_small,
                        ignore_geometry=False)
    # create new column where each intersection is geodataframe
    gdf['gdf_geometry'] = tuple(map(get_gdf_poly, gdf['geometry']))
    # calculate weight of each intersection correspond to it native size
    gdf['weights'] = gdf['gdf_geometry'].apply(
        lambda x: gpd.overlay(gdf_your_WS, x)).apply(find_poly_area) /\
        gdf['gdf_geometry'].apply(find_poly_area)
    # calculate area with weight appliance
    gdf['weight_area'] = tuple(map(find_poly_area,
                                   gdf['gdf_geometry'])) * gdf['weights']

    # calculate each variable weighted mean
    geo_vector = gdf[hydrology_variables +
                     physiography_variables +
                     climate_variables +
                     landcover_variables +
                     soil_and_geo_variables +
                     urban_variables].apply(
        lambda x:
            np.sum(x * gdf['weights'])/np.sum(gdf['weights']))
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
    geo_vector['ws_area'] = find_poly_area(gdf_your_WS)

    return geo_vector


def save_results(extracted_data: list,
                 gauge_ids: pd.Series,
                 path_to_save: str):

    Path(path_to_save).mkdir(exist_ok=True, parents=True)

    # create DataFrame to save it by categories
    df_to_disk = pd.concat(extracted_data).set_index(gauge_ids)

    df_to_disk.to_csv(f'{path_to_save}/geo_vector.csv')

    save_names = {'hydro': hydrology_variables,
                  'physio': physiography_variables+['ws_area'],
                  'climate': climate_variables,
                  'landcover': landcover_variables,
                  'soil_geo': soil_and_geo_variables,
                  'urban': urban_variables}

    for key, values in save_names.items():
        df_to_disk[values].to_csv(f'{path_to_save}/{key}.csv')

    return
