from scripts.grid_calculator import Gridder
from scripts.loaders import multi_var_nc, aggregation_definer
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm

russia_ws = gpd.read_file('./data/russia_ws.gpkg')

meteo_path = '/home/anton/dima_experiments/geo_data/meteorology'
era5_land = Path(f'{meteo_path}/era5-land/russia')
era5 = Path(f'{meteo_path}/era5/russia')
imerg = Path(f'{meteo_path}/imerg_year')
gpcp = Path(f'{meteo_path}/gpcp_year')
gleam = Path(f'{meteo_path}/gleam_vars')
mswep = Path(f'{meteo_path}/mswep')

place_to_save = '/home/anton/dima_experiments/geo_data/meteo_grids'

ds_description = {
    'era5_land': {'res': 0.05,
                  'f_path': multi_var_nc(era5_land)},
    'era5': {'res': 0.125,
             'f_path': multi_var_nc(era5)},
    'imerg': {'res': 0.05,
              'f_path': multi_var_nc(imerg)},
    'gpcp': {'res': 0.25,
             'f_path': multi_var_nc(gpcp)},
    'gleam': {'res': 0.125,
              'f_path': multi_var_nc(gleam)},
    'mswep': {'res': 0.05,
              'f_path': multi_var_nc(mswep)}}


for dataset, settings in ds_description.items():

    grid_res = settings['res']

    for i, gauge_id in enumerate(tqdm(russia_ws['gauge_id'])):

        ws_geometry = russia_ws.loc[i, 'geometry']

        for variable, pathes in settings['f_path'].items():

            aggregation = aggregation_definer(dataset, variable)

            meteo_grid = Gridder(half_grid_resolution=grid_res,
                                 ws_geom=ws_geometry,
                                 gauge_id=gauge_id,
                                 path_to_save=Path(f'{place_to_save}'),
                                 nc_pathes=pathes,
                                 dataset=dataset,
                                 var=variable,
                                 aggregation_type=aggregation)

            meteo_grid.grid_value_ws()