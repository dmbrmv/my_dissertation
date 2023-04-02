from scripts.grid_calculator import Gridder
from scripts.loaders import multi_var_nc, aggregation_definer
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm

russia_ws = gpd.read_file('../geo_data/great_db/geometry/russia_ws.gpkg')

meteo_path = '../geo_data/meteorology'
era5_land = Path(f'{meteo_path}/era5-land/russia')
era5 = Path(f'{meteo_path}/era5/russia')
imerg = Path(f'{meteo_path}/imerg_year_new')
gpcp = Path(f'{meteo_path}/gpcp_year_new')
gleam = Path(f'{meteo_path}/gleam_vars')
mswep = Path(f'{meteo_path}/mswep')
icon = '../geo_data/icon_data'

place_to_save = '../geo_data/meteo_grids'

ds_description = {
    # 'era5_land': {'res': 0.05,
    #               'f_path': multi_var_nc(era5_land,
    #                                      file_extension='nc')},
    # 'era5': {'res': 0.125,
    #          'f_path': multi_var_nc(era5,
    #                                 file_extension='nc')},
    # 'imerg': {'res': 0.05,
    #           'f_path': multi_var_nc(imerg,
    #                                  file_extension='nc')},
    # 'gpcp': {'res': 0.25,
    #          'f_path': multi_var_nc(gpcp,
    #                                 file_extension='nc')},
    # 'gleam': {'res': 0.125,
    #           'f_path': multi_var_nc(gleam,
    #                                  file_extension='nc')},
    'mswep': {'res': 0.05,
              'f_path': multi_var_nc(mswep,
                                     file_extension='nc')}}

# ds_description = {
#     'icon': {'res': 0.0625,
#              'f_path': multi_var_nc(Path(icon),
#                                     file_extension='nc')}}


for dataset, settings in ds_description.items():

    grid_res = settings['res']

    for i, gauge_id in enumerate(tqdm(russia_ws['gauge_id'])):
        print(f"\nNow it's {gauge_id}\n")
        ws_geometry = russia_ws.loc[i, 'geometry']

        for variable, pathes in settings['f_path'].items():
            print(f'Calculation for {variable} of {dataset}')
            aggregation = aggregation_definer(dataset, variable)
            if dataset == 'imerg':
                meteo_grid = Gridder(half_grid_resolution=grid_res,
                                     ws_geom=ws_geometry,
                                     gauge_id=gauge_id,
                                     path_to_save=Path(f'{place_to_save}'),
                                     nc_pathes=pathes,
                                     dataset=dataset,
                                     var=variable,
                                     aggregation_type=aggregation,
                                     force_weights=True,
                                     weight_mark='imerg')
                meteo_grid.grid_value_ws()
            elif dataset == 'gpcp':
                meteo_grid = Gridder(half_grid_resolution=grid_res,
                                     ws_geom=ws_geometry,
                                     gauge_id=gauge_id,
                                     path_to_save=Path(f'{place_to_save}'),
                                     nc_pathes=pathes,
                                     dataset=dataset,
                                     var=variable,
                                     aggregation_type=aggregation,
                                     force_weights=True,
                                     weight_mark='gpcp')
                meteo_grid.grid_value_ws()
            elif dataset == 'mswep':
                meteo_grid = Gridder(half_grid_resolution=grid_res,
                                     ws_geom=ws_geometry,
                                     gauge_id=gauge_id,
                                     path_to_save=Path(f'{place_to_save}'),
                                     nc_pathes=pathes,
                                     dataset=dataset,
                                     var=variable,
                                     aggregation_type=aggregation,
                                     force_weights=True,
                                     weight_mark='imerg')
                meteo_grid.grid_value_ws()
            elif dataset == 'gleam':
                meteo_grid = Gridder(half_grid_resolution=grid_res,
                                     ws_geom=ws_geometry,
                                     gauge_id=gauge_id,
                                     path_to_save=Path(f'{place_to_save}'),
                                     nc_pathes=pathes,
                                     dataset=dataset,
                                     var=variable,
                                     aggregation_type=aggregation,
                                     force_weights=True,
                                     weight_mark='gleam')
                meteo_grid.grid_value_ws()
            elif dataset == 'icon':
                for path in pathes:
                    meteo_grid = Gridder(half_grid_resolution=grid_res,
                                         ws_geom=ws_geometry,
                                         gauge_id=gauge_id,
                                         path_to_save=Path(f'{place_to_save}'),
                                         nc_pathes=path,
                                         dataset=dataset,
                                         var=variable,
                                         aggregation_type=aggregation,
                                         force_weights=True,
                                         weight_mark='icon')
                    meteo_grid.grid_value_ws()
            else:
                meteo_grid = Gridder(half_grid_resolution=grid_res,
                                     ws_geom=ws_geometry,
                                     gauge_id=gauge_id,
                                     path_to_save=Path(f'{place_to_save}'),
                                     nc_pathes=pathes,
                                     dataset=dataset,
                                     var=variable,
                                     aggregation_type=aggregation)
                meteo_grid.grid_value_ws()
