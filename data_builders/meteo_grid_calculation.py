from scripts.grid_calculator import Gridder
from scripts.loaders import aggregation_definer
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm


class MeteoGrids():

    def __init__(self,
                 cfg,
                 ds_description: dict) -> None:

        ws_geom = gpd.read_file(
            f'{cfg.watershed_storage}/{cfg.watershed_name}.gpkg')
        place_to_save = f'{cfg.grid_storage}'

        for dataset, settings in ds_description.items():

            grid_res = settings['res']

            for i, gauge_id in enumerate(tqdm(ws_geom['gauge_id'])):
                print(f"\nNow it's {gauge_id}\n")
                ws_geometry = ws_geom.loc[i, 'geometry']

                for variable, pathes in settings['f_path'].items():
                    print(f'Calculation for {variable} of {dataset}')
                    aggregation = aggregation_definer(dataset, variable)
                    if dataset == 'imerg':
                        meteo_grid = Gridder(half_grid_resolution=grid_res,
                                             ws_geom=ws_geometry,
                                             gauge_id=gauge_id,
                                             path_to_save=Path(
                                                 f'{place_to_save}'),
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
                                             path_to_save=Path(
                                                 f'{place_to_save}'),
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
                                             path_to_save=Path(
                                                 f'{place_to_save}'),
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
                                             path_to_save=Path(
                                                 f'{place_to_save}'),
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
                                                 path_to_save=Path(
                                                     f'{place_to_save}'),
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
                                             path_to_save=Path(
                                                 f'{place_to_save}'),
                                             nc_pathes=pathes,
                                             dataset=dataset,
                                             var=variable,
                                             aggregation_type=aggregation)
                        meteo_grid.grid_value_ws()


# watershed_geometry = gpd.read_file(
#     f'{config_info.watershed_storage}/{config_info.watershed_name}.gpkg')

# meteo_path = f'{config_info.initial_meteo}'
# era5_land = Path(f'{meteo_path}/era5-land/russia')
# era5 = Path(f'{meteo_path}/era5/russia')
# imerg = Path(f'{meteo_path}/imerg_year_new')
# gpcp = Path(f'{meteo_path}/gpcp_year_new')
# gleam = Path(f'{meteo_path}/gleam_vars')
# mswep = Path(f'{meteo_path}/mswep_new')
# icon = '../geo_data/icon_data'

# place_to_save = f'{config_info.grid_storage}'


# ds_description = {**grid_descriptor(dataset_name='era5_land',
#                                     half_resolution=0.05,
#                                     files=era5_land),
#                   **grid_descriptor(dataset_name='era5',
#                                     half_resolution=0.125,
#                                     files=era5_land),
#                   **grid_descriptor(dataset_name='imerg',
#                                     half_resolution=0.05,
#                                     files=imerg),
#                   **grid_descriptor(dataset_name='gpcp',
#                                     half_resolution=0.25,
#                                     files=gpcp),
#                   **grid_descriptor(dataset_name='gleam',
#                                     half_resolution=0.125,
#                                     files=gleam),
#                   **grid_descriptor(dataset_name='mswep',
#                                     half_resolution=0.05,
#                                     files=mswep)}

# ds_description = {
#     'era5_land': {'res': 0.05,
#                   'f_path': multi_var_nc(era5_land,
#                                          file_extension='nc')},
#     'era5': {'res': 0.125,
#              'f_path': multi_var_nc(era5,
#                                     file_extension='nc')},
#     'imerg': {'res': 0.05,
#               'f_path': multi_var_nc(imerg,
#                                      file_extension='nc')},
#     'gpcp': {'res': 0.25,
#              'f_path': multi_var_nc(gpcp,
#                                     file_extension='nc')},
#     'gleam': {'res': 0.125,
#               'f_path': multi_var_nc(gleam,
#                                      file_extension='nc')},
#     'mswep': {'res': 0.05,
#               'f_path': multi_var_nc(mswep,
#                                      file_extension='nc')},
#     'icon': {'res': 0.0625,
#              'f_path': multi_var_nc(Path(icon),
#                                     file_extension='nc')}}

# for dataset, settings in ds_description.items():

#     grid_res = settings['res']

#     for i, gauge_id in enumerate(tqdm(watershed_geometry['gauge_id'])):
#         print(f"\nNow it's {gauge_id}\n")
#         ws_geometry = watershed_geometry.loc[i, 'geometry']

#         for variable, pathes in settings['f_path'].items():
#             print(f'Calculation for {variable} of {dataset}')
#             aggregation = aggregation_definer(dataset, variable)
#             if dataset == 'imerg':
#                 meteo_grid = Gridder(half_grid_resolution=grid_res,
#                                      ws_geom=ws_geometry,
#                                      gauge_id=gauge_id,
#                                      path_to_save=Path(f'{place_to_save}'),
#                                      nc_pathes=pathes,
#                                      dataset=dataset,
#                                      var=variable,
#                                      aggregation_type=aggregation,
#                                      force_weights=True,
#                                      weight_mark='imerg')
#                 meteo_grid.grid_value_ws()
#             elif dataset == 'gpcp':
#                 meteo_grid = Gridder(half_grid_resolution=grid_res,
#                                      ws_geom=ws_geometry,
#                                      gauge_id=gauge_id,
#                                      path_to_save=Path(f'{place_to_save}'),
#                                      nc_pathes=pathes,
#                                      dataset=dataset,
#                                      var=variable,
#                                      aggregation_type=aggregation,
#                                      force_weights=True,
#                                      weight_mark='gpcp')
#                 meteo_grid.grid_value_ws()
#             elif dataset == 'mswep':
#                 meteo_grid = Gridder(half_grid_resolution=grid_res,
#                                      ws_geom=ws_geometry,
#                                      gauge_id=gauge_id,
#                                      path_to_save=Path(f'{place_to_save}'),
#                                      nc_pathes=pathes,
#                                      dataset=dataset,
#                                      var=variable,
#                                      aggregation_type=aggregation,
#                                      force_weights=True,
#                                      weight_mark='imerg')
#                 meteo_grid.grid_value_ws()
#             elif dataset == 'gleam':
#                 meteo_grid = Gridder(half_grid_resolution=grid_res,
#                                      ws_geom=ws_geometry,
#                                      gauge_id=gauge_id,
#                                      path_to_save=Path(f'{place_to_save}'),
#                                      nc_pathes=pathes,
#                                      dataset=dataset,
#                                      var=variable,
#                                      aggregation_type=aggregation,
#                                      force_weights=True,
#                                      weight_mark='gleam')
#                 meteo_grid.grid_value_ws()
#             elif dataset == 'icon':
#                 for path in pathes:
#                     meteo_grid = Gridder(half_grid_resolution=grid_res,
#                                          ws_geom=ws_geometry,
#                                          gauge_id=gauge_id,
#                                          path_to_save=Path(f'{place_to_save}'),
#                                          nc_pathes=path,
#                                          dataset=dataset,
#                                          var=variable,
#                                          aggregation_type=aggregation,
#                                          force_weights=True,
#                                          weight_mark='icon')
#                     meteo_grid.grid_value_ws()
#             else:
#                 meteo_grid = Gridder(half_grid_resolution=grid_res,
#                                      ws_geom=ws_geometry,
#                                      gauge_id=gauge_id,
#                                      path_to_save=Path(f'{place_to_save}'),
#                                      nc_pathes=pathes,
#                                      dataset=dataset,
#                                      var=variable,
#                                      aggregation_type=aggregation)
#                 meteo_grid.grid_value_ws()
