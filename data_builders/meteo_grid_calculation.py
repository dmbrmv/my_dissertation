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
