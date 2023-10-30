import sys
sys.path.append('/home/anton/dima_experiments/my_dissertation/')
from meteo_grids_parser.scripts.grid_calculator import Gridder
from meteo_grids_parser.scripts.loaders import (aggregation_definer,
                                                grid_descriptor)
import geopandas as gpd
import pandas as pd
from functools import reduce
import glob
from pathlib import Path
from tqdm import tqdm


class Icon_merger:

    def __init__(self,
                 ws_path: str, icon_gauges: str,
                 place_to_save: str, dataset_name: str):
        self.ws_path = ws_path
        self.icon_gauges = icon_gauges
        self.place_to_save = place_to_save
        self.dataset_name = dataset_name

    def merger(self):

        ws_data = gpd.read_file(f'{self.ws_path}')

        ds_description = {**grid_descriptor(dataset_name=f'{self.dataset_name}',
                                            half_resolution=0.0625,
                                            files=Path(self.icon_gauges))}

        for dataset, settings in ds_description.items():

            grid_res = settings['res']

            for i, gauge_id in enumerate(ws_data['gauge_id']):
                ws_geometry = ws_data.loc[i, 'geometry']
                for variable, pathes in settings['f_path'].items():
                    aggregation = aggregation_definer(dataset, variable)
                    for path in pathes:
                        meteo_grid = Gridder(half_grid_resolution=grid_res,
                                             ws_geom=ws_geometry,
                                             gauge_id=gauge_id,
                                             path_to_save=Path(
                                                 f'{self.place_to_save}'),
                                             nc_pathes=path,
                                             dataset=dataset,
                                             var=variable,
                                             aggregation_type=aggregation,
                                             prcp_coef=1e-2,
                                             weight_mark='icon',
                                             extend_data=False,
                                             merge_data=True)
                        meteo_grid.grid_value_ws()

        save_folder = Path(f'{self.place_to_save}/merge_result')
        save_folder.mkdir(exist_ok=True, parents=True)
        csv_folder = Path(f'{self.place_to_save}/{self.dataset_name}')

        gauge_dict = dict()

        for i, gauge_id in enumerate(tqdm(ws_data['gauge_id'],
                                          'Merge for gauges')):
            gauge_dict[gauge_id] = list()
            for path in glob.glob(f'{csv_folder}/*/*{gauge_id}*'):
                csv_file = pd.read_csv(path)
                gauge_dict[gauge_id].append(csv_file)
            gauge_dict[gauge_id] = reduce(lambda df1, df2: pd.merge(df1, df2),
                                          gauge_dict[gauge_id])
            gauge_dict[gauge_id] = gauge_dict[gauge_id].rename(
                columns={'TMAX_2M': 'icon_t2_max', '2t': 'icon_t2_avg',
                         'al': 'icon_albedo', 'tp': 'icon_prcp',
                         'TMIN_2M': 'icon_t2_min'})
            gauge_dict[gauge_id] = gauge_dict[gauge_id][
                ['date', 'icon_t2_min', 'icon_t2_avg',
                 'icon_t2_max', 'icon_prcp', 'forecast_horizon']]
            gauge_dict[gauge_id][['icon_t2_min',
                                  'icon_t2_avg', 'icon_t2_max']] -= 273.15
            gauge_dict[gauge_id].to_csv(f'{save_folder}/{gauge_id}.csv',
                                        index=False)
