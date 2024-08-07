from scripts.data_preparation import read_gauge, tile_determinator
from scripts.grid_calc import my_catchment
import geopandas as gpd
import glob
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import gc

path_to_save = './data/great_db'
final_save = Path(f'{path_to_save}/kamchatka')
final_save.mkdir(exist_ok=True, parents=True)

gauge_file = gpd.read_file(
    './data/initial_gauges/great_db/kamchatka_gauges.gpkg')

_ = tile_determinator(gauge_file,
                      final_save)

target_files = glob.glob(f'{final_save}/*.gpkg')


for target in tqdm(target_files, 'Parsing through sub-gauges'):

    tile_tag = target.split('/')[-1][:-5]

    test_gauge = gpd.read_file(target, encoding='utf-8')
    test_gauge = read_gauge(test_gauge)

    dir_tiff = f'./data/aois_rasters/fdir/{tile_tag}_dir.tif'
    acc_tiff = f'./data/aois_rasters/acc/{tile_tag}_acc.tiff'

    _ = my_catchment(grid_p=dir_tiff,
                     fdir_p=dir_tiff,
                     facc_p=acc_tiff,
                     acc_faktor=1e3,
                     gauges_file=test_gauge,
                     save_p=f'{final_save}/catchments',
                     region_name=f'{tile_tag}_mo')
    gc.collect()

res_folder = Path(f'{final_save}/res')
res_folder.mkdir(exist_ok=True, parents=True)

concat_list = pd.concat([
    gpd.read_file(gauge, encoding='utf-8')
    for gauge in
    glob.glob(f'{final_save}/catchments/*.gpkg')]).reset_index(drop=True)

concat_list.to_file(f'{res_folder}/kamchatka_ws.gpkg',
                    encoding='utf-8')
