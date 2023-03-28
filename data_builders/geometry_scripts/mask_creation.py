import geopandas as gpd
import glob
from pathlib import Path
from tqdm import tqdm

from geom_functions import poly_from_multipoly
from data_preparation import aoi_tiles
from grid_calc import dir_acc_for_aoi


raster_storage = config_info.raster_storage

mask_files = {file.split('/')[-1][:-9]: select_big_from_MP(gpd.read_file(
    file, encoding='utf-8').loc[0, 'geometry'])
    for file in glob.glob(f'{config_info.mask_storage}/*.gpkg')}

path_to_masks = Path(f'{config_info.accum_masks}')
path_to_masks.mkdir(exist_ok=True, parents=True)

for region, geom in mask_files.items():
    geom = gpd.GeoDataFrame(index=[0],
                            geometry=[geom])  # type: ignore
    geom = geom.set_crs(epsg=4326)
    geom.to_file(f'{path_to_masks}/{region}_mask.gpkg')

masks_aoi = {file.split('/')[-1][:-10]: file
             for file in glob.glob(f'{path_to_masks}/*.gpkg')}

acc_ready = [existing.split('/')[-1][:-9]
             for existing
             in glob.glob(f'{raster_storage}/acc/*.tiff')]

for region, aoi_file in tqdm(masks_aoi.items()):
    tiles = aoi_tiles(topo_p=f'{config_info.initial_fdir}',
                      aoi_shp=aoi_file)['dir']
    try:
        if region not in acc_ready:
            print(f'\nРасчёт для {region}')
            dir_acc_for_aoi(region_name=region,
                            tiles=tiles,
                            result_folder=f'{raster_storage}')
        else:
            print(f'{region} уже посчитан')
    except IndexError:
        with open(f'{raster_storage}/error_tiles.txt', 'w') as corrupt_file:
            for tile in tiles:
                corrupt_file.writelines(f'{tile}\n')
        continue
