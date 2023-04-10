import glob
from tqdm import tqdm
from pathlib import Path
from scripts.config_definition import config_info
from scripts.loaders import grid_descriptor
from mask_creation import AccumulationMask
from river_creator import RiverNetworkGPKG
from watershed_creator import WatershedGPKG
from meteo_grid_calculation import MeteoGrids
print('Accumulation masks calculation\n')
# accumulation masks from flow direction
AccumulationMask(config_info)
# river network image based on accumulation mask
raster_tags = [file.split('/')[-1][:-4]
               for file
               in glob.glob(f'{config_info.raster_storage}/*.vrt')]
# get accumulations
print('River network creation\n')
for tile_tag in tqdm(raster_tags, 'rasters ..'):
    river_net = RiverNetworkGPKG(config_info=config_info,
                                 tile_tag=tile_tag)
    river_net.network_creator()
# deselect from small squares
for tile_tag in tqdm(raster_tags, 'geometry ..'):
    river_net = RiverNetworkGPKG(config_info=config_info,
                                 tile_tag=tile_tag)
    river_net.river_separator()
# create watershed geometry from points
print('Catchments for gauges\n')
WatershedGPKG(config_info)
# calculate meteorology from defined geometry
meteo_path = f'{config_info.initial_meteo}'
era5_land = Path(f'{meteo_path}/era5-land/russia')
era5 = Path(f'{meteo_path}/era5/russia')
imerg = Path(f'{meteo_path}/imerg_year_new')
gpcp = Path(f'{meteo_path}/gpcp_year_new')
gleam = Path(f'{meteo_path}/gleam_vars')
mswep = Path(f'{meteo_path}/mswep_new')
# define data
ds_description = {**grid_descriptor(dataset_name='era5_land',
                                    half_resolution=0.05,
                                    files=era5_land),
                  **grid_descriptor(dataset_name='era5',
                                    half_resolution=0.125,
                                    files=era5_land),
                  **grid_descriptor(dataset_name='imerg',
                                    half_resolution=0.05,
                                    files=imerg),
                  **grid_descriptor(dataset_name='gpcp',
                                    half_resolution=0.25,
                                    files=gpcp),
                  **grid_descriptor(dataset_name='gleam',
                                    half_resolution=0.125,
                                    files=gleam),
                  **grid_descriptor(dataset_name='mswep',
                                    half_resolution=0.05,
                                    files=mswep)}
print('Meteo for catchments\n')
MeteoGrids(config_info, ds_description)
