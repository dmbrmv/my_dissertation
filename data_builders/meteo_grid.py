import glob
from pathlib import Path

from meteo_grid_calculation import MeteoGrids
from scripts.config_definition import config_info
from scripts.loaders import grid_descriptor, merge_cmip_pathes

# inside container ../ outside /home/dima/Data
cmip_245 = "/home/dima/Data/geo_data/cmip_res_interp_025/ssp2_4_5"
cmip_585 = "/home/dima/Data/geo_data/cmip_res_interp_025/ssp5_8_5"

cmip_models_245 = {model.split("/")[-1]: model for model in glob.glob(f"{cmip_245}/*")}
cmip_models_245 = [
    grid_descriptor(dataset_name=model, half_resolution=0.125, files=Path(model_path))
    for model, model_path in cmip_models_245.items()
]
cmip_models_585 = {model.split("/")[-1]: model for model in glob.glob(f"{cmip_585}/*")}
cmip_models_585 = [
    grid_descriptor(dataset_name=model, half_resolution=0.125, files=Path(model_path))
    for model, model_path in cmip_models_585.items()
]
# define data
ds_description = {
    # **grid_descriptor(dataset_name='era5_land',
    #                                 half_resolution=0.05,
    #                                 files=era5_land),
    #   **grid_descriptor(dataset_name='era5',
    #                     half_resolution=0.125,
    #                     files=era5_land),
    #   **grid_descriptor(dataset_name='imerg',
    #                     half_resolution=0.05,
    #                     files=imerg),
    #   **grid_descriptor(dataset_name='gpcp',
    #                     half_resolution=0.25,
    #                     files=gpcp),
    #   **grid_descriptor(dataset_name='gleam',
    #                     half_resolution=0.125,
    #                     files=gleam),
    # **grid_descriptor(dataset_name='mswep',
    #                   half_resolution=0.05,
    #                   files=mswep),
    # **merge_cmip_pathes(cmip_models_245)
    **merge_cmip_pathes(cmip_models_585)
}
print("Meteo for catchments\n")
MeteoGrids(config_info, ds_description)
