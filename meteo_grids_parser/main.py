import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from pathlib import Path
import geopandas as gpd
from tqdm import tqdm

from scripts.grid_calculator import Gridder
from scripts.loaders import aggregation_definer, grid_descriptor


# import time
russia_ws = gpd.read_file("/home/dima/ESG/hsi/my_dissertation/geo_data/geometry/russia_cis_ws.gpkg")
# print("I'm waiting for copy..")
# time.sleep(7200)
# print("Let's roll !")
meteo_path = "/home/dima/ESG/meteo_data_nc"
era5_land = Path(f"{meteo_path}/era5-land/russia")
era5 = Path(f"{meteo_path}/era5/russia")
imerg = Path(f"{meteo_path}/imerg_year_new")
gpcp = Path(f"{meteo_path}/gpcp_year_new")
gleam = Path(f"{meteo_path}/gleam_vars")
mswep = Path(f"{meteo_path}/mswep_new")
icon = Path("/home/dima/ESG/icon_data")
# define data
ds_description = {
    **grid_descriptor(dataset_name="era5_land", half_resolution=0.05, files=era5_land),
    **grid_descriptor(dataset_name="era5", half_resolution=0.125, files=era5),
    **grid_descriptor(dataset_name="imerg", half_resolution=0.05, files=imerg),
    **grid_descriptor(dataset_name="gpcp", half_resolution=0.25, files=gpcp),
    **grid_descriptor(dataset_name="gleam", half_resolution=0.125, files=gleam),
    **grid_descriptor(dataset_name="mswep", half_resolution=0.05, files=mswep),
}

place_to_save = Path("/home/dima/ESG/hsi/my_dissertation/geo_data/meteo_grids_2024/")
place_to_save.mkdir(exist_ok=True, parents=True)

for dataset, settings in ds_description.items():
    grid_res = settings["res"]

    for i, gauge_id in enumerate(tqdm(russia_ws["gauge_id"])):
        ws_geometry = russia_ws.loc[i, "geometry"]
        print(f"\n{gauge_id}\n")
        # if gauge_id != "2272":
        #     continue
        for variable, pathes in settings["f_path"].items():
            print(f"Calculation for {variable} of {dataset}")
            aggregation = aggregation_definer(dataset, variable)
            if dataset == "imerg":
                meteo_grid = Gridder(
                    half_grid_resolution=grid_res,
                    ws_geom=ws_geometry,
                    gauge_id=gauge_id,
                    path_to_save=Path(f"{place_to_save}"),
                    nc_pathes=pathes,
                    dataset=dataset,
                    var=variable,
                    aggregation_type=aggregation,
                    force_weights=True,
                    weight_mark="imerg",
                )
                meteo_grid.grid_value_ws()
            elif dataset == "gpcp":
                meteo_grid = Gridder(
                    half_grid_resolution=grid_res,
                    ws_geom=ws_geometry,
                    gauge_id=gauge_id,
                    path_to_save=Path(f"{place_to_save}"),
                    nc_pathes=pathes,
                    dataset=dataset,
                    var=variable,
                    aggregation_type=aggregation,
                    force_weights=True,
                    weight_mark="gpcp",
                )
                meteo_grid.grid_value_ws()
            elif dataset == "mswep":
                meteo_grid = Gridder(
                    half_grid_resolution=grid_res,
                    ws_geom=ws_geometry,
                    gauge_id=gauge_id,
                    path_to_save=Path(f"{place_to_save}"),
                    nc_pathes=pathes,
                    dataset=dataset,
                    var=variable,
                    aggregation_type=aggregation,
                    force_weights=True,
                    weight_mark="imerg",
                )
                meteo_grid.grid_value_ws()
            elif dataset == "gleam":
                meteo_grid = Gridder(
                    half_grid_resolution=grid_res,
                    ws_geom=ws_geometry,
                    gauge_id=gauge_id,
                    path_to_save=Path(f"{place_to_save}"),
                    nc_pathes=pathes,
                    dataset=dataset,
                    var=variable,
                    aggregation_type=aggregation,
                    force_weights=True,
                    weight_mark="gleam",
                )
                meteo_grid.grid_value_ws()
            elif dataset == "icon":
                for path in pathes:
                    meteo_grid = Gridder(
                        half_grid_resolution=grid_res,
                        ws_geom=ws_geometry,
                        gauge_id=gauge_id,
                        path_to_save=Path(f"{place_to_save}"),
                        nc_pathes=path,
                        dataset=dataset,
                        var=variable,
                        aggregation_type=aggregation,
                        force_weights=True,
                        weight_mark="icon",
                    )
                    meteo_grid.grid_value_ws()
            else:
                meteo_grid = Gridder(
                    half_grid_resolution=grid_res,
                    ws_geom=ws_geometry,
                    gauge_id=gauge_id,
                    path_to_save=Path(f"{place_to_save}"),
                    nc_pathes=pathes,
                    dataset=dataset,
                    var=variable,
                    aggregation_type=aggregation,
                )
                meteo_grid.grid_value_ws()
