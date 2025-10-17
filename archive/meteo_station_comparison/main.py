import sys

sys.path.append("/home/anton/dima_experiments/my_dissertation")
import glob

import geopandas as gpd
import pandas as pd
import xarray as xr
from scripts.file_readers import read_meteo_station, station_lat_lon, xr_for_point
from tqdm import tqdm

from data_builders.scripts.loaders import multi_var_nc

gdf_file = gpd.read_file("../data/great_db/meteo_stations.gpkg")
gdf_file["gauge_id"] = gdf_file["gauge_id"].astype(str)
gdf_file = gdf_file.set_index("gauge_id")

meteo_dict = {
    "era5_land": multi_var_nc(
        path_to_nc="../data/meteorology/era5-land/russia", file_extension="nc"
    ),
    "era5": multi_var_nc(
        path_to_nc="../data/meteorology/era5/russia", file_extension="nc"
    ),
    "gpcp": multi_var_nc(
        path_to_nc="../data/meteorology/gpcp_year_new", file_extension="nc"
    ),
    "imerg": multi_var_nc(
        path_to_nc="../data/meteorology/imerg_year_new", file_extension="nc"
    ),
    "mswep": multi_var_nc(
        path_to_nc="../data/meteorology/mswep_new", file_extension="nc"
    ),
}

for file in tqdm(
    glob.glob("../data/meteorology/meteo_ru/*.csv"),
    desc="Calculation for Meteo Stations",
):
    station_id = file.split("/")[-1][:-4]
    # read df for meteo station
    meteo_station = read_meteo_station(file)

    meteo_lon, meteo_lat = station_lat_lon(gdf_file.loc[f"{station_id}", "geometry"])

    for ds, data_desc in meteo_dict.items():
        for var, pathes in data_desc.items():
            if "precipitation" in var:
                col_var = f"{ds}_prcp"
                xr_file = xr.open_mfdataset(pathes)
                if "longitude" in xr_file.dims:
                    xr_file = xr_file.rename({"longitude": "lon", "latitude": "lat"})
                xr_df = xr_for_point(xr_file, col_var, meteo_lat, meteo_lon)
                meteo_station = pd.concat([meteo_station, xr_df], axis=1)
            else:
                col_var = "t2m"
                col_var = var.split("_")
                col_var[1] = ds
                col_var[1], col_var[0], col_var[2] = col_var[0], col_var[1], col_var[2]
                col_var = "_".join(col_var)

                xr_file = xr.open_mfdataset(pathes)
                if "longitude" in xr_file.dims:
                    xr_file = xr_file.rename({"longitude": "lon", "latitude": "lat"})
                xr_df = xr_for_point(xr_file, col_var, meteo_lat, meteo_lon)
                meteo_station = pd.concat([meteo_station, xr_df], axis=1)
    meteo_station = meteo_station.loc["2008":"2020", :]
    meteo_station.to_csv(f"../data/meteorology/station_vs_reanalysis/{station_id}.csv")
