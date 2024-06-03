import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pysheds.grid import Grid  # type: ignore
from scripts.config_definition import config_info
from shapely.geometry import LineString


class RiverNetworkGPKG:
    def __init__(self, config_info, tile_tag: str) -> None:
        self.config_info = config_info
        self.tile_tag = tile_tag

    def network_creator(self):
        acc_tiff = f"{config_info.raster_storage}/acc/{self.tile_tag}_acc.tiff"
        print(acc_tiff)
        # initialize calculation grid
        acc = xr.open_dataset(acc_tiff)

        # define lat, lon square split
        min_lon = np.round(acc["x"].min().values)
        max_lon = np.round(acc["x"].max().values)

        min_lat = np.round(acc["y"].min().values)
        max_lat = np.round(acc["y"].max().values)

        lons = np.arange(start=min_lon, stop=max_lon + 2.5, step=2.5)
        lats = np.arange(start=min_lat, stop=max_lat + 2.5, step=2.5)
        # write tiles from whole raster in predefined place
        acc_tiles_storage = Path(
            f"{config_info.river_net_storage}/rasters/{self.tile_tag}"
        )
        acc_tiles_storage.mkdir(exist_ok=True, parents=True)
        tile_counter = 0
        for lon_index in range(len(lons) - 2):
            new_min_lon, new_max_lon = lons[lon_index : lon_index + 2]
            for lat_index in range(len(lats) - 2):
                new_min_lat, new_max_lat = lats[lat_index : lat_index + 2]

                tile_acc = (
                    acc.where(acc.x <= new_max_lon, drop=True)
                    .where(acc.x >= new_min_lon, drop=True)
                    .where(acc.y <= new_max_lat, drop=True)
                    .where(acc.y >= new_min_lat, drop=True)
                )
                tile_acc = tile_acc.squeeze("band")
                tile_acc = tile_acc.drop(["band"])
                tile_counter += 1
                tile_acc.rio.to_raster(
                    f"{acc_tiles_storage}/{self.tile_tag}_{tile_counter}.tiff"
                )

    def get_river_geom(self, pseudo_rank, branches):
        temp_river = pd.DataFrame()
        temp_river["id"] = 0
        temp_river["rank"] = ""
        temp_river["geometry"] = np.NaN

        for i, branch in enumerate(branches["features"]):
            line = LineString(branch["geometry"]["coordinates"])
            temp_river.loc[i, "id"] = int(branch["id"])
            temp_river.loc[i, "rank"] = pseudo_rank
            temp_river.loc[i, "geometry"] = line  # type: ignore

        return temp_river

    def rank_to_acc(self, rank_row: pd.Series):
        if rank_row["rank"] == "big_creeks":
            return f"{1e3:.0f} - {1e4:.0f}"
        elif rank_row["rank"] == "small_rivers":
            return f"{1e4:.0f} - {1e5:.0f}"
        elif rank_row["rank"] == "medium_rivers":
            return f"{1e5:.0f} - {1e6:.0f}"
        elif rank_row["rank"] == "rivers":
            return f"{1e6:.0f} - {1e7:.0f}"
        elif rank_row["rank"] == "big_rivers":
            return f"{1e7:.0f} - {1e8:.0f}"
        elif rank_row["rank"] == "large_rivers":
            return f"{1e8:.0f} - {1e9:.0f}"
        else:
            return f"{1e2:.0f} - {1e3:.0f}"

    def river_separator(self):
        geometry_storage = Path(
            f"{config_info.river_net_storage}/geometry/{self.tile_tag}"
        )
        geometry_storage.mkdir(exist_ok=True, parents=True)
        # get preselected tiles
        tiles_to_network = glob.glob(
            f"{config_info.river_net_storage}/rasters/{self.tile_tag}/*.tiff"
        )

        dir_tiff = f"{config_info.raster_storage}/fdir/{self.tile_tag}_dir.tif"

        # initialize calculation grid
        grid = Grid.from_raster(dir_tiff, data_name="fdir_grid", nodata=0)
        fdir = grid.read_raster(dir_tiff, data_name="fdir", nodata=0)

        for tile in tiles_to_network:
            tile_name = tile.split("/")[-1][:-5]
            acc = grid.read_raster(tile, data_name="acc", nodata=0)

            river_parts = {
                "small_creeks": grid.extract_river_network(
                    fdir, (1e2 < acc) & (acc <= 1e3)
                ),  # type: ignore
                "big_creeks": grid.extract_river_network(
                    fdir, (1e3 < acc) & (acc <= 1e4)
                ),  # type: ignore
                "small_rivers": grid.extract_river_network(
                    fdir, (1e4 < acc) & (acc <= 1e5)
                ),  # type: ignore
                "medium_rivers": grid.extract_river_network(
                    fdir, (1e5 < acc) & (acc <= 1e6)
                ),  # type: ignore
                "rivers": grid.extract_river_network(fdir, (1e6 < acc) & (acc <= 1e7)),  # type: ignore
                "big_rivers": grid.extract_river_network(
                    fdir, (1e7 < acc) & (acc <= 1e8)
                ),  # type: ignore
                "large_rivers": grid.extract_river_network(fdir, (1e8 < acc)),
            }  # type: ignore

            temp_res_riv = gpd.GeoDataFrame(
                pd.concat(
                    [
                        self.get_river_geom(rank, network)
                        for rank, network in river_parts.items()
                    ]
                ),
                geometry="geometry",
            )  # type: ignore
            temp_res_riv = temp_res_riv.set_crs(epsg=4326)
            temp_res_riv["acc_range"] = temp_res_riv.apply(
                lambda row: self.rank_to_acc(row), axis=1
            )
            temp_res_riv = temp_res_riv[["id", "rank", "acc_range", "geometry"]]

            temp_res_riv.to_file(f"{geometry_storage}/{tile_name}.gpkg")  # type: ignore
