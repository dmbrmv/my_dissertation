import glob
from pathlib import Path

import geopandas as gpd
from scripts.data_preparation import aoi_tiles
from scripts.geom_functions import poly_from_multipoly
from scripts.grid_calc import dir_acc_for_aoi
from tqdm import tqdm


class AccumulationMask:
    def __init__(self, config_info) -> None:
        raster_storage = config_info.raster_storage

        mask_files = {
            file.split("/")[-1][:-9]: poly_from_multipoly(
                gpd.read_file(file, encoding="utf-8").loc[0, "geometry"]
            )
            for file in glob.glob(f"{config_info.mask_storage}/*.gpkg")
        }

        path_to_masks = Path(f"{config_info.accum_masks}")
        path_to_masks.mkdir(exist_ok=True, parents=True)

        for region, geom in mask_files.items():
            geom = gpd.GeoDataFrame(index=[0], geometry=[geom])  # type: ignore
            geom = geom.set_crs(epsg=4326)
            geom.to_file(f"{path_to_masks}/{region}_mask.gpkg")  # type: ignore

        masks_aoi = {
            file.split("/")[-1][:-10]: file
            for file in glob.glob(f"{path_to_masks}/*.gpkg")
        }

        acc_ready = [
            existing.split("/")[-1][:-9]
            for existing in glob.glob(f"{raster_storage}/acc/*.tiff")
        ]

        for region, aoi_file in tqdm(masks_aoi.items()):
            tiles = aoi_tiles(topo_p=f"{config_info.initial_fdir}", aoi_shp=aoi_file)[
                "dir"
            ]
            try:
                if region not in acc_ready:
                    print(f"\nРасчёт для {region}")
                    dir_acc_for_aoi(
                        region_name=region,
                        tiles=tiles,
                        result_folder=f"{raster_storage}",
                    )
                else:
                    print(f"{region} уже посчитан")
            except IndexError:
                with open(f"{raster_storage}/error_tiles.txt", "w") as f:
                    for tile in tiles:
                        f.writelines(f"{tile}\n")
                continue
