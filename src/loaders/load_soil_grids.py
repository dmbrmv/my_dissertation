"""SoilGrids Downloader and Reprojector.

Downloads and reprojects selected SoilGrids layers for a specified AOI.

Usage:
    python soil_grids.py --aoi 70.0 10.0 42.0 45.0 --res_folder ../data/SpatialData/SoilGrids/

Arguments:
    --aoi         North West South East (in WGS84 degrees), e.g. 70.0 10.0 42.0 45.0
    --res_folder  Output folder for results (default: ../data/SpatialData/SoilGrids/)
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fsspec
from osgeo import gdal
from pyproj import CRS, Transformer
from tqdm import tqdm

from src.utils.logger import setup_logger

log = setup_logger(
    "soil_grids_loader", log_file="logs/soil_grids_loader.log", level="INFO"
)

gdal.UseExceptions()

# ----------------------------------------------------------------------
# SoilGrids 250 m v2.0 native CRS = Goode Interrupted Homolosine
SOILGRIDS_CRS = CRS.from_proj4(
    "+proj=igh +lon_0=0 +datum=WGS84 +units=m +no_defs"
)  # equivalent to EPSG:152160 but portable

WGS84 = CRS.from_epsg(4326)
TRANS = Transformer.from_crs(WGS84, SOILGRIDS_CRS, always_xy=True)
# ----------------------------------------------------------------------

ALLOWED_PROPERTIES = {
    "bdod",
    "cec",
    "cfvo",
    "clay",
    "nitrogen",
    "ocd",
    "ocs",
    "phh2o",
    "sand",
    "silt",
    "soc",
}
ALLOWED_DEPTHS = (
    None  # e.g. {"0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"}
)
ALLOWED_STATS = {"mean"}  # e.g. {"mean", "uncertainty", "Q0.05", "Q0.5", "Q0.95"}

full_name_table = {
    "bdod": "Bulk density of the fine earth fraction (cg/cm³)",
    "cec": "Cation Exchange Capacity of the soil (mmol(c)/kg)",
    "cfvo": "Volumetric fraction of coarse fragments (> 2 mm) (cm³/dm³ (vol‰))",
    "clay": "Proportion of clay particles (< 0.002 mm) in the fine earth fraction (g/kg)",
    "nitrogen": "Total nitrogen (cg/kg)",
    "ocd": "Organic carbon density (hg/m³)",
    "ocs": "Organic carbon stocks (t/ha)",
    "phh2o": "Soil pH (pHx10)",
    "sand": "Proportion of sand particles (> 0.05/0.063 mm) in the fine earth fraction (g/kg)",
    "silt": "Proportion of silt particles (≥ 0.002 mm and ≤ 0.05/0.063 mm) in the fine earth fraction (g/kg)",
    "soc": "Soil organic carbon content in the fine earth fraction (dg/kg)",
}
convertion_table = {
    "bdod": 100,  # Bulk density in cg/cm³ to kg/dm³
    "cec": 10,  # Cation Exchange Capacity in mmol(c)/kg to cmol(c)/kg
    "cfvo": 10,  # Coarse fragments in cm³/dm³ (vol‰) to cm³/100cm³ (vol%)
    "clay": 10,  # Clay content in g/kg to g/100g%
    "nitrogen": 100,  # Total nitrogen in cg/kg to g/kg
    "ocd": 10,  # Organic carbon density in hg/m³ to kg/m³
    "ocs": 10,  # Organic carbon stocks in t/ha to kg/m²
    "phh2o": 10,  # Soil pH in pHx10 to pH
    "sand": 10,  # Sand content in g/kg to g/100g%
    "silt": 10,  # Silt content in g/kg to g/100g%
    "soc": 10,  # Soil organic carbon content in dg/kg to g/kg
}

BASE = "https://files.isric.org/soilgrids/latest/data"
IGH = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
RES = 250


def bbox_lonlat_to_homolosine(
    north: float,
    west: float,
    south: float,
    east: float,
    *,
    for_gdal_projwin: bool = True,
) -> tuple[float, float, float, float]:
    """Convert [N, W, S, E] geographic degrees to SoilGrids Homolosine metres.

    Args:
        north (float): Northern latitude.
        west (float): Western longitude.
        south (float): Southern latitude.
        east (float): Eastern longitude.
        for_gdal_projwin (bool): If True, returns (min_x, max_y, max_x, min_y) for GDAL's -projwin.

    Returns:
        tuple: Bounding box in Homolosine metres.

    Raises:
        ValueError: If the AOI crosses a Homolosine lobe.
    """
    x_ul, y_ul = TRANS.transform(west, north)  # upper-left
    x_lr, y_lr = TRANS.transform(east, south)  # lower-right

    if x_lr < x_ul:
        raise ValueError("AOI crosses an interrupted lobe – split it in two.")

    min_x, max_x = x_ul, x_lr
    max_y, min_y = y_ul, y_lr

    if for_gdal_projwin:
        return min_x, max_y, max_x, min_y
    else:
        return min_x, min_y, max_x, max_y


def discover_coverages(
    fs: fsspec.AbstractFileSystem,
    allowed_properties: set,
    allowed_depths: set | None,
    allowed_stats: set | None,
    base_url: str,
) -> dict[str, list[str]]:
    """Discover available SoilGrids coverages.

    Args:
        fs (fsspec.AbstractFileSystem): Filesystem object.
        allowed_properties (set): Allowed soil properties.
        allowed_depths (set or None): Allowed depths.
        allowed_stats (set or None): Allowed statistics.
        base_url (str): Base URL for SoilGrids.

    Returns:
        dict: Mapping from property to list of layer names.
    """
    coverages: dict[str, list[str]] = defaultdict(list)
    for prop_url in fs.ls(base_url, detail=False):
        prop = prop_url.rstrip("/").split("/")[-1]
        if prop in {"landmask", "wrb"} or (
            allowed_properties and prop not in allowed_properties
        ):
            continue
        for fn in fs.ls(prop_url, detail=False):
            if not fn.endswith(".vrt"):
                continue
            name = fn.split("/")[-1].removesuffix(".vrt")
            try:
                prop_, depth_, stat_ = name.split("_", 2)
            except ValueError:
                continue
            if (allowed_depths is None or depth_ in allowed_depths) and (
                allowed_stats is None or stat_ in allowed_stats
            ):
                coverages[prop_].append(name)
    return coverages


def process_layer(
    var: str, layer: str, var_folder: Path, sg_url: str, kwargs: dict
) -> None:
    """Download and reproject a single SoilGrids layer.

    Args:
        var (str): Soil property name.
        layer (str): Layer identifier.
        var_folder (Path): Output directory for the variable.
        sg_url (str): Base SoilGrids URL.
        kwargs (dict): GDAL translate options.

    Raises:
        Exception: If GDAL processing fails.
    """
    out_path = var_folder / f"{layer}.tif"
    try:
        ds = gdal.Translate(str(out_path), sg_url + f"/{var}/{layer}.vrt", **kwargs)
        del ds  # flush contents
        ds = gdal.Warp(str(out_path), str(out_path), dstSRS="EPSG:4326")
        del ds
    except Exception as e:
        raise RuntimeError(f"Error processing {var}/{layer}: {e}") from e


def main(
    aoi: tuple[float, float, float, float],
    res_folder: str = "../data/SpatialData/SoilGrids/",
) -> None:
    """Main function to download and reproject SoilGrids layers.

    Args:
        aoi (tuple): (north, west, south, east) in WGS84 degrees.
        res_folder (str): Output folder for results.
    """
    try:
        bb = bbox_lonlat_to_homolosine(*aoi)
    except Exception as e:
        log.info(f"Invalid AOI: {e}")
        return

    res_folder_path = Path(res_folder)
    res_folder_path.mkdir(parents=True, exist_ok=True)

    fs = fsspec.filesystem("https")
    coverages = discover_coverages(
        fs, ALLOWED_PROPERTIES, ALLOWED_DEPTHS, ALLOWED_STATS, BASE
    )

    sg_url = f"/vsicurl?max_retry=3&retry_delay=1&list_dir=no&url={BASE}"
    kwargs = {
        "format": "GTiff",
        "projWin": bb,
        "projWinSRS": IGH,
        "xRes": RES,
        "yRes": RES,
        "creationOptions": [
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
            "BIGTIFF=YES",
        ],
    }

    for var, layers in coverages.items():
        var_folder = res_folder_path / var
        var_folder.mkdir(parents=True, exist_ok=True)
        tasks = [(var, layer, var_folder, sg_url, kwargs) for layer in layers]
        with (
            ThreadPoolExecutor(max_workers=min(8, len(layers))) as executor,
            tqdm(total=len(tasks), desc=f"Processing {var}", unit="layer") as pbar,
        ):
            futures = [executor.submit(process_layer, *task) for task in tasks]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    log.info(exc)
                pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and reproject SoilGrids layers for a given AOI."
    )
    parser.add_argument(
        "--aoi",
        nargs=4,
        type=float,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        default=[70.0, 10.0, 42.0, 45.0],
        help="AOI as four floats: north west south east (WGS84 degrees). Example: 70.0 10.0 42.0 45.0",
    )
    parser.add_argument(
        "--res_folder",
        type=str,
        default="../data/SpatialData/SoilGrids/",
        help="Output folder for results (default: ../data/SpatialData/SoilGrids/)",
    )
    args = parser.parse_args()
    main(tuple(args.aoi), args.res_folder)
