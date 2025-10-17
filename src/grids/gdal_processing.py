from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
from osgeo import gdal
import rasterio
from rasterio.warp import transform

gdal.UseExceptions()


def gdal_extent_clipper(
    initial_tiff: str | Path,
    extent: tuple[float, float, float, float],
    tmp_tiff: str | Path,
    final_tiff: str | Path,
    crs_epsg: int,
) -> None:
    """Clip and project a .tiff file for a desired extent and EPSG code.

    This function takes a .tiff file, clips it to the desired extent, and then
    projects it to the specified EPSG code.

    Args:
        initial_tiff (Union[str, Path]): Path to the input .tiff file to be trimmed.
        extent (tuple[float, float, float, float]): (minX, maxY, maxX, minY)
            coordinates for area of interest.
        tmp_tiff (Union[str, Path]): Path for the intermediate clipped file (not projected).
        final_tiff (Union[str, Path]): Path for the final projected and trimmed file.
        crs_epsg (int): Target EPSG code for reprojection.

    Raises:
        FileNotFoundError: If the input file does not exist.
        RuntimeError: If GDAL operations fail.

    Returns:
        None
    """
    initial_tiff = str(initial_tiff)
    tmp_tiff = str(tmp_tiff)
    final_tiff = str(final_tiff)

    if not Path(initial_tiff).is_file():
        raise FileNotFoundError(f"Input file not found: {initial_tiff}")

    # Clip input tiff to extent
    clipped_ds = gdal.Translate(destName=tmp_tiff, srcDS=initial_tiff, projWin=extent)
    if clipped_ds is None:
        raise RuntimeError(f"GDAL Translate failed for {initial_tiff} with extent {extent}")
    clipped_ds.FlushCache()

    # Reproject the clipped data to the desired EPSG code
    projected_ds = gdal.Warp(
        destNameOrDestDS=final_tiff,
        format="GTiff",
        dstNodata=None,
        srcDSOrSrcDSTab=tmp_tiff,
        dstSRS=f"EPSG:{crs_epsg}",
    )
    if projected_ds is None:
        raise RuntimeError(f"GDAL Warp failed for {tmp_tiff} to EPSG:{crs_epsg}")
    projected_ds.FlushCache()


def create_mosaic(
    file_path: str | Path,
    file_name: str,
    tiles: Sequence[str | Path],
) -> str:
    """Generate a VRT mosaic for GDAL from a list of .tiff tiles.

    Args:
        file_path (Union[Path, str]): Directory where the mosaic will be saved.
        file_name (str): Name for the output VRT file (without extension).
        tiles (Sequence[Union[str, Path]]): List of .tiff file paths to include in the mosaic.

    Raises:
        ValueError: If the tiles list is empty.
        RuntimeError: If GDAL BuildVRT fails.

    Returns:
        str: Path to the created VRT mosaic.
    """
    if not tiles:
        raise ValueError("Tiles list must not be empty.")

    file_path = Path(file_path)
    file_path.mkdir(parents=True, exist_ok=True)
    file_target = str(file_path / f"{file_name}.vrt")

    # Convert all tile paths to strings for GDAL compatibility
    tile_paths = [str(tile) for tile in tiles]

    # Build a virtual raster mosaic (VRT) from the list of tiles
    mosaic_ds = gdal.BuildVRT(destName=file_target, srcDSOrSrcDSTab=tile_paths)
    if mosaic_ds is None:
        raise RuntimeError(f"GDAL BuildVRT failed for tiles: {tile_paths}")
    mosaic_ds.FlushCache()

    return file_target


def get_point_height_from_dem(pt_geoser: gpd.GeoSeries, dem_path):
    """Retrieve the height of a point from a digital elevation model (DEM).

    Parameters
    ----------
    pt_geom : Point
        The geographical point representing the location.
    gauge_id : str
        Identifier for the gauge, used to locate the corresponding DEM file.
    dem_path : str
        Path to the DEM file.

    Returns:
    -------
    float
        The elevation value at the specified point.

    """
    point_crs = pt_geoser.crs
    pt_geom = pt_geoser.geometry.values[0]  # Extract the geometry
    # Ensure the geometry is a shapely Point
    if not hasattr(pt_geom, "x") or not hasattr(pt_geom, "y"):
        raise TypeError("The provided geometry is not a Point and does not have 'x' and 'y' attributes.")
    # Open the DEM file
    with rasterio.open(dem_path) as src:
        # ── 1. Re-project point to the DEM’s CRS, if necessary ────────────────────────
        if src.crs != point_crs:
            coords = transform(
                point_crs,  # source CRS (point layer)
                src.crs,  # target CRS (DEM)
                [pt_geom.x],
                [pt_geom.y],  # coordinate arrays
            )
            x_pt, y_pt = coords[0], coords[1]
        else:
            x_pt, y_pt = [pt_geom.x], [pt_geom.y]

        # ── 2. Convert projected coordinates to DEM row/col ───────────────────────────
        row, col = src.index(x_pt[0], y_pt[0])  # integer pixel indices

        # ── 3. Read the single value ──────────────────────────────────────────────────
        value = src.read(1)[row, col]  # band 1 as 2-D array
        # If nodata handling matters:
        if value == src.nodata:
            value = np.nan

    return value
