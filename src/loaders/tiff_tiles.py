from itertools import chain, product
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon

from src.geometry.geom_processing import (
    create_gdf,
    gauge_buffer_creator,
    gauge_to_utm,
)
from src.grids.gdal_processing import create_mosaic, gdal_extent_clipper


def tile_for_gauge(gauge_geom: Point, mask_storage: Path | str, ws_area: float) -> str:
    """Define tag from each AOI drawn across Russia for gauge depenging on its watershed area.

    Args:
    ----
        gauge_geom (Point): Geometry instance of gauge
        mask_storage (Union[pathlib.Path, str]): Storage with .GPKG aoi files
        ws_area (float): Area of gauge watershed

    Returns:
    -------
        str: tag for further calls into predifined .tiff files for acc and dir

    """
    all_tiles = Path(mask_storage).glob("*.gpkg")

    tile_dict = {}

    for tile in all_tiles:
        # trim "_aoi" letters from tag
        tile_name = Path(tile).stem[:-4]
        aoi = gpd.read_file(tile, encoding="utf-8").loc[0, "geometry"]
        if aoi.contains(gauge_geom):
            tile_dict[tile_name] = aoi.area

    if ws_area > 24000:
        tile_tag = max(tile_dict, key=tile_dict.get)
    else:
        tile_tag = min(tile_dict, key=tile_dict.get)

    return tile_tag


def round_up(x: float, round_val: float = 5) -> int:
    """Round up a number to the nearest multiple of `round_val`.

    Args:
        x: The number to round up.
        round_val: The value to round up to. Defaults to 5.

    Returns:
        The rounded up number as an integer.

    """
    if round_val <= 0:
        raise ValueError("round_val must be positive.")
    return int(np.ceil(x / round_val)) * int(round_val)


def round_down(x: float, round_val: float = 5) -> int:
    """Round down a number to the nearest multiple of `round_val`.

    Args:
        x: The number to round down.
        round_val: The value to round down to. Defaults to 5.

    Returns:
        The rounded down number as an integer.

    """
    if round_val <= 0:
        raise ValueError("round_val must be positive.")
    return int(np.floor(x / round_val)) * int(round_val)


def format_tile(lat: int, lon: int) -> str:
    """Format tile name based on latitude and longitude.

    Args:
        lat: Latitude integer.
        lon: Longitude integer.

    Returns:
        Formatted tile string.
    """
    lat_prefix = f"n{abs(lat)}" if lat >= 0 else f"s{abs(lat)}"
    lon_prefix = f"e{abs(lon):03}" if lon >= 0 else f"w{abs(lon):03}"
    return f"{lat_prefix}{lon_prefix}"


def roi_extent_tiles(
    topo_p: Path | str, extent_coords: tuple[float, float, float, float]
) -> dict[str, list[Path]]:
    """Return the paths for .tiff files of elevation and flow direction in a given folder.

    Args:
        topo_p: Folder with pre-downloaded files.
        extent_coords: tuple with extent (x_min, y_max, x_max, y_min).

    Returns:
        Dictionary with ['elv'] and ['dir'] keys, containing the corresponding files
        for the gauge of interest.

    Raises:
        ValueError: If extent_coords is not a tuple of four floats.
    """
    if not (
        isinstance(extent_coords, tuple)
        and len(extent_coords) == 4
        and all(isinstance(v, float | int) for v in extent_coords)
    ):
        raise ValueError(
            "extent_coords must be a tuple of four floats or ints (x_min, y_max, x_max, y_min)."
        )

    x_min, y_max, x_max, y_min = extent_coords

    # Adjust extent coordinates
    x_min, y_min = round_down(x_min), round_down(y_min)
    x_max, y_max = round_up(x_max), round_up(y_max)

    # Generate tile boundaries
    latitudes = np.arange(start=y_min, stop=y_max + 5, step=5, dtype=int)
    longitudes = np.arange(start=x_min, stop=x_max + 5, step=5, dtype=int)
    tile_boundaries = [format_tile(lat, lon) for lat, lon in product(latitudes, longitudes)]

    variables = ["dir", "elv"]
    topo_path = Path(topo_p)

    tiles: dict[str, list[Path]] = {}
    for var in variables:
        var_dir = topo_path / var
        if not var_dir.exists():
            raise FileNotFoundError(f"Directory '{var_dir}' does not exist.")
        found_tiles = list(
            chain.from_iterable(var_dir.rglob(f"{boundary}_{var}.tiff") for boundary in tile_boundaries)
        )
        tiles[var] = found_tiles

    return tiles


def create_tiff_get_area(
    tmp_raster_folder: Path,
    gauge_id: str,
    geom_point: gpd.GeoSeries,
    ws_geom: Polygon | MultiPolygon,
    fdir_path: str | Path,
    result_tiff_storage: str | Path,
) -> tuple[float, float]:
    """Create a clipped elevation TIFF for a gauge and calculate watershed area.

    Clips the elevation raster to the AOI buffer and mosaics the tiles. Calculates
    the watershed area and accumulation coefficient.

    Args:
        tmp_raster_folder: Temporary folder for intermediate files.
        gauge_id: Gauge identifier.
        geom_point: Gauge point geometry.
        ws_geom: Watershed geometry (Polygon or MultiPolygon).
        fdir_path: Path to the flow direction raster directory.
        result_tiff_storage: Output directory for the clipped elevation TIFF.

    Returns:
        Tuple containing (accumulation coefficient, watershed area in sq. km).

    Raises:
        FileNotFoundError: If required directories or files are missing.
        ValueError: If input geometries are invalid.
    """
    # Validate input types
    if not isinstance(gauge_id, str) or not gauge_id:
        raise ValueError("gauge_id must be a non-empty string.")
    if not isinstance(geom_point, gpd.GeoSeries):
        raise ValueError("geom_point must be a GeoSeries.")
    if not isinstance(ws_geom, Polygon | MultiPolygon):
        raise ValueError("ws_geom must be a Polygon or MultiPolygon.")
    if not isinstance(fdir_path, str) or not fdir_path:
        raise ValueError("fdir_path must be a non-empty string.")
    if not isinstance(result_tiff_storage, str) or not result_tiff_storage:
        raise ValueError("result_tiff_storage must be a non-empty string.")

    tif_trim_folder = tmp_raster_folder / "trimmed_tifs"
    elv_path_dir = Path(result_tiff_storage)

    # Create necessary directories
    for folder in (tmp_raster_folder, tif_trim_folder, elv_path_dir):
        folder.mkdir(exist_ok=True, parents=True)

    # Project gauge to UTM and create AOI buffer
    _, tif_epsg = gauge_to_utm(gauge_series=geom_point, return_gdf=True)  # type: ignore

    # Ensure the geometry is a Point
    gauge_geom = geom_point.geometry.values[0]
    if not gauge_geom.geom_type == "Point":
        raise ValueError("geom_point must contain a Point geometry.")

    _, wgs_window, acc_coef, ws_area = gauge_buffer_creator(
        gauge_geometry=gauge_geom,
        ws_gdf=create_gdf(ws_geom),
        tif_epsg=tif_epsg,
    )

    # Get relevant elevation tiles for the AOI
    elv_dir_tiff = roi_extent_tiles(topo_p=fdir_path, extent_coords=wgs_window)
    elv_tiles = elv_dir_tiff.get("elv")
    if not elv_tiles:
        raise FileNotFoundError("No elevation tiles found for the specified AOI.")

    # Create mosaic for AOI buffer with elevation data
    elv_vrt = create_mosaic(
        file_path=str(tmp_raster_folder),
        file_name=f"{gauge_id}_elv",
        tiles=elv_tiles,
    )

    # Clip the mosaic to the AOI extent
    trimmed_tiff = tif_trim_folder / f"{gauge_id}_elv.tiff"
    final_tiff = elv_path_dir / f"{gauge_id}.tiff"
    gdal_extent_clipper(
        initial_tiff=elv_vrt,
        extent=wgs_window,
        tmp_tiff=str(trimmed_tiff),
        final_tiff=str(final_tiff),
        crs_epsg=4326,
    )
    Path(elv_vrt).unlink(missing_ok=True)

    return acc_coef, ws_area
