import geopandas as gpd
import numpy as np
from numpy import sin
from shapely.geometry import MultiPolygon, Point, Polygon


def filter_watersheds_by_lon(ws_gdf: gpd.GeoDataFrame, max_lon: float) -> gpd.GeoDataFrame:
    """Filter watersheds whose maximum longitude is less than or equal to max_lon.

    Args:
        ws_gdf (gpd.GeoDataFrame): GeoDataFrame of watersheds.
        max_lon (float): Maximum usable longitude.

    Returns:
        gpd.GeoDataFrame: Filtered GeoDataFrame with watersheds fully within max_lon.
    """
    if "geometry" not in ws_gdf:
        raise ValueError("Input GeoDataFrame must contain a 'geometry' column.")

    ws_gdf = ws_gdf.copy()
    ws_gdf["max_lon"] = ws_gdf.geometry.apply(
        lambda geom: geom.bounds[2] if geom and not geom.is_empty else float("-inf")
    )
    filtered = ws_gdf[ws_gdf["max_lon"] <= max_lon]
    return filtered


def polygon_area(geo_shape: Polygon, radius: float = 6378137.0) -> float:
    """Calculate the area of a polygon on a sphere using the spherical excess formula.

    Args:
        geo_shape (Polygon): The polygon whose area is to be computed.
        radius (float, optional): The radius of the sphere in meters. Defaults to 6378137.0.

    Returns:
        float: The area of the polygon in square kilometers.

    Raises:
        ValueError: If geo_shape is not a valid Polygon.
    """
    if not isinstance(geo_shape, Polygon):
        raise ValueError("Input geometry must be a shapely Polygon.")

    coords = np.asarray(geo_shape.exterior.coords)
    if coords.shape[0] < 3:
        return 0.0

    lons_rad = np.radians(coords[:-1, 0])
    lats_rad = np.radians(coords[:-1, 1])

    # Ensure polygon is closed
    if not np.allclose(coords[0], coords[-1]):
        lons_rad = np.append(lons_rad, lons_rad[0])
        lats_rad = np.append(lats_rad, lats_rad[0])

    lons_next = np.roll(lons_rad, -1)
    lats_next = np.roll(lats_rad, -1)

    dlon = lons_next - lons_rad
    lat_terms = 2 + sin(lats_rad) + sin(lats_next)

    area = abs(np.sum(dlon * lat_terms)) * (radius**2) * 0.5
    return area * 1e-6  # Convert to square kilometers


def poly_from_multipoly(ws_geom: Polygon | MultiPolygon) -> Polygon:
    """Return the largest polygon from a MultiPolygon or the input Polygon.

    Args:
        ws_geom (Polygon | MultiPolygon): The geometry of the watershed.

    Returns:
        Polygon: The largest polygon in the watershed.

    Raises:
        ValueError: If ws_geom is not a Polygon or MultiPolygon.
    """
    if isinstance(ws_geom, MultiPolygon):
        if not ws_geom.geoms:
            raise ValueError("MultiPolygon contains no geometries.")
        areas = [polygon.area for polygon in ws_geom.geoms]
        max_index = int(np.argmax(areas))
        return ws_geom.geoms[max_index]
    elif isinstance(ws_geom, Polygon):
        return ws_geom
    else:
        raise ValueError("Input geometry must be a Polygon or MultiPolygon.")


def create_gdf(shape: Polygon | MultiPolygon) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame with the largest polygon from the input shape.

    Args:
        shape (Polygon | MultiPolygon): The input geometry.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with a single geometry.
    """
    largest_poly = poly_from_multipoly(shape)
    gdf = gpd.GeoDataFrame(geometry=[largest_poly], crs="EPSG:4326")
    return gdf


def area_from_gdf(poly: gpd.GeoDataFrame) -> float:
    """Calculate the area of a shape stored in a GeoDataFrame.

    Args:
        poly (gpd.GeoDataFrame): The GeoDataFrame containing the geometry.

    Returns:
        float: The area of the object in square kilometers.

    Raises:
        ValueError: If the GeoDataFrame does not contain a valid geometry.
    """
    if poly.empty or "geometry" not in poly or poly.geometry.is_empty.any():
        return np.nan
    shape = poly_from_multipoly(poly.geometry.iloc[0])
    return polygon_area(shape)


def gauge_buffer_creator(
    gauge_geometry: Point, ws_gdf: gpd.GeoSeries | gpd.GeoDataFrame, tif_epsg: int
) -> tuple[gpd.GeoDataFrame, tuple[float, float, float, float], float, float]:
    """Create a square buffer for flood modelling extent around a gauge point.

    Args:
        gauge_geometry (Point): Shapely Point object from geometry column.
        ws_gdf (gpd.GeoSeries | gpd.GeoDataFrame): Watershed geometry for the gauge.
        tif_epsg (int): Metric EPSG code for area calculation.

    Returns:
        tuple: (
            buffer_gdf (gpd.GeoDataFrame): Buffer for river intersection search,
            wgs_window (tuple[float, float, float, float]): Extent coordinates (minx, maxy, maxx, miny),
            acc_coef (float): Number of 90m cells in the watershed,
            ws_area (float): Watershed area in sq. km
        )

    Raises:
        ValueError: If input geometry is invalid.
    """
    if not isinstance(gauge_geometry, Point):
        raise ValueError("gauge_geometry must be a shapely Point.")
    if ws_gdf.empty or "geometry" not in ws_gdf:
        raise ValueError(
            "ws_gdf must be a non-empty GeoSeries or GeoDataFrame with a 'geometry' column."
        )

    # Calculate watershed area in square kilometers
    ws_area = ws_gdf.to_crs(epsg=tif_epsg).geometry.area.iloc[0] * 1e-6

    # Determine buffer size in degrees based on area
    if ws_area > 500_000:
        area_size = 0.30
    elif ws_area > 50_000:
        area_size = 0.20
    elif ws_area > 5_000:
        area_size = 0.10
    else:
        area_size = 0.05

    # Calculate number of 90m cells in the watershed
    acc_coef = (ws_area * 1e6) / (90 * 90)

    # Create square buffer for extent and for river intersection search
    buffer = gauge_geometry.buffer(area_size, cap_style="square")
    buffer_isc = gauge_geometry.buffer(area_size - 0.015, cap_style="square")

    # Create GeoDataFrame for buffer_isc
    buffer_gdf = gpd.GeoDataFrame(geometry=[buffer_isc], crs="EPSG:4326")

    # Get bounds: (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = buffer.bounds
    # Return as (minx, maxy, maxx, miny) for wgs_window
    wgs_window = (minx, maxy, maxx, miny)

    return buffer_gdf, wgs_window, acc_coef, ws_area


def gauge_to_utm(
    gauge_series: gpd.GeoSeries, return_gdf: bool = False
) -> tuple[gpd.GeoDataFrame, int] | Point | None:
    """Project a gauge geometry from WGS84 to its appropriate UTM zone.

    Args:
        gauge_series (gpd.GeoSeries): GeoSeries containing geometry in WGS84 (EPSG:4326).
        return_gdf (bool, optional): If True, return a tuple of (projected GeoDataFrame, EPSG code).
            If False, return the projected Point geometry. Defaults to False.

    Returns:
        tuple[gpd.GeoDataFrame, int] | Point | None: Projected geometry and EPSG code if return_gdf is True,
            otherwise the projected Point. Returns None if input is empty.

    Raises:
        ValueError: If gauge_series is empty or does not contain valid geometry.
    """
    if gauge_series.empty or not hasattr(gauge_series, "geometry"):
        raise ValueError("Input gauge_series must be a non-empty GeoSeries with a 'geometry' attribute.")

    # Ensure input is a GeoDataFrame with correct CRS
    gdf = gpd.GeoDataFrame(geometry=gauge_series, crs="EPSG:4326")

    # Estimate UTM CRS and EPSG code
    utm_crs = gdf.estimate_utm_crs()
    utm_epsg = utm_crs.to_epsg() if utm_crs is not None else None
    if utm_epsg is None:
        raise ValueError("Unable to estimate UTM CRS for the provided geometry.")

    # Project to UTM CRS
    gdf_utm = gdf.to_crs(epsg=utm_epsg)

    if return_gdf:
        return gdf_utm, utm_epsg

    # Return the projected Point geometry
    geom = gdf_utm.geometry.iloc[0]
    if not isinstance(geom, Point):
        raise ValueError("Projected geometry is not a Point.")
    return geom
