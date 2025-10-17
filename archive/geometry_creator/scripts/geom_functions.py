import numpy as np
from numpy import (arctan2, cos, sin, sqrt, pi, append, diff)
from shapely.geometry import MultiPolygon, Polygon
import geopandas as gpd


def find_float_len(number: float) -> int:

    return len(str(number).split('.')[1]) >= 2


def min_max_xy(shp_file):

    x_max, x_min = np.max(shp_file.exterior.xy[0]), np.min(
        shp_file.exterior.xy[0])

    y_max, y_min = np.max(shp_file.exterior.xy[1]), np.min(
        shp_file.exterior.xy[1])

    return (x_min, y_min, x_max, y_max)


def polygon_area(lats, lons, radius=6378137):
    """
    Computes area of spherical polygon, assuming spherical Earth.
    Returns result in ratio of the sphere's area if the radius is specified.
    Otherwise, in the units of provided radius.
    lats and lons are in degrees.
    """

    lats, lons = np.deg2rad(lats), np.deg2rad(lons)

    # Line integral based on Green's Theorem, assumes spherical Earth

    # close polygon
    if lats[0] != lats[-1]:
        lats = append(lats, lats[0])
        lons = append(lons, lons[0])

    # colatitudes relative to (0,0)
    a = sin(lats/2)**2 + cos(lats) * sin(lons/2)**2
    colat = 2*arctan2(sqrt(a), sqrt(1-a))

    # azimuths relative to (0,0)
    az = arctan2(cos(lats) * sin(lons), sin(lats)) % (2*pi)

    # Calculate diffs
    # daz = diff(az) % (2*pi)
    daz = diff(az)
    daz = (daz + pi) % (2 * pi) - pi

    deltas = diff(colat)/2
    colat = colat[0:-1]+deltas

    # Perform integral
    integrands = (1-cos(colat)) * daz

    # Integrate
    area = abs(sum(integrands))/(4*pi)

    area = min(area, 1-area)
    if radius is not None:  # return in units of radius
        return area * 4 * pi * radius**2 / 10**6
    else:  # return in ratio of sphere total area
        return area / 10**6


def select_big_from_MP(ws_geometry):
    if isinstance(ws_geometry, MultiPolygon):
        big_area = [polygon_area(lats=polygon.exterior.coords.xy[1],
                                 lons=polygon.exterior.coords.xy[0])
                    for polygon in ws_geometry.geoms]
        ws_geometry = ws_geometry.geoms[np.argmax(big_area)]
    else:
        ws_geometry = ws_geometry
    return ws_geometry


def ws_AOI(ws, shp_path: str):

    def my_ceil(a, precision=0):
        return np.true_divide(np.ceil(a * 10**precision), 10**precision)

    def my_floor(a, precision=0):
        return np.true_divide(np.floor(a * 10**precision), 10**precision)

    LONS, LATS = ws.exterior.xy
    max_LAT = np.max(LATS)
    max_LON = np.max(LONS)
    min_LAT = np.min(LATS)
    min_LON = np.min(LONS)

    min_LON, max_LON, min_LAT, max_LAT = (
        my_floor(min_LON, 2), my_ceil(max_LON, 2),
        my_floor(min_LAT, 2), my_ceil(max_LAT, 2))

    test = gpd.GeoDataFrame()
    aoi_geom = Polygon([[min_LON, min_LAT], [min_LON, max_LAT],
                        [max_LON, max_LAT], [max_LON, min_LAT]])
    test.loc[0, 'geometry'] = aoi_geom  # type: ignore
    test = test.set_crs(epsg=4326)

    test.to_file(f'{shp_path}', index=False)
    return shp_path


def round_up(x):
    return int(np.ceil(x / 5.0)) * 5


def round_down(x):
    return int(np.floor(x / 5.0)) * 5
