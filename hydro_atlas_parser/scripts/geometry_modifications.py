import numpy as np
import geopandas as gpd
from numpy import arctan2, cos, sin, sqrt, pi, append, diff, deg2rad
from shapely.geometry import Polygon, MultiPolygon


def polygon_area(lats: np.ndarray, lons: np.ndarray, radius=6378137):
    """
    Computes area of spherical polygon, assuming spherical Earth.
    Returns result in ratio of the sphere's area if the radius is specified.
    Otherwise, in the units of provided radius.
    lats and lons are in degrees.

    Args:
        lats (list): list of latitudinal coordinates
        lons (list): list of longitudinal coordinates
        radius (int, optional): Earth radius. Defaults to 6378137.

    Returns:
        area: area of object in sq. km
    """

    lats, lons = deg2rad(lats), deg2rad(lons)

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


def select_big_from_mp(WS_geometry):
    """
    Select biggest polygon from MultiPolygon object.
    Needs to eliminate fake structures out of basin shape

    Args:
        WS_geometry (Geometry): Desired basin shape

    Returns:
        Ws_geometry: Biggest polygon which correspond to natural form
    """
    if type(WS_geometry) == MultiPolygon:
        big_area = [polygon_area(lats=polygon.exterior.coords.xy[1],
                                 lons=polygon.exterior.coords.xy[0])
                    for polygon in WS_geometry]
        WS_geometry = WS_geometry[np.argmax(big_area)]
    else:
        WS_geometry = WS_geometry
    return WS_geometry


def get_gdf_poly(some_geometry: Polygon):
    """
    Transform Polygon object to GeoDataFrame with EPSG projection

    Args:
        some_geometry (Polygon): Polygon from GeoDataFrame of Polygon's

    Returns:
        target (GeoDataFrame): Polygon stored in GeoDataFrame format
    """
    target = gpd.GeoSeries(
        select_big_from_mp(some_geometry))
    target = gpd.GeoDataFrame({'geometry': target}).set_crs('EPSG:4326')

    return target


def find_poly_area(poly):
    """
    Calculates area of shape stored in GeoDataFrame

    Args:
        poly (GeoDataFrame): Desired shape

    Returns:
        area (float): area of object in sq. km
    """
    if poly.empty:
        return np.NaN
    else:
        poly = select_big_from_mp(poly['geometry'][0])
        area = polygon_area(lats=poly.exterior.xy[1],
                            lons=poly.exterior.xy[0])
    return area


def parallelize_function(WS: gpd.GeoDataFrame, path_to_HydroATLAS: str):
    """
    This function generate list of tuples
    where each tuple stands for row in DF
    of watersheds

    Args:
        WS (GeoDataFrame): GeoDataFrame of desired watersheds
        path_to_HydroATLAS (str): path to gdb of HydroATLAS on disk
        layer_small (fiona): [description]

    Returns:
        mp_tuples (tuple): tuple of values which will be required for 
        parallel launch
    """
    mp_tuples = list()

    for row in range(len(WS)):
        mp_tuples.append((WS.loc[row, 'geometry'],
                          path_to_HydroATLAS))

    return mp_tuples
