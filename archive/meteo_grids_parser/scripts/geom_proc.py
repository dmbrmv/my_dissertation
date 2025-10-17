import math
from functools import reduce
from typing import Union

import geopandas as gpd
import numpy as np
from numpy import append, arctan2, cos, diff, pi, sin, sqrt
from shapely.geometry import MultiPolygon, Polygon


def area_from_gdf(poly):
    """Calculates area of shape stored in GeoDataFrame

    Args:
        poly (GeoDataFrame): Desired shape

    Returns:
        area (float): area of object in sq. km

    """
    if poly.empty:
        return np.nan
    else:
        poly = poly_from_multipoly(poly["geometry"][0])
        area = polygon_area(poly)
    return area


def poly_from_multipoly(ws_geom):
    """Function return only biggest polygon
    from multipolygon WS
    It's the real WS, and not malfunctioned part of it

    """
    if type(ws_geom) == MultiPolygon:
        big_area = [polygon_area(geo_shape=polygon) for polygon in ws_geom.geoms]
        ws_geom = ws_geom.geoms[np.argmax(big_area)]
    else:
        ws_geom = ws_geom
    return ws_geom


def polygon_area(geo_shape, radius=6378137):
    """Computes area of spherical polygon, assuming spherical Earth
    Returns result in ratio of the sphere's area if the radius is specified.
    Otherwise, in the units of provided radius.
    lats and lons are in degrees.
    """
    lons, lats = geo_shape.exterior.xy
    lats, lons = np.deg2rad(lats), np.deg2rad(lons)
    # Line integral based on Green's Theorem, assumes spherical Earth

    # close polygon
    if lats[0] != lats[-1]:
        lats = append(lats, lats[0])
        lons = append(lons, lons[0])

    # colatitudes relative to (0,0)
    a = sin(lats / 2) ** 2 + cos(lats) * sin(lons / 2) ** 2
    colat = 2 * arctan2(sqrt(a), sqrt(1 - a))

    # azimuths relative to (0,0)
    az = arctan2(cos(lats) * sin(lons), sin(lats)) % (2 * pi)

    # Calculate diffs
    # daz = diff(az) % (2*pi)
    daz = diff(az)
    daz = (daz + pi) % (2 * pi) - pi

    deltas = diff(colat) / 2
    colat = colat[0:-1] + deltas

    # Perform integral
    integrands = (1 - cos(colat)) * daz

    # Integrate
    area = abs(sum(integrands)) / (4 * pi)

    area = min(area, 1 - area)
    if radius is not None:  # return in units of radius
        return area * 4 * pi * radius**2 / 10**6
    else:  # return in ratio of sphere total area
        return area / 10**6


def find_extent(ws: Polygon, grid_res: float, dataset: str = ""):
    """_summary_

    Args:
        ws (Polygon): _description_
        grid_res (float): _description_

    """

    def x_round(x):
        return round((x - 0.25) * 2) / 2 + 0.25

    def round_nearest(x, a):
        max_frac_digits = 10
        for i in range(max_frac_digits):
            if round(a, -int(math.floor(math.log10(a))) + i) == a:
                frac_digits = -int(math.floor(math.log10(a))) + i
                break
        return round(round(x / a) * a, frac_digits)  # type: ignore

    lons, lats = ws.exterior.xy  # type: ignore
    max_LAT = max(lats)
    max_LON = max(lons)
    min_LAT = min(lats)
    min_LON = min(lons)

    if dataset == "gpcp":
        return [x_round(min_LON), x_round(max_LON), x_round(min_LAT), x_round(max_LAT)]
    elif bool(dataset):
        # find extent coordinates
        min_lon, max_lon, min_lat, max_lat = (
            round_nearest(min_LON, grid_res),
            round_nearest(max_LON, grid_res),
            round_nearest(min_LAT, grid_res),
            round_nearest(max_LAT, grid_res),
        )
        # if max_lat == min_lat:
        #     max_lat += grid_res*2
        # if max_lon == min_lon:
        #     max_lon += grid_res*2
        return [min_lon, max_lon, min_lat, max_lat]
    else:
        raise Exception(f"Something wrong ! {dataset} -- {grid_res}")


def create_gdf(shape: Union[Polygon, MultiPolygon]) -> gpd.GeoDataFrame:
    """Create geodataframe with given shape
    as a geometry

    """
    gdf_your_WS = poly_from_multipoly(ws_geom=shape)
    # WS from your data
    gdf_your_WS = gpd.GeoSeries(data=[gdf_your_WS])

    # Create extra gdf to use geopandas functions
    gdf_your_WS = gpd.GeoDataFrame(data={"geometry": gdf_your_WS})
    gdf_your_WS = gdf_your_WS.set_crs("EPSG:4326")

    return gdf_your_WS


def RotM(alpha):
    """Rotation Matrix for angle ``alpha``"""
    sa, ca = np.sin(alpha), np.cos(alpha)
    return np.array([[ca, -sa], [sa, ca]])


def getSquareVertices(mm, h, phi):
    """Calculate the for vertices for square with center ``mm``,
    side length ``h`` and rotation ``phi``
    """
    hh0 = np.ones(2) * h  # initial corner
    vv = [
        np.asarray(mm) + reduce(np.dot, [RotM(phi), RotM(np.pi / 2 * c), hh0]) for c in range(4)
    ]  # rotate initial corner four times by 90°
    return np.asarray(vv)
