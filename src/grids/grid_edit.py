"""Module for grid-polygon intersection calculations and weight computation.

This module provides functions for:
- Rotation matrix calculations
- Square vertex generation for grid cells
- Polygon-grid intersection processing with multiprocessing support
- Weight calculation for meteorological data aggregation
- Aggregation method determination based on variable types
"""

from collections.abc import Generator, Hashable
from functools import reduce

import geopandas as gpd
import numpy as np
import pandas as pd

from src.geometry.geom_processing import create_gdf


def rot_m(alpha: float) -> np.ndarray:
    """Rotation Matrix for angle ``alpha``."""
    sa, ca = np.sin(alpha), np.cos(alpha)
    return np.array([[ca, -sa], [sa, ca]])


def get_square_vertices(mm: tuple[float, float], h: float, phi: float) -> np.ndarray:
    """Calculate the four vertices for square with center ``mm``.

    Args:
        mm: Center coordinates (x, y) of the square
        h: Side length of the square
        phi: Rotation angle in radians

    Returns:
        Array of four vertex coordinates
    """
    hh0 = np.ones(2) * h  # initial corner
    vv = [
        np.asarray(mm) + reduce(np.dot, [rot_m(phi), rot_m(np.pi / 2 * c), hh0]) for c in range(4)
    ]  # rotate initial corner four times by 90°
    return np.asarray(vv)


def make_intersected_generator(
    inter: list[gpd.GeoDataFrame],
) -> Generator[pd.DataFrame | gpd.GeoDataFrame, None, None]:
    """Create a generator that returns a geodataframe.

    For each section in the passed list of intersections (inter).
    This geodataframe is empty if the section is empty
    contains the geometry of the largest feature in that section otherwise.

    Args:
    ----
        inter (list[gpd.GeoDataFrame]) : list with geoDataFrames

    Returns:
    -------
        Generator[gpd.GeoDataFrame]: Generator yielding geodataframes

    """
    return (
        create_gdf(section.loc[0, "geometry"])
        if len(section) != 0 and hasattr(section.loc[0, "geometry"], "exterior")
        else gpd.GeoDataFrame()
        for section in inter
    )


def aggregation_definer(variable: str | Hashable) -> str:
    """Determine the aggregation method for a given variable in a meteorological dataset.

    Args:
        variable: Name of the meteorological variable to aggregate.

    Returns:
        The aggregation method: 'sum' for precipitation/evaporation variables, 'mean' for others.

    Examples:
        >>> aggregation_definer("prcp")
        'sum'
        >>> aggregation_definer("t_min")
        'mean'

    """
    if isinstance(variable, Hashable):
        variable = str(variable)
    # Define precipitation-related keywords for efficient lookup
    precipitation_keywords = frozenset(["precipitation", "evaporation", "tot_prec", "pr", "prcp", "rr"])

    # Convert variable to lowercase once for case-insensitive comparison
    variable_lower = variable.lower()

    # Check if any precipitation keyword is in the variable name
    return "sum" if any(keyword in variable_lower for keyword in precipitation_keywords) else "mean"
