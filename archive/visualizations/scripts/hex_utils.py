"""Utility helpers for working with hexagon-based aggregations."""

from __future__ import annotations

import math
from typing import Literal

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

EQ_AREA_CRS = "EPSG:6933"


def to_equal_area(gdf: gpd.GeoDataFrame, crs: str = EQ_AREA_CRS) -> gpd.GeoDataFrame:
    """Project a GeoDataFrame to the supplied equal-area CRS for consistent area metrics."""
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame must have a CRS.")
    return gdf.to_crs(crs)


def _hexagon(cx: float, cy: float, r: float) -> Polygon:
    """Build a pointy-top hexagon centered on (cx, cy) with circumradius ``r``."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    x = cx + r * np.sin(angles)
    y = cy + r * np.cos(angles)
    return Polygon(np.c_[x, y])


def build_hex_grid(
    extent_poly: Polygon, r_km: float = 75.0, crs: str = EQ_AREA_CRS
) -> gpd.GeoDataFrame:
    """Create a pointy-top hex grid covering ``extent_poly`` in the supplied CRS."""
    r = r_km * 1_000.0
    w = math.sqrt(3) * r
    h = 2 * r
    dx = w
    dy = 1.5 * r

    minx, miny, maxx, maxy = extent_poly.bounds
    minx -= w
    miny -= h
    maxx += w
    maxy += h

    hexes = []
    row = 0
    y = miny
    while y <= maxy:
        x_offset = 0 if row % 2 == 0 else dx / 2.0
        x = minx + x_offset
        while x <= maxx:
            hpoly = _hexagon(x, y, r)
            if hpoly.intersects(extent_poly):
                hexes.append(hpoly)
            x += dx
        y += dy
        row += 1

    return gpd.GeoDataFrame({"geometry": hexes}, crs=crs)


def aggregate_nse_to_hex(
    watersheds: gpd.GeoDataFrame,
    hexes: gpd.GeoDataFrame,
    nse_col: str = "NSE",
    agg: Literal["median", "mean", "max", "min"] = "median",
    area_weighted: bool = False,
    min_overlap_share: float = 0.1,
) -> gpd.GeoDataFrame:
    """Aggregate watershed metric values to hex cells using centroid or area weighting."""
    if watersheds.crs != hexes.crs:
        raise ValueError("CRS mismatch. Reproject both to the same equal-area CRS.")

    w_cent = watersheds.copy()
    w_cent["geometry"] = w_cent.geometry.centroid

    joined = gpd.sjoin(
        w_cent[[nse_col, "geometry"]],
        hexes.reset_index(names="hex_id"),
        how="inner",
        predicate="within",
    )

    if joined.empty:
        return hexes.assign(count=0, **{f"{agg}_{nse_col}": np.nan})

    if not area_weighted:
        grouped = joined.groupby("hex_id")[nse_col]
        val = getattr(grouped, agg)()
        cnt = grouped.size()
    else:
        inter = gpd.overlay(
            watersheds[[nse_col, "geometry", "orig_area"]].copy(),
            hexes.reset_index(names="hex_id"),
            how="intersection",
        )
        inter["a"] = inter.area
        inter = inter[inter["a"] >= min_overlap_share * inter["orig_area"]]
        grouped = inter.groupby("hex_id")
        val = grouped.apply(lambda df: np.average(df[nse_col], weights=df["a"]))
        cnt = grouped.size()

    out = hexes.reset_index(names="hex_id").merge(
        val.rename(f"{agg}_{nse_col}"), on="hex_id", how="left"
    )
    out["count"] = out["hex_id"].map(cnt).fillna(0).astype(int)
    return out.loc[out["count"] > 0].copy()


def suggest_hex_radius(
    watersheds: gpd.GeoDataFrame,
    target_ws_per_hex: float = 4.0,
    quantile: float = 0.5,
    min_r_km: float = 20.0,
    max_r_km: float = 120.0,
    crs: str = EQ_AREA_CRS,
) -> float:
    """Recommend a hex radius (km) from watershed areas with clipping bounds."""
    if target_ws_per_hex <= 0:
        raise ValueError("target_ws_per_hex must be positive.")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1].")

    watersheds_eq = to_equal_area(watersheds, crs=crs)
    areas_km2 = watersheds_eq.geometry.area / 1_000_000.0
    if areas_km2.empty:
        raise ValueError("Watersheds GeoDataFrame has no geometries.")

    representative_area = np.quantile(areas_km2, quantile)
    if representative_area <= 0:
        raise ValueError("Representative watershed area must be positive.")

    target_hex_area_km2 = representative_area * target_ws_per_hex
    r_km = math.sqrt(target_hex_area_km2 / 2.598)
    return float(np.clip(r_km, min_r_km, max_r_km))


def summarize_hex_coverage(
    hexes: gpd.GeoDataFrame, count_col: str = "count"
) -> dict[str, float]:
    """Return descriptive stats for how many watersheds fall into each hex."""
    if count_col not in hexes.columns:
        raise KeyError(f"Column '{count_col}' not found in hex GeoDataFrame.")

    counts = hexes[count_col].astype(float)
    if counts.empty:
        raise ValueError("Hex GeoDataFrame contains no rows to summarise.")

    return {
        "total_hexes": float(len(counts)),
        "total_watersheds": float(counts.sum()),
        "mean_count": float(counts.mean()),
        "median_count": float(counts.median()),
        "p10_count": float(counts.quantile(0.1)),
        "p90_count": float(counts.quantile(0.9)),
        "min_count": float(counts.min()),
        "max_count": float(counts.max()),
    }


__all__ = [
    "EQ_AREA_CRS",
    "aggregate_nse_to_hex",
    "build_hex_grid",
    "summarize_hex_coverage",
    "suggest_hex_radius",
    "to_equal_area",
]
