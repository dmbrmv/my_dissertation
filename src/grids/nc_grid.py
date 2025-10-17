"""Module for NetCDF grid processing and meteorological data extraction.

This module provides functions for:
- Extracting maximum longitude from NetCDF files
- Rounding coordinates to nearest grid resolution
- Finding spatial extents for watersheds
- Subsetting NetCDF data by spatial extent
- Calculating intersection weights for polygon-grid overlap
- Computing weighted meteorological aggregations
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
from numpy import dtype, float64
import pandas as pd
from shapely import from_wkb, to_wkb
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
import xarray as xr

from src.geometry.geom_processing import poly_from_multipoly, polygon_area
from src.grids.grid_edit import aggregation_definer, get_square_vertices
from src.utils.logger import setup_logger

logging = setup_logger("nc_grid", log_file="../../logs/nc_grid.log")


def get_max_lon_from_netcdf(nc_path: Path) -> float:
    """Extract the maximum longitude from a NetCDF file.

    Args:
        nc_path (Path): Path to the NetCDF file.

    Returns:
        float: The maximum longitude value in the NetCDF file.

    Raises:
        ValueError: If longitude variable is not found in the NetCDF file.

    """
    with xr.open_dataset(nc_path) as ds:
        # Try common longitude variable names
        for lon_name in ["lon", "longitude", "y"]:
            if lon_name in ds.variables:
                max_lon = float(ds[lon_name].max())
                return max_lon
    raise ValueError("longitude variable not found in NetCDF file.")


@lru_cache(maxsize=1024)
def round_nearest(value: float, step: float = 0.05, get_min: bool = False):
    """Round a value to the nearest multiple of a given step.

    Args:
        value (float): The value to round.
        step (float, optional): The step size to round to. Defaults to 0.05.
        get_min (bool, optional): If True, round down; otherwise, round up. Defaults to False.

    Returns:
        float: The rounded value.
    """
    q = Decimal(str(step))  # the quantum (0.01, 0.05, …)
    d_val = Decimal(str(value))

    lower = d_val.quantize(q, rounding=ROUND_DOWN)
    upper = d_val.quantize(q, rounding=ROUND_UP)
    if get_min:
        return float(lower)
    else:
        return float(upper)


def count_decimals(value) -> int:
    """Return the number of decimal places for a float or decimal.Decimal.

    Args:
        value: The number to check (float or decimal.Decimal).

    Returns:
        Number of decimal places as an integer.
    """
    if isinstance(value, float):
        # Convert to string, split at decimal point, and count digits after
        s = f"{value:.16f}".rstrip("0").rstrip(".")
        if "." in s:
            return len(s.split(".")[1])
        return 0
    elif hasattr(value, "as_tuple"):  # decimal.Decimal
        return abs(value.as_tuple().exponent)
    else:
        raise TypeError("Input must be a float or decimal.Decimal.")


def find_extent(
    ws: Polygon, grid_res: float
) -> np.ndarray[tuple[int, ...], dtype[float64]] | list[float]:
    """Find extent of watershed with given grid resolution.

    Args:
        ws: Watershed polygon
        grid_res: Grid resolution in decimal degrees

    Returns:
        List of four floats representing the extent [min_lon, max_lon, min_lat, max_lat]

    Raises:
        ValueError: If dataset is provided but not recognized
        AttributeError: If polygon doesn't have exterior coordinates

    """
    try:
        # Get bounds directly from polygon for better performance
        min_lon, min_lat, max_lon, max_lat = ws.bounds
    except AttributeError:
        raise AttributeError("Invalid polygon geometry - cannot extract bounds") from None
    true_grid_res = grid_res * 2

    # Round values to the nearest grid_res
    min_lon = round_nearest(min_lon, true_grid_res, get_min=True)
    max_lon = round_nearest(max_lon, true_grid_res, get_min=False)
    min_lat = round_nearest(min_lat, true_grid_res, get_min=True)
    max_lat = round_nearest(max_lat, true_grid_res, get_min=False)

    # Check if the extent is too small and adjust it
    lon_diff = abs(min_lon - max_lon)
    lat_diff = abs(min_lat - max_lat)

    if np.round(lon_diff, 3) <= true_grid_res:
        max_lon = round_nearest(max_lon + grid_res, true_grid_res, get_min=False)
        min_lon = round_nearest(min_lon - grid_res, true_grid_res, get_min=True)

    if np.round(lat_diff, 3) <= true_grid_res:
        max_lat = round_nearest(max_lat + grid_res, true_grid_res, get_min=False)
        min_lat = round_nearest(min_lat - grid_res, true_grid_res, get_min=True)

    min_lon = round(min_lon, count_decimals(grid_res))
    max_lon = round(max_lon, count_decimals(grid_res))
    min_lat = round(min_lat, count_decimals(grid_res))
    max_lat = round(max_lat, count_decimals(grid_res))

    return [min_lon, max_lon, min_lat, max_lat]


def nc_by_extent(nc: xr.Dataset, shape: Polygon | MultiPolygon, grid_res: float) -> xr.Dataset:
    """Select netCDF data by extent of given shape. Return masked netCDF.

    Args:
        nc: NetCDF dataset
        shape: Shape of the area of interest
        grid_res: Grid resolution in decimal degrees

    Returns:
        Masked netCDF dataset

    Raises:
        ValueError: If required dimensions are missing from dataset

    """
    # Standardize coordinate names more efficiently
    coord_mapping = {"latitude": "lat", "longitude": "lon"}

    # Only rename if necessary
    rename_dict = {old: new for old, new in coord_mapping.items() if old in nc.dims}
    if rename_dict:
        nc = nc.rename(rename_dict)

    # Validate required coordinates
    if "lat" not in nc.dims or "lon" not in nc.dims:
        raise ValueError("Dataset must contain 'lat' and 'lon' dimensions")

    # Find biggest polygon
    big_shape = poly_from_multipoly(ws_geom=shape)

    # Find extent coordinates
    min_lon, max_lon, min_lat, max_lat = find_extent(ws=big_shape, grid_res=grid_res)

    # More efficient subsetting using sel with slice
    try:
        masked_nc = nc.sel(
            lat=slice(max_lat, min_lat),
            lon=slice(min_lon, max_lon),
        )
    except (KeyError, ValueError):
        # Fallback to where method if slice doesn't work
        masked_nc = (
            nc.where(nc.lat >= min_lat, drop=True)
            .where(nc.lat <= max_lat, drop=True)
            .where(nc.lon >= min_lon, drop=True)
            .where(nc.lon <= max_lon, drop=True)
        )

    return masked_nc


# ---------------------- Глобалы для process-воркеров ----------------------
_WS_SHAPE: BaseGeometry | None = None
_WS_AREA: float | None = None
_GRID_RES: float | None = None


def _init_ws_globals(ws_wkb: bytes, ws_area: float, grid_res: float) -> None:
    """Инициализатор для ProcessPoolExecutor: один раз восстанавливает геометрию."""
    global _WS_SHAPE, _WS_AREA, _GRID_RES
    _WS_SHAPE = from_wkb(ws_wkb)
    _WS_AREA = ws_area
    _GRID_RES = grid_res


def _process_cell_proc(task: tuple[int, int, float, float]) -> tuple[int, int, bool, float]:
    """Процессный воркер: использует глобалы, заданные в initializer."""
    i, j, lat, lon = task
    try:
        # эти глобалы выставляются _init_ws_globals
        ws_shape = _WS_SHAPE
        ws_area = _WS_AREA
        grid_res = _GRID_RES
        if ws_shape is None or ws_area is None or grid_res is None:
            return i, j, False, 0.0

        cell = Polygon(get_square_vertices(mm=(lon, lat), h=grid_res, phi=0))
        inter = ws_shape.intersection(cell)
        if not inter.is_empty and ws_area > 0.0:
            w = polygon_area(poly_from_multipoly(inter)) / ws_area
            return i, j, True, float(w)
    except Exception as exc:  # pragma: no cover
        logging.exception(f"_process_cell_proc error @ lat={lat}, lon={lon}: {exc}")
    return i, j, False, 0.0


def _process_cell_thread(
    task: tuple[int, int, float, float, float, BaseGeometry, float],
) -> tuple[int, int, bool, float]:
    """Тредовый воркер: получает ссылку на геометрию в аргументах (без пиклинга)."""
    i, j, lat, lon, grid_res, ws_shape, ws_area = task
    try:
        cell = Polygon(get_square_vertices(mm=(lon, lat), h=grid_res, phi=0))
        inter = ws_shape.intersection(cell)
        if not inter.is_empty and ws_area > 0.0:
            w = polygon_area(poly_from_multipoly(inter)) / ws_area
            return i, j, True, float(w)
    except Exception as exc:  # pragma: no cover
        logging.exception(f"_process_cell_thread error @ lat={lat}, lon={lon}: {exc}")
    return i, j, False, 0.0


# ---------------------------- Основная функция ----------------------------


def get_weights(
    weight_path: Path,
    mask_nc: xr.Dataset,
    ws_geom: Polygon,
    ws_area: float,
    grid_res: float = 0.05,
    n_workers: int | None = None,
    parallel: str = "thread",  # "none" | "thread" | "process"
) -> xr.DataArray:
    """Вычисляет веса пересечения полигон(ов) водосбора с регулярной решёткой.

    Параметры
    ---------
    weight_path : Path
        Путь к .nc файлу с кэшем весов (создаётся при отсутствии).
    mask_nc : xr.Dataset
        Dataset с координатами 'lat' и 'lon' (в градусах).
    ws_geom : Polygon
        Геометрия водосбора (MultiPolygon допустим — будет сведён к Polygon).
    ws_area : float
        Площадь водосбора (в тех же единицах, что и polygon_area).
    grid_res : float
        Шаг сетки (в градусах).
    n_workers : int | None
        Количество воркеров. По умолчанию: max(1, cpu_count - 1) для thread/process.
    parallel : {"none", "thread", "process"}
        Режим параллелизма внутри функции. По умолчанию "thread" (без nested-processes).

    Возвращает
    ----------
    xr.DataArray [lat, lon]
        Веса (доли площади ячеек, принадлежащей водосбору). Вне пересечения — 0.
    """
    # --- кэш ---
    if weight_path.is_file():
        # загружаем в память, чтобы не держать открытым файл
        return xr.load_dataarray(weight_path)

    # --- подготовка ---
    ws_shape: BaseGeometry = poly_from_multipoly(ws_geom)
    nc_lat = np.asarray(mask_nc["lat"].values)
    nc_lon = np.asarray(mask_nc["lon"].values)

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    # ограничимся bbox водосбора
    minx, miny, maxx, maxy = ws_shape.bounds  # x=lon, y=lat
    lat_lo = miny - grid_res
    lat_hi = maxy + grid_res
    lon_lo = minx - grid_res
    lon_hi = maxx + grid_res

    lat_idx = np.where((nc_lat >= lat_lo) & (nc_lat <= lat_hi))[0]
    lon_idx = np.where((nc_lon >= lon_lo) & (nc_lon <= lon_hi))[0]

    # если ничего не попало — вернуть нули с правильными координатами
    if lat_idx.size == 0 or lon_idx.size == 0:
        empty = xr.DataArray(
            data=np.zeros((0, 0), dtype=np.float32),
            dims=["lat", "lon"],
            coords={"lat": nc_lat[:0], "lon": nc_lon[:0]},
            name="weights",
        )
        empty.to_netcdf(weight_path)
        return empty

    # создаём контейнеры под итог
    weights_data = np.zeros((nc_lat.size, nc_lon.size), dtype=np.float32)
    inter_mask = np.zeros((nc_lat.size, nc_lon.size), dtype=bool)

    # формируем задачи только в подокне bbox
    if parallel == "process":
        # минимальные payload'ы на таск
        tasks = [(i, j, float(nc_lat[i]), float(nc_lon[j])) for i in lat_idx for j in lon_idx]
        ws_wkb = to_wkb(ws_shape)

        # Для надёжности избегаем nested-processes: используйте этот режим,
        # если get_weights вызывается из главного процесса.
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_ws_globals,
            initargs=(ws_wkb, float(ws_area), float(grid_res)),
        ) as ex:
            futures = [ex.submit(_process_cell_proc, t) for t in tasks]
            for fut in as_completed(futures):
                i, j, ok, w = fut.result()
                if ok:
                    inter_mask[i, j] = True
                    weights_data[i, j] = w

    elif parallel == "thread":
        # Передаём ссылку на ws_shape прямо в аргументах (без сериализации)
        tasks = [
            (i, j, float(nc_lat[i]), float(nc_lon[j]), float(grid_res), ws_shape, float(ws_area))
            for i in lat_idx
            for j in lon_idx
        ]
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_process_cell_thread, t) for t in tasks]
            for fut in as_completed(futures):
                i, j, ok, w = fut.result()
                if ok:
                    inter_mask[i, j] = True
                    weights_data[i, j] = w

    else:  # "none" — последовательный режим (надёжно при nested-processes)
        for i in lat_idx:
            for j in lon_idx:
                _, _, ok, w = _process_cell_thread(
                    (i, j, float(nc_lat[i]), float(nc_lon[j]), float(grid_res), ws_shape, float(ws_area))
                )
                if ok:
                    inter_mask[i, j] = True
                    weights_data[i, j] = w

    # собираем DataArray, обрезаем по маске и сохраняем
    inter_da = xr.DataArray(inter_mask, dims=["lat", "lon"], coords=[nc_lat, nc_lon])
    weights = xr.DataArray(weights_data, dims=["lat", "lon"], coords=[nc_lat, nc_lon], name="weights")
    weights = weights.where(inter_da, drop=True).fillna(0.0)

    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weights.to_netcdf(weight_path)
    return weights


def calculate_weighted_meteorology(
    ws_nc: xr.Dataset,
    weights: xr.DataArray,
    magnitude_factor: float = 1e2,
) -> pd.DataFrame:
    """Calculate weighted meteorological data from NetCDF dataset.

    Args:
        ws_nc: NetCDF dataset with meteorological variables
        weights: DataArray with spatial weights for aggregation
        magnitude_factor: Factor to scale the results (default: 1e2)

    Returns:
        DataFrame with weighted meteorological data indexed by date

    """
    time_coord = next(
        (coord for coord in ws_nc.coords if np.issubdtype(ws_nc[coord].dtype, np.datetime64)), "time"
    )

    # Create weighted dataset for all variables at once
    weighted_ds = ws_nc.weighted(weights)

    # Calculate aggregated values for all variables efficiently
    aggregated_data = {}
    for var in ws_nc.data_vars:
        agg_method = aggregation_definer(var)
        if agg_method == "sum":
            aggregated_data[var] = (weighted_ds.sum(dim=["lat", "lon"])[var] * magnitude_factor).values

        else:
            aggregated_data[var] = (weighted_ds.mean(dim=["lat", "lon"])[var]).values
    # Create DataFrame efficiently
    result_df = pd.DataFrame({"date": ws_nc[time_coord].values, **aggregated_data}).set_index("date")
    # Clip precipitation if requested
    if "prcp" in result_df.columns:
        result_df["prcp"] = result_df["prcp"].clip(lower=0)
    return result_df
