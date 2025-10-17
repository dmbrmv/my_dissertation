import geopandas as gpd

from src.utils.logger import setup_logger

loader_logger = setup_logger("geom_loader", log_file="logs/data_loader.log")


def load_geodata(
    folder_depth: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load watershed and gauge geometry data."""
    ws = gpd.read_file(
        f"{folder_depth}/data/geometry/russia_50k_ws.gpkg",
        ignore_geometry=False,
    )
    ws.set_index("gauge_id", inplace=True)
    gauges = gpd.read_file(
        f"{folder_depth}/data/geometry/russia_50k_gauges.gpkg",
        ignore_geometry=False,
    )
    gauges.set_index("gauge_id", inplace=True)
    return ws, gauges
