from typing import Hashable, Tuple, Union
import xarray as xr
from shapely.geometry import Polygon

from .geom_functions import (poly_from_multipoly, find_extent)


# !TODO
# Rewrite with dependance on aggregation parameter
def read_netcdf(net_cdf_on_disk: Union[str, list]) -> Union[
    Tuple[Hashable,
          xr.DataArray, xr.DataArray,
          xr.DataArray],
    Tuple[Hashable,
          None, None,
          xr.DataArray]]:
    """_summary_

    Args:
        net_cdf_on_disk (_type_): _description_

    Returns:
        _type_: _description_
    """

    dataset = xr.open_mfdataset(net_cdf_on_disk,
                                chunks='auto')

    variable_name = list(dataset.variables)[-1]

    varible_data = dataset[variable_name]
    dataset.close()
    # calculate daily values
    if variable_name in ['t2m']:
        varible_data_mean = varible_data.resample(time='1D').mean()
        variable_data_min = varible_data.resample(time='1D').min()
        variable_data_max = varible_data.resample(time='1D').max()
        return (variable_name,
                variable_data_min, variable_data_max,
                varible_data_mean)
    elif variable_name in ['tp']:
        varible_data_sum = varible_data.resample(time='1D').sum()
        return (variable_name,
                None, None,
                varible_data_sum)
    else:
        varible_data_mean = varible_data.resample(time='1D').mean()
        return (variable_name,
                None, None,
                varible_data_mean)


def nc_by_extent(nc: xr.Dataset,
                 shape: Polygon,
                 grid_res: float,
                 dataset: str = ''):
    """

    select net_cdf by extent of given shape

    return masked net_cdf

    """
    if 'latitude' in nc.dims:
        nc = nc.rename({'latitude': 'lat', 'longitude': 'lon'})
    else:
        pass

    # find biggest polygon
    big_shape = poly_from_multipoly(ws_geom=shape)

    # find extent coordinates
    min_lon, max_lon, min_lat, max_lat = find_extent(ws=big_shape,
                                                     grid_res=grid_res,
                                                     dataset=dataset)

    # select nc inside of extent
    masked_nc = nc.where(
        nc.lat >= min_lat, drop=True).where(
        nc.lat <= max_lat, drop=True).where(
        nc.lon >= min_lon, drop=True).where(
        nc.lon <= max_lon, drop=True)
    masked_nc = masked_nc.chunk(chunks='auto')
    return masked_nc
