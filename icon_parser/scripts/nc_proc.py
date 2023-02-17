from pathlib import Path
import xarray as xr
import glob
from datetime import datetime, timedelta


def from_icon_to_ru_nc(path_to_nc: Path, path_to_result: Path,
                       coord_limits: dict,
                       icon_time: str, var: str):
    """_summary_

    Args:
        path_to_nc (Path): _description_
        var (str): _description_
        icon_time (str): _description_
        path_to_result (Path): _description_
    """

    if var == 'tot_prec':
        ds_to_save = xr.open_mfdataset(
            glob.glob(
                f'{path_to_nc}/{icon_time}/{var}/*.nc')).resample(
            time='1d').sum()
    else:
        ds_to_save = xr.open_mfdataset(
            glob.glob(
                f'{path_to_nc}/{icon_time}/{var}/*.nc')).resample(
            time='1d').mean()

    ds_to_save = ds_to_save.where(
        ds_to_save.lon <= coord_limits['max_lon'], drop=True).where(
        ds_to_save.lon >= coord_limits['min_lon'], drop=True).where(
        ds_to_save.lat <= coord_limits['max_lat'], drop=True).where(
        ds_to_save.lat >= coord_limits['min_lat'], drop=True)

    today_date = datetime.today().strftime('%Y-%m-(%d')
    day_step = (datetime.today() + timedelta(7)).strftime('-%d)')

    forecast_period = today_date + day_step

    path_to_result = Path(f'{path_to_result}/{var}/')
    path_to_result.mkdir(exist_ok=True, parents=True)

    ds_to_save.to_netcdf(f'{path_to_result}/{var}_{forecast_period}.nc')

    return ds_to_save
