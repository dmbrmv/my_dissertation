import xarray as xr
from pathlib import Path
import shutil
import glob


def file_checker(file_path: str,
                 meteo_predictors: list,
                 hydro_target: str,
                 possible_nans: int = 0):

    ds_file = xr.open_dataset(file_path).to_dataframe()[
        [hydro_target, *meteo_predictors]]

    condition_1 = ds_file.isna().sum().sum() != possible_nans
    condition_2 = max([max(value/ds_file[hydro_target].to_numpy())
                       for value in ds_file[hydro_target].to_numpy()]) > 1500
    condition = condition_1 | condition_2

    return condition


def file_rewriter(q_pathes: list,
                  ts_dir: Path,
                  hydro_target: str,
                  meteo_predictors: list,
                  possible_nans: int = 0):
    shutil.rmtree(ts_dir)
    ts_dir.mkdir(exist_ok=True, parents=True)

    for file in q_pathes:
        cond = file_checker(file_path=file,
                            hydro_target=hydro_target,
                            meteo_predictors=meteo_predictors,
                            possible_nans=possible_nans)
        if cond:
            continue
        else:
            ds = xr.open_dataset(file)
            filename = file.split('/')[-1]
            try:
                ds = ds.drop('gauge_id')
                ds.to_netcdf(f'{ts_dir}/{filename}')
            except ValueError:
                continue
    basins = [file.split('/')[-1][:-3] for
              file in glob.glob(f'{ts_dir}/*.nc')]
    with open('./openf_basins.txt', 'w') as the_file:
        for gauge_name in basins:
            the_file.write(f'{int(gauge_name)}\n')
