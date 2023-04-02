import xarray as xr
from pathlib import Path
import shutil
import glob


def file_checker(file_path: str,
                 meteo_predictors: list,
                 hydro_target: str,
                 possible_nans: int = 0):
    condition = xr.open_dataset(
        file_path).to_dataframe()[
            [hydro_target,
             *meteo_predictors]].isna().sum().sum() != possible_nans
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
