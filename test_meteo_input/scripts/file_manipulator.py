import xarray as xr
from pathlib import Path
import shutil
import glob
from .blacklist import really_bad_gauges


def file_checker(file_path: str,
                 meteo_predictors: list,
                 hydro_target: str,
                 possible_nans: int = 0):

    ds_file = xr.open_dataset(file_path).to_dataframe()[
        [hydro_target, *meteo_predictors]]

    condition_1 = ds_file.isna().sum().sum() != possible_nans
    # if (hydro_target == 'q_mm_day') | (hydro_target == 'q_cms_s'):
    #     condition_2 = max([max(value/ds_file[hydro_target].to_numpy())
    #                        for value
    #                        in ds_file[hydro_target].to_numpy()]) > 2000
    #     condition = condition_1 | condition_2
    # else:
    #     condition = condition_1

    return condition_1


def file_rewriter(file_pathes: list,
                  area_index,
                  ts_dir: Path,
                  hydro_target: str,
                  meteo_predictors: list,
                  possible_nans: int = 0):
    shutil.rmtree(ts_dir)
    ts_dir.mkdir(exist_ok=True, parents=True)

    for file in file_pathes:
        gauge_id = file.split('/')[-1][:-3]

        if gauge_id in really_bad_gauges:
            pass
        elif gauge_id not in area_index:
            pass
        else:
            cond = file_checker(file_path=file,
                                hydro_target=hydro_target,
                                meteo_predictors=meteo_predictors,
                                possible_nans=possible_nans)
            if cond:
                continue
            else:
                ds = xr.open_dataset(file)
                try:
                    filename = file.split('/')[-1]
                    ds = ds.drop('gauge_id')
                    ds = ds.sel()
                    ds.to_netcdf(f'{ts_dir}/{filename}')
                except ValueError:
                    continue
    basins = [file.split('/')[-1][:-3] for
              file in glob.glob(f'{ts_dir}/*.nc')]
    with open('./openf_basins.txt', 'w') as the_file:
        for gauge_name in basins:
            the_file.write(f'{int(gauge_name)}\n')
