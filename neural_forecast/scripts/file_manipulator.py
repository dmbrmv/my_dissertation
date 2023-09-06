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
    # if (hydro_target == 'q_mm_day') | (hydro_target == 'q_cms_s'):
    #     condition_2 = max([max(value/ds_file[hydro_target].to_numpy())
    #                        for value
    #                        in ds_file[hydro_target].to_numpy()]) > 2000
    #     condition = condition_1 | condition_2
    # else:
    #     condition = condition_1

    return condition_1


def train_rewriter(era_pathes: list,
                   area_index,
                   ts_dir: Path,
                   hydro_target: str,
                   predictors: list,
                   possible_nans: int = 0):
    shutil.rmtree(ts_dir)
    ts_dir.mkdir(exist_ok=True, parents=True)

    for file in era_pathes:
        gauge_id = file.split('/')[-1][:-3]

        if gauge_id not in area_index:
            pass
        else:
            cond = file_checker(file_path=file,
                                hydro_target=hydro_target,
                                meteo_predictors=predictors,
                                possible_nans=possible_nans)
            if cond:
                continue
            else:
                era_file = xr.open_dataset(file)

                try:
                    filename = file.split('/')[-1]
                    era_file = era_file.to_dataframe()[[hydro_target,
                                                        *predictors]
                                                       ].to_xarray()
                    era_file.to_netcdf(f'{ts_dir}/{filename}')
                except ValueError:
                    continue
    basins = [file.split('/')[-1][:-3] for
              file in glob.glob(f'{ts_dir}/*.nc')]
    with open('./every_basin.txt', 'w') as the_file:
        for gauge_name in basins:
            the_file.write(f'{int(gauge_name)}\n')


def test_rewriter(train_with_obs: list,
                  cmip_storage: str,
                  predictors: list,
                  test_index,
                  ts_dir: Path):
    shutil.rmtree(ts_dir)
    ts_dir.mkdir(exist_ok=True, parents=True)

    for file in train_with_obs:
        gauge_id = file.split('/')[-1][:-3]
        filename = file.split('/')[-1]

        if gauge_id not in test_index:
            pass
        else:
            try:
                obs_file = xr.open_dataset(file)
                obs_file = obs_file.to_dataframe(
                )[['lvl_sm', 'q_cms_s', 'lvl_mbs', 'q_mm_day']]

                cmip_file = xr.open_dataset(f'{cmip_storage}/{gauge_id}.nc')
                cmip_file = cmip_file.to_dataframe().droplevel(1)[predictors]

                res_file = cmip_file.join(obs_file).to_xarray()
                res_file.to_netcdf(f'{ts_dir}/{filename}')
            except ValueError:
                continue
    basins = [file.split('/')[-1][:-3] for
              file in glob.glob(f'{ts_dir}/*.nc')]
    with open('./every_basin.txt', 'w') as the_file:
        for gauge_name in basins:
            the_file.write(f'{int(gauge_name)}\n')
