import xarray as xr
from pathlib import Path
import shutil
import glob


def file_checker(dataset: xr.Dataset,
                 possible_nans: int = 0):

    ds_file = dataset.to_dataframe()

    condition_1 = ds_file.isna().sum().sum() != possible_nans

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
            era_file = xr.open_dataset(file)[[hydro_target, *predictors]]
            cond = file_checker(era_file, possible_nans=0)
            if cond:
                continue
            else:
                try:
                    filename = file.split('/')[-1]
                    era_file = era_file.to_dataframe().to_xarray()
                    era_file.to_netcdf(f'{ts_dir}/{filename}')
                except ValueError:
                    continue
    basins = [file.split('/')[-1][:-3] for
              file in glob.glob(f'{ts_dir}/*.nc')]
    with open('./every_basin.txt', 'w') as the_file:
        for gauge_name in basins:
            the_file.write(f'{int(gauge_name)}\n')


def train_cmip_val(era_pathes: list,
                   area_index,
                   ts_dir: Path,
                   hydro_target: str,
                   era_cols: list,
                   cmip_cols: list,
                   cmip_storage: str,
                   val_start: str,
                   val_end: str,
                   possible_nans: int = 0):
    shutil.rmtree(ts_dir)
    ts_dir.mkdir(exist_ok=True, parents=True)

    for file in era_pathes:
        gauge_id = file.split('/')[-1][:-3]

        if gauge_id not in area_index:
            pass
        else:
            era_file = xr.open_dataset(file)[[hydro_target,
                                              *era_cols]]
            cmip_file = xr.open_dataset(
                f'{cmip_storage}/{gauge_id}.nc')[cmip_cols]
            cond_1 = file_checker(dataset=era_file,
                                  possible_nans=possible_nans)
            if cond_1:
                # read era with necessary columns
                era_file = era_file.to_dataframe()
                era_file = era_file.interpolate()
            else:
                era_file = era_file.to_dataframe()

            cond_2 = file_checker(dataset=cmip_file,
                                  possible_nans=possible_nans)
            if cond_2:
                # read cmip file
                cmip_file = cmip_file.to_dataframe()
                cmip_file = cmip_file.interpolate()
            else:
                cmip_file = cmip_file.to_dataframe()
            # rename columns with era correspondend names
            cmip_file = cmip_file.rename(
                columns={old_col: new_col
                         for new_col, old_col
                         in zip(era_cols, cmip_cols)})
            era_file.loc[
                f'{val_start}':f'{val_end}',
                era_cols] = cmip_file.loc[f'{val_start}':f'{val_end}',
                                          era_cols]
            try:
                filename = file.split('/')[-1]
                era_file = era_file.to_xarray()
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
                  era_names: list,
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
                # cmip_file = cmip_file.dropna(dim='date')
                cmip_file = cmip_file.to_dataframe().droplevel(1)[predictors]
                cmip_file = cmip_file.dropna()
                cmip_file = cmip_file.rename(
                    columns={old_col: new_col
                             for new_col, old_col
                             in zip(era_names, predictors)})

                res_file = cmip_file.join(obs_file).to_xarray()
                res_file.to_netcdf(f'{ts_dir}/{filename}')
            except ValueError:
                continue
    basins = [file.split('/')[-1][:-3] for
              file in glob.glob(f'{ts_dir}/*.nc')]
    with open('./every_basin.txt', 'w') as the_file:
        for gauge_name in basins:
            the_file.write(f'{int(gauge_name)}\n')
