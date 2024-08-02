import glob
import shutil
from pathlib import Path

import xarray as xr


def file_checker(dataset: xr.Dataset, possible_nans: int = 0):
    ds_file = dataset.to_dataframe()

    condition_1 = ds_file.isna().sum().sum() > possible_nans

    return condition_1


def train_rewriter(
    era_paths: list,
    area_index,
    ts_dir: Path,
    hydro_target: str,
    predictors: list,
    possible_nans: int = 0,
) -> None:
    """Process ERA files, checks for missing data, and writes valid datasets to netCDF files.

    Parameters
    ----------
    - era_paths: List of paths to ERA dataset files.
    - area_index: Index or list of gauge IDs to include in the processing.
    - ts_dir: Path object representing the directory to save processed netCDF files.
    - hydro_target: The name of the hydrological target variable in the dataset.
    - predictors: List of predictor variable names to include from the ERA dataset.
    - possible_nans: The allowable number of NaN values in the dataset. Defaults to 0.

    This function reads each ERA file specified in `era_paths`, filters out files not listed in
    `area_index`, checks for missing data exceeding `possible_nans`, and writes the cleaned and
    filtered data to new netCDF files in `ts_dir`. Each output file is named after its corresponding
    gauge ID. Additionally, it generates a text file listing all processed gauge IDs.

    """
    shutil.rmtree(ts_dir)
    ts_dir.mkdir(exist_ok=True, parents=True)

    for file in era_paths:
        gauge_id = Path(file).stem

        if gauge_id not in area_index:
            pass
        else:
            era_file = xr.open_dataset(file)[[hydro_target, *predictors]]
            cond = file_checker(era_file, possible_nans=possible_nans)
            if cond:
                continue
            else:
                try:
                    filename = file.split("/")[-1]
                    era_file = era_file.to_dataframe().to_xarray()
                    era_file.to_netcdf(f"{ts_dir}/{filename}")
                except ValueError:
                    continue
    basins = [Path(file).stem for file in glob.glob(f"{ts_dir}/*.nc")]
    with open("./every_basin.txt", "w") as the_file:
        for gauge_name in basins:
            the_file.write(f"{int(gauge_name)}\n")


def train_cmip_val(
    era_paths: list,
    area_index,
    ts_dir: Path,
    hydro_target: str,
    era_cols: list,
    cmip_cols: list,
    cmip_storage: str,
    val_start: str,
    val_end: str,
    possible_nans: int = 0,
) -> None:
    """Process ERA files, checks for missing data, and writes valid datasets to netCDF files.
    
    This function reads each ERA file specified in `era_paths`, filters out files not listed in
    `area_index`, checks for missing data exceeding `possible_nans`, and writes the cleaned and
    filtered data to new netCDF files in `ts_dir`. Each output file is named after its corresponding
    gauge ID. Additionally, it generates a text file listing all processed gauge IDs.

    Parameters
    ----------
    - era_paths: List of paths to ERA dataset files.
    - area_index: Index or list of gauge IDs to include in the processing.
    - ts_dir: Path object representing the directory to save processed netCDF files.
    - hydro_target: The name of the hydrological target variable in the dataset.
    - era_cols: List of predictor variable names to include from the ERA dataset.
    - cmip_cols: List of predictor variable names to include from the CMIP dataset.
    - cmip_storage: The path to the CMIP dataset storage.
    - val_start: The start date of the validation period in the format 'YYYY-MM-DD'.
    - val_end: The end date of the validation period in the format 'YYYY-MM-DD'.
    - possible_nans: The allowable number of NaN values in the dataset. Defaults to 0.

    Returns
    -------
    None: This function does not return any value. It writes the processed data to netCDF files and
    generates a text file listing all processed gauge IDs.

    Raises
    ------
    ValueError: If the ERA file or CMIP file contains missing data exceeding `possible_nans`.
    
    """
    shutil.rmtree(ts_dir)
    ts_dir.mkdir(exist_ok=True, parents=True)

    for file in era_paths:
        gauge_id = file.split("/")[-1][:-3]

        if gauge_id not in area_index:
            pass
        else:
            era_file = xr.open_dataset(file)[[hydro_target, *era_cols]]
            cmip_file = xr.open_dataset(f"{cmip_storage}/{gauge_id}.nc")[cmip_cols]
            cond_1 = file_checker(dataset=era_file, possible_nans=possible_nans)
            if cond_1:
                # read era with necessary columns
                era_file = era_file.to_dataframe()
                era_file = era_file.interpolate()
            else:
                era_file = era_file.to_dataframe()

            cond_2 = file_checker(dataset=cmip_file, possible_nans=possible_nans)
            if cond_2:
                # read cmip file
                cmip_file = cmip_file.to_dataframe()
                cmip_file = cmip_file.interpolate()
            else:
                cmip_file = cmip_file.to_dataframe()
            # rename columns with era correspondend names
            cmip_file = cmip_file.rename(
                columns={
                    old_col: new_col for new_col, old_col in zip(era_cols, cmip_cols)
                }
            )
            era_file.loc[f"{val_start}" : f"{val_end}", era_cols] = cmip_file.loc[
                f"{val_start}" : f"{val_end}", era_cols
            ]
            try:
                filename = file.split("/")[-1]
                era_file = era_file.to_xarray()
                era_file.to_netcdf(f"{ts_dir}/{filename}")
            except ValueError:
                continue
    basins = [file.split("/")[-1][:-3] for file in glob.glob(f"{ts_dir}/*.nc")]
    with open("./every_basin.txt", "w") as the_file:
        for gauge_name in basins:
            the_file.write(f"{int(gauge_name)}\n")


def test_rewriter(
    train_with_obs: list,
    cmip_storage: str,
    predictors: list,
    era_names: list,
    test_index,
    ts_dir: Path,
):
    shutil.rmtree(ts_dir)
    ts_dir.mkdir(exist_ok=True, parents=True)

    for file in train_with_obs:
        gauge_id = file.split("/")[-1][:-3]
        filename = file.split("/")[-1]

        if gauge_id not in test_index:
            pass
        else:
            try:
                obs_file = xr.open_dataset(file)
                obs_file = obs_file.to_dataframe()[
                    ["lvl_sm", "q_cms_s", "lvl_mbs", "q_mm_day"]
                ]

                cmip_file = xr.open_dataset(f"{cmip_storage}/{gauge_id}.nc")
                # cmip_file = cmip_file.dropna(dim='date')
                cmip_file = cmip_file.to_dataframe().droplevel(1)[predictors]
                cmip_file = cmip_file.dropna()
                cmip_file = cmip_file.rename(
                    columns={
                        old_col: new_col
                        for new_col, old_col in zip(era_names, predictors)
                    }
                )

                res_file = cmip_file.join(obs_file).to_xarray()
                res_file.to_netcdf(f"{ts_dir}/{filename}")
            except ValueError:
                continue
    basins = [file.split("/")[-1][:-3] for file in glob.glob(f"{ts_dir}/*.nc")]
    with open("./every_basin.txt", "w") as the_file:
        for gauge_name in basins:
            the_file.write(f"{int(gauge_name)}\n")
