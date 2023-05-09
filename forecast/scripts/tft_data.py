import pandas as pd
from tqdm.notebook import tqdm
import xarray as xr


def file_checker(file_path: str,
                 meteo_predictors: list,
                 hydro_target: str):
    condition = xr.open_dataset(
        file_path).to_dataframe()[
            [hydro_target,
             *meteo_predictors]].isna().sum().sum() != 0
    return condition


def open_for_tft(nc_files: list,
                 static_path: str,
                 meteo_predictors: list,
                 hydro_target: str,
                 index_col: str = 'gauge_id') -> pd.DataFrame:

    static_attributes = pd.read_csv(static_path,
                                    index_col=index_col)
    static_attributes.index = static_attributes.index.astype(str)

    res_file = list()
    for file_path in tqdm(nc_files):
        gauge_id = file_path.split('/')[-1][:-3]
        try:
            gauge_static = static_attributes.loc[[gauge_id], :]
        except KeyError:
            print(f'No data for {gauge_id} !')
            continue
        cond = file_checker(file_path=file_path,
                            meteo_predictors=meteo_predictors,
                            hydro_target=hydro_target)
        if cond:
            continue
        else:
            file = xr.open_dataset(file_path)
            file = file.to_dataframe()
            # file['date'] = file.index
            file = file.reset_index()
            file['time_idx'] = file.index

            for col in gauge_static.columns:
                file[col] = gauge_static.loc[gauge_id, col]

            res_file.append(file)

    file = pd.concat(res_file, axis=0)
    file = file.reset_index(drop=True)
    return file
