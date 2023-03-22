import pandas as pd
import xarray as xr


def ais_reader(file_path: str):

    data = pd.read_csv(file_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')

    return data


def file_by_district(all_files: list,
                     save_storage: str,
                     variable_tag: str,
                     height_bs: pd.DataFrame):
    lvl_dict = {}
    for gauge in height_bs.index:
        gauge = str(gauge)
        lvl_dict[gauge] = []
        for lvl_f in all_files:
            if gauge == lvl_f.split('/')[-1][:-4]:
                lvl_dict[gauge].append(lvl_f)

    for gauge, files in lvl_dict.items():
        try:
            res_df = pd.concat([ais_reader(file)
                                for file in files]).sort_index()
            res_df.to_csv(f'{save_storage}/{variable_tag}/{gauge}.csv')
        except ValueError:
            continue


def read_with_date_index(file_path: str):

    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')

    return data


def xr_opener(file):
    ds = xr.open_dataset(file)
    if 'index' in ds.coords:
        ds = ds.rename({'index': 'date'})
    return ds
