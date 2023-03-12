import pandas as pd
import xarray as xr


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
