import pandas as pd
import numpy as np


def station_lat_lon(geom_row):
    lon, lat = geom_row.xy
    lon, lat = lon[0], lat[0]

    return lon, lat


def read_meteo_station(csv_path):
    test_meteo = pd.read_csv(csv_path, index_col='date')
    test_meteo.index = pd.to_datetime(test_meteo.index)
    test_meteo = test_meteo.loc['2008':'2020', :]
    test_meteo = test_meteo.applymap(
        lambda x: pd.to_numeric(x, errors='coerce'))

    return test_meteo


def xr_for_point(xr_file, meteo_col, meteo_lat, meteo_lon):
    xr_file = xr_file.to_array().squeeze()
    xr_lons = xr_file['lon'].values
    xr_lats = xr_file['lat'].values
    lons_id = np.argmin(np.abs((xr_lons - meteo_lon)))
    lats_id = np.argmin(np.abs((xr_lats - meteo_lat)))
    # write down to dataframe
    xr_df = pd.DataFrame()
    xr_df[f'{meteo_col}'] = xr_file[:, lats_id, lons_id]
    xr_df['date'] = xr_file.time
    xr_df = xr_df.set_index('date')
    xr_df = xr_df.loc['2008':'2020', :]

    return xr_df


def df_reader(csv_path):
    """_summary_

    Args:
        csv_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    test_df = pd.read_csv(csv_path, index_col='date')
    test_df.index = pd.to_datetime(test_df.index)
    test_df[['era5_land_prcp']] *= 1e2
    test_df[['era5_prcp']] *= 1e3
    test_df[['imerg_prcp']] *= 1e-1
    test_df[['era5_land_2m_max', 'era5_land_2m_min',
            'era5_2m_max', 'era5_2m_min']] -= 273.15

    return test_df
