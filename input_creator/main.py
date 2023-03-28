from tqdm import tqdm
import geopandas as gpd
from pathlib import Path
import numpy as np
import pandas as pd
import glob
import xarray as xr
from scripts.readers import read_with_date_index, xr_opener
import sys
sys.path.append('/workspaces/my_dissertation/')
from meteo_grids_parser.scripts.loaders import multi_var_nc


meteo_data = Path('/workspaces/my_dissertation/geo_data/meteo_grids')
# era5 land
e5l_t2max = f'{meteo_data}/era5_land/2m_temperature_max'
e5l_t2min = f'{meteo_data}/era5_land/2m_temperature_min'
e5l_prcp = f'{meteo_data}/era5_land/total_precipitation'
# era5
e5_t2max = f'{meteo_data}/era5/2m_temperature_max'
e5_t2min = f'{meteo_data}/era5/2m_temperature_min'
e5_prcp = f'{meteo_data}/era5/total_precipitation'
# gpcp
gpcp_prcp = f'{meteo_data}/gpcp/precipitation'
# imerg
imerg_prcp = f'{meteo_data}/imerg/precipitation'
# mswep
mswep_prcp = f'{meteo_data}/mswep/precipitation'
# gleam
gleam_vals = f'{meteo_data}/gleam'
gleam_vals = multi_var_nc(Path(gleam_vals),
                          file_extension='csv')


geometry_file = gpd.read_file(
    '/workspaces/my_dissertation/geo_data/geometry/russia_ws.gpkg')
geometry_file = geometry_file.set_index('gauge_id')

geo_folder = '/workspaces/my_dissertation/geo_data/great_db'
hydrological_storage = f'{geo_folder}/ais_data'
gauge_data = dict()
bad_gauges = list()

for gauge in tqdm(geometry_file.index):
    try:
        # get precomputed value in form of dataframe
        qh_vals = read_with_date_index(
            f'{hydrological_storage}/q_h/{gauge}.csv')
        # era5 land
        tmax_e5l = read_with_date_index(f'{e5l_t2max}/{gauge}.csv')
        tmax_e5l = tmax_e5l.rename(columns={'t2m': 't_max_e5l'})
        tmax_e5l -= 273.15

        tmin_e5l = read_with_date_index(f'{e5l_t2min}/{gauge}.csv')
        tmin_e5l = tmin_e5l.rename(columns={'t2m': 't_min_e5l'})
        tmin_e5l -= 273.15

        prcp_e5l = read_with_date_index(f'{e5l_prcp}/{gauge}.csv')
        prcp_e5l = prcp_e5l.rename(columns={'tp': 'prcp_e5l'})
        prcp_e5l *= 1e2
        # era5
        tmax_e5 = read_with_date_index(f'{e5_t2max}/{gauge}.csv')
        tmax_e5 = tmax_e5.rename(columns={'t2m': 't_max_e5'})
        tmax_e5 -= 273.15

        tmin_e5 = read_with_date_index(f'{e5_t2min}/{gauge}.csv')
        tmin_e5 = tmin_e5.rename(columns={'t2m': 't_min_e5'})
        tmin_e5 -= 273.15

        prcp_e5 = read_with_date_index(f'{e5_prcp}/{gauge}.csv')
        prcp_e5 = prcp_e5.rename(columns={'tp': 'prcp_e5'})
        prcp_e5 *= 1e3
        # gpcp
        prcp_gpcp = read_with_date_index(f'{gpcp_prcp}/{gauge}.csv')
        prcp_gpcp = prcp_gpcp.rename(columns={'precip': 'prcp_gpcp'})
        # imerg
        # because imerg not show any valid data above 60 lat fill with nan
        # for any watershed wich lays above
        ws_geom = geometry_file.loc[gauge, 'geometry']

        _, lat = ws_geom.exterior.xy
        if (np.array(lat) > 60).any():
            prcp_imerg = pd.DataFrame()
            prcp_imerg.index = pd.date_range(start='01/01/2008',
                                             end='12/31/2020')
            prcp_imerg['prcp_imerg'] = np.NaN
        else:
            prcp_imerg = read_with_date_index(f'{imerg_prcp}/{gauge}.csv')
            prcp_imerg = prcp_imerg.rename(
                columns={'imerg_prcp': 'prcp_imerg'})
            prcp_imerg *= 1e-1
        # mswep
        prcp_mswep = read_with_date_index(f'{mswep_prcp}/{gauge}.csv')
        prcp_mswep = prcp_mswep.rename(columns={'precipitation': 'prcp_mswep'})
        # gleam
        gleam_res = dict()
        gleam_res[gauge] = list()
        for gleam_var in gleam_vals.keys():  # type: ignore
            for files in gleam_vals[gleam_var]:  # type: ignore
                if gauge == files.split('/')[-1][:-4]:
                    gleam_res[gauge].append(files)
        gleam_res = pd.concat([read_with_date_index(file)
                               for file in gleam_res[gauge]],
                              axis=1)['2008':'2020']
        # combine it to xarray
        res_xr = pd.concat([qh_vals, tmax_e5l, tmax_e5,
                            tmin_e5l, tmin_e5,
                            prcp_e5l, prcp_e5,
                            prcp_gpcp, prcp_imerg,
                            prcp_mswep, gleam_res],
                           axis=1)['2008':'2020'].to_xarray()
        res_xr = res_xr.assign_coords(gauge_id=('gauge_id', [gauge]))

        res_xr.to_netcdf(f'{geo_folder}/nc_concat/{gauge}.nc')
        # store to one
        gauge_data[gauge] = res_xr
    except FileNotFoundError:
        with open('bad_gauges.txt', 'w') as text_file:
            text_file.write(f'{gauge} has no observations\n')
        bad_gauges.append(gauge)

# read all stored files in one file
files = glob.glob(f'{geo_folder}/nc_concat/*.nc')
big_file = xr.concat([xr_opener(file) for file in files],
                     dim='gauge_id')
# save only one's with no miss data on discharge in qms/s
for gauge in big_file['q_cms_s'].dropna(dim='gauge_id')['gauge_id'].values:
    ds = big_file.sel(gauge_id=gauge)
    ds.to_netcdf(f'{geo_folder}/nc_all_q/{gauge}.nc')
# save only one's with no miss data on level in relative sm
for gauge in big_file['lvl_mbs'].dropna(dim='gauge_id')['gauge_id'].values:
    ds = big_file.sel(gauge_id=gauge)
    ds.to_netcdf(f'{geo_folder}/nc_all_h/{gauge}.nc')
