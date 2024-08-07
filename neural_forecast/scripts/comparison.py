import xarray as xr
from scipy.stats import pearsonr
import pandas as pd
import numpy as np


def percent_diff(era_data,
                 cmip_data):
    dif = np.mean(era_data - cmip_data)/np.mean(era_data)

    return np.abs(dif * 100)


def gauge_cmip(gauge_id: str,
               cmip_storage: str,
               era_storage: str,
               compare_start: str = '2018-01-01',
               compare_end: str = '2020-01-01'):
    with xr.open_dataset(f'{cmip_storage}/{gauge_id}.nc') as cmip_f:
        cmip_gauge = cmip_f
    with xr.open_dataset(f'{era_storage}/{gauge_id}.nc') as era_f:
        era_gauge = era_f
        
    # select range from era5l
    if 'index' in era_gauge.coords:
        era_prcp = era_gauge[['prcp_e5l']].rename({'index': 'date'}).sel(
            date=slice(compare_start,
                       compare_end)).to_dataframe()
    else:
        era_prcp = era_gauge[['prcp_e5l']].sel(
            date=slice(compare_start,
                       compare_end)).to_dataframe()
    era_gauge.close()
    # get model names in dataframe and for later records
    model_names = {'_'.join(v.split('_')[1:]): v
                   for v in cmip_gauge.data_vars if 'precipitation' in v}
    # create unified dataframe with cmip and era precipitation
    for model_name, model_column in model_names.items():
        cmip_p = cmip_gauge[[model_column]].sel(
            date=slice(compare_start, compare_end)).to_dataframe()
        cmip_p = cmip_p.rename(columns={model_column: model_name})
        era_prcp = era_prcp.join(cmip_p)

    cmip_gauge.close()
    # create final result
    res_df = pd.DataFrame()
    era_p = era_prcp['prcp_e5l']
    model_order = dict()

    for i, model_name in enumerate(model_names.keys()):
        model_order[i+1] = model_name
        model_p = era_prcp.loc[:, model_name]
        if sum(model_p.isna()) > len(model_p) * 0.1:
            continue
        elif sum(model_p.isna()) == 0:
            model_p = model_p
        else:
            model_p = model_p.interpolate()
        res_df.loc[model_name, 'number'] = i+1
        res_df.loc[model_name, 'r'] = pearsonr(x=era_p, y=model_p)[0]
        res_df.loc[model_name, 'error'] = percent_diff(era_data=era_p,
                                                       cmip_data=model_p)
    res_df = res_df.sort_values(by=['error', 'r'], ascending=True)

    return res_df, model_order
