from calibration.calibrator import calibrate_gauge
from hydro_models import hbv, gr4j_cema_neige
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def nse(predictions, targets):
    return 1-(
        np.sum((targets-predictions)**2)/np.sum(
            (targets-np.mean(targets))**2))


def read_gauge(gauge_id: str):
    test_df = xr.open_dataset(
        f'../geo_data/great_db/nc_all_q/{gauge_id}.nc').to_dataframe(
            )[['q_mm_day', 'prcp_e5l', 't_min_e5l', 't_max_e5l', 'Ep']]
    test_df['Temp'] = test_df[['t_max_e5l', 't_min_e5l']].mean(axis=1)
    test_df = test_df.rename(columns={'prcp_e5l': 'Prec',
                                      'Ep': 'Evap',
                                      'q_mm_day': 'Q_mm'})
    test_df.index.name = 'Date'
    test_df = test_df.drop(['t_min_e5l', 't_max_e5l'], axis=1)

    train = test_df[:'2018']
    test = test_df['2018':]

    return train, test


def get_params(model_name: str,
               params_path: Path,
               gauge_id: str,
               train: pd.DataFrame,
               test: pd.DataFrame,
               with_plot: bool = False):
    calibrate_gauge(df=train, hydro_models=[model_name],
                    res_calibrate=f'{params_path}/{gauge_id}',
                    # xD
                    iterations=690)

    lines = open(f'{params_path}/{gauge_id}', 'r').read().splitlines()
    params = eval(lines[3].split(':')[1])[0]
    if model_name == 'gr4j':
        test['Q_sim'] = gr4j_cema_neige.simulation(data=test, params=params)
    elif model_name == 'hbv':
        test['Q_sim'] = hbv.simulation(data=test, params=params)

    res_nse = nse(predictions=test['Q_sim'], targets=test['Q_mm'])
    if with_plot:
        test[['Q_sim', 'Q_mm']].plot()
        plt.title(f'Normalized NSE -- {res_nse}')

    return res_nse
