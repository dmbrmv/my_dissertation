from conceptual_runs.calibration.calibrator import calibrate_gauge
from conceptual_runs.hydro_models import hbv, gr4j_cema_neige
from sklearn.metrics import mean_squared_error
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def nse(predictions, targets):
    return 1-(
        np.nansum((targets-predictions)**2)/np.nansum(
            (targets-np.nanmean(targets))**2))


def kge(predictions, targets):
    sim_mean = np.mean(targets, axis=0, dtype=np.float64)
    obs_mean = np.mean(predictions, dtype=np.float64)

    r_num = np.sum((targets - sim_mean) * (predictions - obs_mean),
                   axis=0, dtype=np.float64)
    r_den = np.sqrt(np.sum((targets - sim_mean) ** 2,
                           axis=0, dtype=np.float64)
                    * np.sum((predictions - obs_mean) ** 2,
                             dtype=np.float64))
    r = r_num / r_den
    # calculate error in spread of flow alpha
    alpha = np.std(targets, axis=0) / np.std(predictions, dtype=np.float64)
    # calculate error in volume beta (bias of mean discharge)
    beta = (np.sum(targets, axis=0, dtype=np.float64)
            / np.sum(predictions, dtype=np.float64))
    # calculate the Kling-Gupta Efficiency KGE
    kge_ = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return np.vstack((kge_, r, alpha, beta))


def rmse(predictions, targets):
    return mean_squared_error(targets, predictions, squared=False)


def relative_error(predictions, targets):
    return ((targets - predictions)/targets) * 100


def read_gauge(gauge_id: str,
               simple: bool = False):

    test_df = xr.open_dataset(
        f'../geo_data/great_db/nc_all_q/{gauge_id}.nc').to_dataframe(
    )[['q_mm_day', 'prcp_e5l', 't_min_e5l', 't_max_e5l', 'Ep']]

    if simple:
        test_df.index.name = 'Date'

        train = test_df[:'2018']
        test = test_df['2018':]
    else:
        test_df['Temp'] = test_df[['t_max_e5l', 't_min_e5l']].mean(axis=1)
        test_df = test_df.rename(columns={'prcp_e5l': 'Prec',
                                          'Ep': 'Evap',
                                          'q_mm_day': 'Q_mm'})
        test_df.index.name = 'Date'
        test_df = test_df.drop(['t_min_e5l', 't_max_e5l'], axis=1)
        test_df.loc[test_df['Evap'] < 0, 'Evap'] = 0
        # test_df['Evap'] *= 1e1

        train = test_df[:'2018']
        test = test_df['2018':]

    return train, test


def get_params(model_name: str,
               params_path: Path,
               gauge_id: str,
               train: pd.DataFrame,
               test: pd.DataFrame,
               calibrate: bool = False,
               with_plot: bool = False):
    if calibrate:
        calibrate_gauge(df=train, hydro_models=[model_name],
                        res_calibrate=f'{params_path}/{gauge_id}',
                        # xD
                        iterations=600)

    lines = open(f'{params_path}/{gauge_id}', 'r').read().splitlines()
    params = eval(lines[3].split(':')[1])[0]
    if model_name == 'gr4j':
        test['Q_sim'] = gr4j_cema_neige.simulation(data=test, params=params)
    elif model_name == 'hbv':
        test['Q_sim'] = hbv.simulation(data=test, params=params)

    res_nse = nse(predictions=test['Q_sim'], targets=test['Q_mm'])
    if with_plot:
        test[['Q_sim', 'Q_mm']].plot()
        plt.title(f'NSE -- {res_nse}')
        plt.show()

    return res_nse, test


def day_agg(df: pd.DataFrame,
            day_aggregations: list = [2**n for n in range(9)]):
    for days in day_aggregations:
        df[[f'prcp_{days}']] = df[['prcp_e5l']].rolling(
            window=days).sum()
        df[[f't_min_{days}']] = df[['t_min_e5l']].rolling(
            window=days).mean()
        df[[f't_max_{days}']] = df[['t_min_e5l']].rolling(
            window=days).mean()
    df = df.dropna()

    return df


def feature_target(data: pd.DataFrame,
                   day_aggregations: list = [2**n for n in range(9)]):
    """_summary_
    Args:
        data (pd.DataFrame): _description_
        day_aggregations (list, optional): _description_.
        Defaults to [2**n for n in range(9)].
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: _description_
    """

    data = day_agg(df=data)

    # get meteo columns
    feature_cols = [item for sublist in
                    [[f'{var}_{day}' for day in day_aggregations]
                     for var in ['prcp', 't_min', 't_max']]
                    for item in sublist]
    features = data[feature_cols]
    # normalize features
    features = (features - features.mean())/features.std()

    target = data[['q_mm_day']]
    target = target.to_numpy().ravel()
    features = features.to_numpy()

    return (features, target)
