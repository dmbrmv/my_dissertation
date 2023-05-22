import pandas as pd
import numpy as np
import xarray as xr
import joblib
from copy import deepcopy

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

        train = test_df[:'2018']
        test = test_df['2018':]

    return train, test


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


def nse(predictions, targets):
    return 1-(
        np.nansum((targets-predictions)**2)/np.nansum(
            (targets-np.nanmean(targets))**2))


def rfr_launch(gauge_id: str):
    train, test = read_gauge(gauge_id=gauge_id, simple=True)
    # set data
    x_test, _ = feature_target(deepcopy(test))
    rfr_model = joblib.load(
        f'../conceptual_runs/cal_res/rfr/{gauge_id}.joblib')
    # get prediction
    fin_df = test.iloc[255:, :]
    fin_df['q_mm_rfr'] = rfr_model.predict(x_test)
    res_nse = nse(predictions=fin_df['q_mm_rfr'],
                  targets=fin_df['q_mm_day'])

    return res_nse, fin_df


def hbv_launch(gauge_id: str):
    pass


def gr4f_launch(gauge_id: str):
    pass


def tft_launch(gauge_id: str):
    pass


def lstm_launch(gauge_id: str):
    pass


def catboost_launch(gauge_id: str):
    pass