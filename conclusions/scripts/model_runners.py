from copy import deepcopy
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import xarray as xr


def read_gauge(gauge_id: str, simple: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read data from a gauge station and split it into training and test data.

    The data is read from a netCDF file and contains the following columns:
    - q_mm_day: daily discharge
    - prcp_e5l: daily precipitation
    - t_min_e5l: daily minimum temperature
    - t_max_e5l: daily maximum temperature
    - Ep: daily evaporation

    The data is then split into training and test data. The training data is all the data
    up to the end of 2018 and the test data is all the data from 2018 onwards.

    Args:
        gauge_id (str): The ID of the gauge station.
        simple (bool, optional): If True, only the columns 'q_mm_day', 'prcp_e5l', 't_min_e5l',
            and 't_max_e5l' are returned. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of two pandas DataFrames. The first DataFrame is the training data
            and the second DataFrame is the test data.

    """
    test_df = xr.open_dataset(f"/app/data/nc_all_q/{gauge_id}.nc").to_dataframe()[
        ["q_mm_day", "prcp_e5l", "t_min_e5l", "t_max_e5l", "Ep"]
    ]

    if simple:
        test_df.index.name = "Date"

        train = test_df[:"2018"]
        test = test_df["2018":]
    else:
        # Calculate the mean temperature
        test_df["Temp"] = test_df[["t_max_e5l", "t_min_e5l"]].mean(axis=1)

        # Rename the columns
        test_df = test_df.rename(columns={"prcp_e5l": "Prec", "Ep": "Evap", "q_mm_day": "Q_mm"})

        # Drop the columns that are not needed
        test_df = test_df.drop(["t_min_e5l", "t_max_e5l"], axis=1)

        train = test_df[:"2018"]
        test = test_df["2018":]

    return train, test


def day_agg(df: pd.DataFrame, day_aggregations: list = [2**n for n in range(9)]):
    for days in day_aggregations:
        df[[f"prcp_{days}"]] = df[["prcp_e5l"]].rolling(window=days).sum()
        df[[f"t_min_{days}"]] = df[["t_min_e5l"]].rolling(window=days).mean()
        df[[f"t_max_{days}"]] = df[["t_min_e5l"]].rolling(window=days).mean()
    df = df.dropna()

    return df


def feature_target(
    data: pd.DataFrame, day_aggregations: list = [2**n for n in range(9)]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract and normalize features and target from the input data.

    Features are calculated by aggregating precipitation, minimum and maximum temperature
    over different time periods. The time periods are specified by the day_aggregations list.
    The features are then normalized by subtracting the mean and dividing by the standard deviation.

    The target is the time series of daily discharge.

    Args:
        data (pd.DataFrame): Input data, must contain columns 'prcp_e5l', 't_min_e5l', 't_max_e5l', and 'q_mm_day'.
        day_aggregations (list, optional): List of time periods to aggregate over. Defaults to [2**n for n in range(9)].

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple of features and target.

    """
    data = day_agg(df=data)

    # get meteo columns
    feature_cols = [
        item
        for sublist in [
            [f"{var}_{day}" for day in day_aggregations] for var in ["prcp", "t_min", "t_max"]
        ]
        for item in sublist
    ]
    features = data[feature_cols]
    # normalize features
    features = (features - features.mean()) / features.std()

    target = data[["q_mm_day"]]
    target = target.to_numpy().ravel()
    features = features.to_numpy()

    return (features, target)


def nse(predictions, targets):
    return 1 - (
        np.nansum((targets - predictions) ** 2) / np.nansum((targets - np.nanmean(targets)) ** 2)
    )


def rfr_launch(gauge_id: str):
    train, test = read_gauge(gauge_id=gauge_id, simple=True)
    # set data
    x_test, _ = feature_target(deepcopy(test))
    rfr_model = joblib.load(f"../conceptual_runs/cal_res/rfr/{gauge_id}.joblib")
    # get prediction
    fin_df = test.iloc[255:, :]
    fin_df["q_mm_rfr"] = rfr_model.predict(x_test)
    res_nse = nse(predictions=fin_df["q_mm_rfr"], targets=fin_df["q_mm_day"])

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
