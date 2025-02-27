import numpy as np
import pandas as pd
import xarray as xr


def nse(predictions, targets):
    return 1 - (
        np.nansum((targets - predictions) ** 2)
        / np.nansum((targets - np.nanmean(targets)) ** 2)
    )


def kge(predictions, targets):
    sim_mean = np.mean(targets, axis=0)
    obs_mean = np.mean(predictions)

    r_num = np.sum((targets - sim_mean) * (predictions - obs_mean), axis=0)
    r_den = np.sqrt(
        np.sum((targets - sim_mean) ** 2, axis=0)
        * np.sum(
            (predictions - obs_mean) ** 2,
        )
    )
    if r_den == 0:
        r = np.nan
    else:
        r = r_num / r_den
    # calculate error in spread of flow alpha
    if np.std(predictions) == 0:
        alpha = np.nan
    else:
        alpha = np.std(targets, axis=0) / np.std(predictions)
    # calculate error in volume beta (bias of mean discharge)
    if np.sum(predictions) == 0:
        beta = np.nan
    else:
        beta = np.sum(targets, axis=0) / np.sum(predictions)
    # calculate the Kling-Gupta Efficiency KGE
    kge_ = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return kge_, r, alpha, beta


def root_mean_squared_error(predictions, targets):
    # Mask nan values in both arrays
    mask = ~np.isnan(targets) & ~np.isnan(predictions)

    # Apply mask to remove nans from both arrays
    y_true_clean = targets[mask]
    y_pred_clean = predictions[mask]

    # Compute RMSE on non-nan values
    rmse = np.sqrt(np.nanmean((y_true_clean - y_pred_clean) ** 2))
    return rmse


def relative_error(predictions, targets):
    # Mask nan values in both arrays
    mask = ~np.isnan(targets) & ~np.isnan(predictions)
    # Apply mask to remove nans from both arrays
    y_true_clean = targets[mask]
    y_pred_clean = predictions[mask]
    return np.nanmedian(((y_true_clean - y_pred_clean) / y_true_clean) * 100)


def metric_df(gauge_id, predictions, targets):
    res_df = pd.DataFrame()

    res_df.loc[gauge_id, "NSE"] = nse(predictions, targets)

    res_df.loc[gauge_id, ["KGE", "r", "alpha", "beta"]] = kge(predictions, targets)

    res_df.loc[gauge_id, "RMSE"] = root_mean_squared_error(predictions, targets)

    res_df.loc[gauge_id, "delta"] = relative_error(predictions, targets)

    res_df.index.name = "gauge_id"

    return res_df


def read_gauge(gauge_id: str, simple: bool = False):
    with xr.open_dataset(f"/app/data/nc_all_q/{gauge_id}.nc") as f:
        test_df = f.to_dataframe()[["q_mm_day", "prcp_e5l", "t_min_e5l", "t_max_e5l"]]
    if simple:
        test_df.index.name = "Date"

        train = test_df[:"2018"]
        test = test_df["2018":]
    else:
        test_df["Temp"] = test_df[["t_max_e5l", "t_min_e5l"]].mean(axis=1)
        test_df = test_df.rename(
            columns={"prcp_e5l": "Prec", "Ep": "Evap", "q_mm_day": "Q_mm"}
        )
        test_df.index.name = "Date"
        test_df = test_df.drop(["t_min_e5l", "t_max_e5l"], axis=1)
        test_df.loc[test_df["Evap"] < 0, "Evap"] = 0
        # test_df['Evap'] *= 1e1

        train = test_df[:"2018"]
        test = test_df["2018":]

    return train, test


def day_agg(df: pd.DataFrame, day_aggregations: list = (2**n for n in range(6))):
    for days in day_aggregations:
        df[[f"prcp_{days}"]] = df[["prcp_e5l"]].rolling(window=days).sum()
        df[[f"t_min_{days}"]] = df[["t_min_e5l"]].rolling(window=days).mean()
        df[[f"t_max_{days}"]] = df[["t_min_e5l"]].rolling(window=days).mean()
    df = df.dropna()

    return df


def feature_target(
    data: pd.DataFrame, day_aggregations: list = [2**n for n in range(6)]
):
    """_summary_.

    Args:
        data (pd.DataFrame): _description_
        day_aggregations (list, optional): _description_.
        Defaults to [2**n for n in range(6)].

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: _description_

    """
    data = day_agg(df=data)

    # get meteo columns
    feature_cols = [
        item
        for sublist in [
            [f"{var}_{day}" for day in day_aggregations]
            for var in ["prcp", "t_min", "t_max"]
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


def model_ready_file(data_file: pd.DataFrame):
    obs_q = data_file.loc[:, "q_mm_day"].values
    temp_max = data_file.loc[:, "t_min_e5l"].values
    temp_min = data_file.loc[:, "t_max_e5l"].values
    temp_mean = (temp_max + temp_min) / 2
    evap = data_file.loc[:, "E"].values
    prcp = data_file.loc[:, "prcp_e5l"].values

    model_df = pd.DataFrame()
    model_df.index = data_file.index
    model_df["Temp"] = temp_mean
    model_df["Evap"] = evap
    model_df["Prec"] = prcp
    model_df["Q_mm"] = obs_q

    return model_df
