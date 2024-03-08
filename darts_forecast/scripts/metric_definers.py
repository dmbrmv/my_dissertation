import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, root_mean_squared_error


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
    r = r_num / r_den
    # calculate error in spread of flow alpha
    alpha = np.std(targets, axis=0) / np.std(predictions)
    # calculate error in volume beta (bias of mean discharge)
    beta = np.sum(targets, axis=0) / np.sum(predictions)
    # calculate the Kling-Gupta Efficiency KGE
    kge_ = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return kge_, r, alpha, beta


def rmse(predictions, targets):
    return mean_squared_error(targets, predictions, squared=False)


def relative_error(predictions, targets):
    return np.mean(((targets - predictions) / targets) * 100)


def metric_df(gauge_id, predictions, targets):
    res_df = pd.DataFrame()

    res_df.loc[gauge_id, "NSE"] = nse(predictions, targets)

    res_df.loc[gauge_id, ["KGE", "r", "alpha", "beta"]] = kge(predictions, targets)

    res_df.loc[gauge_id, "RMSE"] = root_mean_squared_error(predictions, targets)

    res_df.loc[gauge_id, "delta"] = relative_error(predictions, targets)

    return res_df
