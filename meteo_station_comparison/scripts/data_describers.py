import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, root_mean_squared_error


def mae_percent(x, y):
    mae_series = (x - y) / x
    mae_series.replace([np.inf, -np.inf], np.nan, inplace=True)
    mae_series = np.abs(mae_series)
    return mae_series.mean() * 1e2


def station_describer(
    df, frequency: str, prcp_station: str, prcp_meteo: str, modifier: str = ""
):
    df = df[[f"{prcp_station}", f"{prcp_meteo}"]].dropna()
    df_len = len(df)
    if df.empty:
        return {
            "df_len": df_len,
            f"{modifier}_r2": np.nan,
            f"{modifier}_pearson": np.nan,
            f"{modifier}_mae": np.nan,
            f"{modifier}_rmse": np.nan,
            f"{modifier}_mean": np.nan,
        }
    elif df_len < 1000:
        return {
            "df_len": df_len,
            f"{modifier}_r2": np.nan,
            f"{modifier}_pearson": np.nan,
            f"{modifier}_mae": np.nan,
            f"{modifier}_rmse": np.nan,
            f"{modifier}_mean": np.nan,
        }
    else:
        df = df.groupby(pd.Grouper(freq=frequency)).sum()
        if df[f"{prcp_meteo}"].sum() == 0:
            return {
                "df_len": df_len,
                f"{modifier}_r2": np.nan,
                f"{modifier}_pearson": np.nan,
                f"{modifier}_mae": np.nan,
                f"{modifier}_rmse": np.nan,
                f"{modifier}_mean": np.nan,
            }
        else:
            r2_res = r2_score(y_true=df[f"{prcp_station}"], y_pred=df[f"{prcp_meteo}"])
            pearson_res = pearsonr(x=df[f"{prcp_station}"], y=df[f"{prcp_meteo}"])[0]
            mae_res = mae_percent(x=df[f"{prcp_station}"], y=df[f"{prcp_meteo}"])
            rmse_res = root_mean_squared_error(
                y_true=df[f"{prcp_station}"], y_pred=df[f"{prcp_meteo}"]
            )
            mean_res = (df[f"{prcp_station}"] - df[f"{prcp_meteo}"]).mean()

            return {
                "df_len": df_len,
                f"{modifier}_r2": r2_res,
                f"{modifier}_pearson": pearson_res,
                f"{modifier}_mae": mae_res,
                f"{modifier}_rmse": rmse_res,
                f"{modifier}_mean": mean_res,
            }
