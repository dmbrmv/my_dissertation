from pytorch_forecasting import TimeSeriesDataSet
from sklearn.preprocessing import MinMaxScaler
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric
from sklearn.metrics import mean_squared_error
from pytorch_forecasting import TemporalFusionTransformer
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class nnse(MultiHorizonMetric):

    def loss(self, pred, target):

        pred = self.to_prediction(pred)
        denom = torch.sum((target-pred)**2)
        divsr = torch.sum((target - torch.mean(target)**2))
        nse = 1 - torch.div(denom, divsr)
        nnse = 1 / (2 - nse)

        return nnse


def nse(predictions, targets):
    return 1-(
        np.nansum((targets-predictions)**2)/np.nansum(
            (targets-np.nanmean(targets))**2))


def kge(predictions, targets):
    sim_mean = np.mean(targets, axis=0)
    obs_mean = np.mean(predictions)

    r_num = np.sum((targets - sim_mean) * (predictions - obs_mean),
                   axis=0)
    r_den = np.sqrt(np.sum((targets - sim_mean) ** 2,
                           axis=0) * np.sum((predictions - obs_mean) ** 2,))
    r = r_num / r_den
    # calculate error in spread of flow alpha
    alpha = np.std(targets, axis=0) / np.std(predictions)
    # calculate error in volume beta (bias of mean discharge)
    beta = (np.sum(targets, axis=0)
            / np.sum(predictions))
    # calculate the Kling-Gupta Efficiency KGE
    kge_ = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return kge_, r, alpha, beta


def rmse(predictions, targets):
    return mean_squared_error(targets, predictions, squared=False)


def relative_error(predictions, targets):
    return np.mean(((targets - predictions)/targets) * 100)


def metric_df(gauge_id, predictions, targets):

    res_df = pd.DataFrame()

    res_df.loc[gauge_id, 'NSE'] = nse(predictions,
                                      targets)

    res_df.loc[gauge_id, ['KGE', 'r', 'alpha', 'beta']] = kge(predictions,
                                                              targets)

    res_df.loc[gauge_id, 'RMSE'] = rmse(predictions,
                                        targets)

    res_df.loc[gauge_id, 'delta'] = relative_error(predictions,
                                                   targets)

    return res_df


def pred_res_builder(gauge_id: str,
                     hydro_target: str,
                     res_storage: str,
                     meteo_input: list, static_parameters: list,
                     model_checkpoint: str,
                     val_df: pd.Series,
                     scaler: MinMaxScaler,
                     val_ts_ds: TimeSeriesDataSet,
                     with_static: bool = True,
                     with_plot: bool = False):

    p = Path(f'{res_storage}')
    p.mkdir(exist_ok=True, parents=True)
    best_tft = TemporalFusionTransformer.load_from_checkpoint(model_checkpoint)
    raw_prediction, _, idx, _, _ = best_tft.predict(
        val_ts_ds.filter(lambda x: x.gauge_id == gauge_id),
        mode="raw",
        return_x=True,
        return_index=True)
    select_rule = (val_df.gauge_id == gauge_id) & \
        (val_df.time_idx.isin(idx['time_idx']))  # type: ignore

    idx.loc[:, 'date'] = val_df['date'][select_rule].values  # type: ignore

    res = np.array([])
    size_pred = len(raw_prediction['prediction'])  # type: ignore

    # type: ignore
    for i, prediction in enumerate(raw_prediction['prediction']):
        if i+6 == size_pred:
            break
        prediction = prediction.squeeze()  # type: ignore
        res = np.concatenate([res[:i], prediction.cpu()])
    # assemble rescaling procedure
    compare_res = val_df[val_df['gauge_id'] == gauge_id]
    compare_res = compare_res[[val in idx['time_idx'].values  # type: ignore
                               for val in compare_res['time_idx'].values]]
    compare_res[f'{hydro_target}_pred'] = res
    if with_static:
        with_pred = compare_res[[f'{hydro_target}_pred',
                                *meteo_input, *static_parameters]]
        with_obs = compare_res[[hydro_target,
                                *meteo_input, *static_parameters]]

        new_scaler = MinMaxScaler()

        new_scaler.min_, new_scaler.scale_ = scaler.min_, scaler.scale_

        # recalculate
        with_pred[[f'{hydro_target}_pred',
                   *meteo_input,
                   *static_parameters]] = new_scaler.inverse_transform(with_pred)
        with_obs[[hydro_target,
                  *meteo_input,
                  *static_parameters]] = scaler.inverse_transform(with_obs)
    else:
        with_pred = compare_res[[f'{hydro_target}_pred', *meteo_input]]
        with_obs = compare_res[[hydro_target, *meteo_input]]

        new_scaler = MinMaxScaler()

        new_scaler.min_, new_scaler.scale_ = scaler.min_, scaler.scale_

        # recalculate
        with_pred[[f'{hydro_target}_pred',
                   *meteo_input]] = new_scaler.inverse_transform(with_pred)
        with_obs[[hydro_target,
                  *meteo_input]] = scaler.inverse_transform(with_obs)

    compare_res = compare_res[['date', f'{hydro_target}',
                               f'{hydro_target}_pred']]
    compare_res[f'{hydro_target}'] = with_obs[hydro_target]
    compare_res[f'{hydro_target}_pred'] = with_pred[f'{hydro_target}_pred']
    compare_res = compare_res.set_index('date')
    compare_res.to_csv(f'{p}/{gauge_id}.csv')

    # get nse
    pred_nse = nse(predictions=compare_res[f'{hydro_target}_pred'],
                   targets=compare_res[f'{hydro_target}'])
    if with_plot:
        compare_res.plot()
        # get nse
        plt.title(f'NSE для поста {gauge_id} = {pred_nse}')

    interpretation = best_tft.interpret_output(raw_prediction, reduction="sum")

    if with_static:
        attnt, static, enc, dec = interpretation_for_gauge(
            interp_dict=interpretation,
            static_parameters=static_parameters,
            encoder_params=[*meteo_input, hydro_target],
            decoder_params=meteo_input)

        static *= 100
        static['gauge_id'] = gauge_id
        static = static.set_index('gauge_id', drop=True)

        attnt_df = pd.DataFrame(attnt.reshape(1, 365), index=[0])
        attnt_df['gauge_id'] = gauge_id
        attnt_df = attnt_df.set_index('gauge_id', drop=True)

        res_df = pd.DataFrame(data={var: val for var, val in
                                    zip(['gauge_id', 'NSE',
                                         'encoder', 'decoder'],
                                        [gauge_id, pred_nse, enc, dec])},
                              index=[0])

        return res_df, interpretation, static, attnt_df

    else:
        attnt, _, enc, dec = interpretation_for_gauge(
            interp_dict=interpretation,
            static_parameters=static_parameters,
            encoder_params=[*meteo_input, hydro_target],
            decoder_params=meteo_input)

        res_df = pd.DataFrame(data={var: val for var, val in
                                    zip(['gauge_id', 'NSE',
                                         'encoder', 'decoder'],
                                        [gauge_id, pred_nse, enc, dec])},
                              index=[0])

        attnt_df = pd.DataFrame(attnt.reshape(1, 365), index=[0])
        attnt_df['gauge_id'] = gauge_id
        attnt_df = attnt_df.set_index('gauge_id', drop=True)

        return res_df, attnt_df

    # res_df = pd.DataFrame(data={var: val for var, val in
    #                             zip(['gauge_id', 'NSE',
    #                                  'day', 'static', 'encoder', 'decoder'],
    #                                 [gauge_id, pred_nse,
    #                                  attnt, static, enc, dec])},
    #                       index=[0])


def interpretation_for_gauge(interp_dict: dict,
                             static_parameters: list,
                             encoder_params: list,
                             decoder_params: list,
                             with_static: bool = True):

    def to_percentage(values: torch.Tensor):
        values = values / values.sum(-1).unsqueeze(-1)
        return values

    def interp_df(interp_tensor: torch.Tensor,
                  df_columns: list):
        interp_tensor = to_percentage(interp_tensor)

        interp = {var: float(val) for var, val in zip(df_columns,
                                                      interp_tensor)}
        interp = pd.DataFrame(interp, index=[0])

        return interp

    # find most informative days
    _, attnt = interp_dict['attention'].sort(descending=True)
    attnt += 1
    # get most valuable static parameters
    if with_static:
        static_worth = interp_df(interp_tensor=interp_dict['static_variables'],
                                 df_columns=static_parameters)
    else:
        static_worth = None
    # stat_col, _ = (static_worth.idxmax(axis=1)[0],
    #                static_worth.max(axis=1)[0])
    # stat_col, _ = (list(static_worth.T.nlargest(n=4, columns=0).T.columns),
    #                list(static_worth.T.nlargest(n=4, columns=0).T.to_numpy()))
    # stat_col, stat_val = static_worth
    # get most valuable encoder parameters
    encoder_worth = interp_df(interp_tensor=interp_dict['encoder_variables'],
                              df_columns=encoder_params)
    enc_col, _ = (encoder_worth.idxmax(axis=1)[0],
                  encoder_worth.max(axis=1)[0])
    # get most valuable decoder parameters
    decoder_worth = interp_df(interp_tensor=interp_dict['decoder_variables'],
                              df_columns=decoder_params)
    dec_col, _ = (decoder_worth.idxmax(axis=1)[0],
                  decoder_worth.max(axis=1)[0])

    return attnt.cpu(), static_worth, enc_col, dec_col


def cmip_prediction(gauge_id: str,
                    hydro_target: str,
                    static_parameters: list,
                    meteo_input: list,
                    model_checkpoint: str,
                    train_dataset: TimeSeriesDataSet,
                    pred_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    scaler: MinMaxScaler,
                    path_to_result: str):

    p = Path(f'{path_to_result}')
    p.mkdir(exist_ok=True, parents=True)
    pred_ds = TimeSeriesDataSet.from_dataset(train_dataset,
                                             pred_df,
                                             scalers={'name': 'None'},
                                             predict=False,
                                             target_normalizer=None)
    best_tft = TemporalFusionTransformer.load_from_checkpoint(
        model_checkpoint)

    raw_prediction, _, _ = best_tft.predict(
        pred_ds.filter(lambda x: x.gauge_id == gauge_id),
        mode="raw",
        return_x=True,
        return_index=True)
    res = np.array([])
    size_pred = len(raw_prediction['prediction'])  # type: ignore

    for i, prediction in enumerate(raw_prediction['prediction']):
        if i+6 == size_pred:
            break
        prediction = prediction.squeeze()  # type: ignore
        res = np.concatenate([res[:i], prediction.cpu()])

    new_scaler = MinMaxScaler()

    new_scaler.min_, new_scaler.scale_ = (scaler.min_,
                                          scaler.scale_)

    test_df[[f'{hydro_target}',
             *meteo_input,
             *static_parameters]] = new_scaler.inverse_transform(
                 test_df[[f'{hydro_target}',
                          *meteo_input,
                          *static_parameters]])
    actual_df = test_df[test_df['gauge_id'] == gauge_id].set_index(
        'date').loc['2019-01-01':'2020-12-25', :]

    pred_df[[f'{hydro_target}',
             *meteo_input,
             *static_parameters]] = new_scaler.inverse_transform(
                 pred_df[[f'{hydro_target}',
                          *meteo_input,
                          *static_parameters]])
    predicted_df = pred_df[pred_df['gauge_id'] == gauge_id].set_index(
        'date').loc['2019-01-01':'2020-12-25', :]

    predicted_df['q_mm_day'] = res[:725]

    res_df = pd.DataFrame()
    res_df['date'] = pd.date_range(start='2019-01-01', end='2030-12-25')
    res_df = res_df.set_index('date')
    res_df['pred'] = predicted_df['q_mm_day']
    res_df['obs'] = actual_df['q_mm_day']

    res_df.to_csv(f'{p}/{gauge_id}.csv')

    return res_df
