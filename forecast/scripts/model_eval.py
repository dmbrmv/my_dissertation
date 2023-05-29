from pytorch_forecasting import TimeSeriesDataSet
from sklearn.preprocessing import MinMaxScaler
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric
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


def nse(pred, target):
    denom = np.sum((target-pred)**2)
    divsr = np.sum((target-np.mean(target))**2)
    return 1-(denom/divsr)


def pred_res_builder(gauge_id: str,
                     hydro_target: str,
                     res_storage: str,
                     meteo_input: list, static_parameters: list,
                     model_checkpoint: str,
                     val_df: pd.Series,
                     scaler: MinMaxScaler,
                     val_ts_ds: TimeSeriesDataSet,
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
    compare_res['q_mm_day_pred'] = res

    with_pred = compare_res[['q_mm_day_pred',
                             *meteo_input, *static_parameters]]
    with_obs = compare_res[[hydro_target, *meteo_input, *static_parameters]]

    new_scaler = MinMaxScaler()

    new_scaler.min_, new_scaler.scale_ = scaler.min_, scaler.scale_

    # recalculate
    with_pred[['q_mm_day_pred',
               *meteo_input,
               *static_parameters]] = new_scaler.inverse_transform(with_pred)
    with_obs[[hydro_target,
              *meteo_input,
              *static_parameters]] = scaler.inverse_transform(with_obs)

    compare_res = compare_res[['date', 'q_mm_day', 'q_mm_day_pred']]
    compare_res['q_mm_day'] = with_obs[hydro_target]
    compare_res['q_mm_day_pred'] = with_pred['q_mm_day_pred']
    compare_res = compare_res.set_index('date')
    compare_res.to_csv(f'{p}/{gauge_id}.csv')

    # get nse
    pred_nse = nse(pred=compare_res['q_mm_day_pred'],
                   target=compare_res['q_mm_day'])
    if with_plot:
        compare_res.plot()
        # get nse
        plt.title(f'NSE для поста {gauge_id} = {pred_nse}')

    interpretation = best_tft.interpret_output(raw_prediction, reduction="sum")

    attnt, static, enc, dec = interpretation_for_gauge(
        interp_dict=interpretation,
        static_parameters=static_parameters,
        encoder_params=[*meteo_input, hydro_target],
        decoder_params=meteo_input)

    # res_df = pd.DataFrame(data={var: val for var, val in
    #                             zip(['gauge_id', 'NSE',
    #                                  'day', 'static', 'encoder', 'decoder'],
    #                                 [gauge_id, pred_nse,
    #                                  attnt, static, enc, dec])},
    #                       index=[0])
    res_df = pd.DataFrame(data={var: val for var, val in
                                zip(['gauge_id', 'NSE',
                                     'day', 'encoder', 'decoder'],
                                    [gauge_id, pred_nse,
                                    attnt, enc, dec])},
                          index=[0])
    static *= 100
    static['gauge_id'] = gauge_id
    static = static.set_index('gauge_id', drop=True)
    return res_df, interpretation, static


def interpretation_for_gauge(interp_dict: dict,
                             static_parameters: list,
                             encoder_params: list,
                             decoder_params: list):

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
    _, indices = interp_dict['attention'].sort(descending=True)
    indices = indices[0]+1
    # get most valuable static parameters
    static_worth = interp_df(interp_tensor=interp_dict['static_variables'],
                             df_columns=static_parameters)
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

    return int(indices), static_worth, enc_col, dec_col
