from typing import List, Union

import pandas as pd


from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback



class LossLogger(Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    # will automatically be called at the end of each epoch
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.train_loss.append(float(trainer.callback_metrics["train_loss"]))

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.val_loss.append(float(trainer.callback_metrics["val_loss"]))


def covariate_creator(
    data_frame: pd.DataFrame, static_parameters: list, meteo_input: list
) -> Union[TimeSeries, List[TimeSeries]]:
    cov_time_series = TimeSeries.from_group_dataframe(
        df=data_frame,
        time_col="date",
        group_cols="gauge_id",
        static_cols=static_parameters,
        value_cols=meteo_input,
    )

    return cov_time_series


def target_creator(
    data_frame: pd.DataFrame,
    target_input: str,
    static_parameters: list,
) -> Union[TimeSeries, List[TimeSeries]]:
    target_time_series = TimeSeries.from_group_dataframe(
        df=data_frame,
        time_col="date",
        group_cols="gauge_id",
        static_cols=static_parameters,
        value_cols=target_input,
    )

    return target_time_series


def type32_converter(
        series_list: TimeSeries | list[TimeSeries],
    ) -> list[TimeSeries]:
        return [s.astype(dtype="float32") for s in series_list]


def scale_with_static(
        series: TimeSeries | list[TimeSeries],
        scaler: Scaler,
        static_scaler: StaticCovariatesTransformer,
    ) -> list[TimeSeries]:
        series = scaler.transform(series=series)
        series = static_scaler.transform(series=series)

        return type32_converter(series_list=series)