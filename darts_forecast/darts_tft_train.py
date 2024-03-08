import logging
import glob
from typing import Union, List
import geopandas as gpd
import xarray as xr
import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, Callback

from darts import TimeSeries
from darts.models import TFTModel

from darts.utils.likelihood_models import QuantileRegression
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer

from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore")

logging.disable(logging.CRITICAL)

# setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

# Additional Info when using cuda
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print("Memory Usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
    print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")


meteo_input = ["prcp_e5l", "t_max_e5l", "t_min_e5l"]
# q_mm_day or lvl_sm
hydro_target = "q_mm_day"

if (hydro_target == "lvl_mbs") | (hydro_target == "lvl_sm"):
    static_parameters = [
        "for_pc_sse",
        "crp_pc_sse",
        "inu_pc_ult",
        "ire_pc_sse",
        "lka_pc_use",
        "prm_pc_sse",
        "pst_pc_sse",
        "cly_pc_sav",
        "slt_pc_sav",
        "snd_pc_sav",
        "kar_pc_sse",
        "urb_pc_sse",
        "gwt_cm_sav",
        "lkv_mc_usu",
        "rev_mc_usu",
        "sgr_dk_sav",
        "slp_dg_sav",
        "ws_area",
        "ele_mt_sav",
        "height_bs",
    ]
    nc_variable = "nc_all_h"
else:
    static_parameters = [
        "for_pc_sse",
        "crp_pc_sse",
        "inu_pc_ult",
        "ire_pc_sse",
        "lka_pc_use",
        "prm_pc_sse",
        "pst_pc_sse",
        "cly_pc_sav",
        "slt_pc_sav",
        "snd_pc_sav",
        "kar_pc_sse",
        "urb_pc_sse",
        "gwt_cm_sav",
        "lkv_mc_usu",
        "rev_mc_usu",
        "sgr_dk_sav",
        "slp_dg_sav",
        "ws_area",
        "ele_mt_sav",
    ]
    nc_variable = "nc_all_q"

ws_file = gpd.read_file("../geo_data/geometry/russia_ws.gpkg")
ws_file = ws_file.set_index("gauge_id")

with_static = True
index_col = "gauge_id"
static_path = "../geo_data/attributes/static_with_height.csv"
nc_files: list[str] = glob.glob(
    pathname=f"../geo_data/ws_related_meteo/{nc_variable}/*.nc"
)
area_index = ws_file.index


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


if with_static:
    static_attributes = pd.read_csv(static_path, index_col=index_col)
    static_attributes.index = static_attributes.index.astype(str)
    static_attributes = static_attributes.replace(np.NaN, 0.0)
else:
    static_attributes = None

input_chunk_length = 365
output_chunk_length = 7

train_start = "2008-01-01"
train_end = "2016-12-31"

val_start = pd.to_datetime("2017-01-01")
val_end = pd.to_datetime("2018-12-31")

pred_start = pd.to_datetime("2019-01-01") - pd.to_timedelta(f"{input_chunk_length}D")
pred_end = "2020-12-31"


# past_cov_start, past_cov_end = (pd.to_datetime(arg=train_start), pd.to_datetime(arg=train_end))
# fut_cov_start, fut_cov_end = (pd.to_datetime(arg=train_start),
#                               pd.to_datetime(arg=test_end) - pd.to_timedelta(arg=f"{output_chunk_length}D"))

past_list = list()
future_list = list()
val_past_list = list()
val_future_list = list()
pred_list = list()

file_counter = 0

for file_path in nc_files:
    gauge_id = file_path.split("/")[-1][:-3]
    try:
        if isinstance(static_attributes, pd.DataFrame):
            static_attributes = static_attributes[[*static_parameters]]
            gauge_static = static_attributes.loc[[gauge_id], :]
        else:
            gauge_static = None
    except KeyError:
        print(f"No data for {gauge_id} !")
        file_counter -= 1
        continue
    file = xr.open_dataset(filename_or_obj=file_path).to_dataframe()
    file = file[[*meteo_input, hydro_target, "gauge_id"]]
    if gauge_id not in area_index:
        file_counter -= 1
        continue
    if file.isna().any().any():
        file_counter -= 1
        continue
    if isinstance(gauge_static, pd.DataFrame):
        for col in gauge_static.columns:
            file[col] = gauge_static.loc[gauge_id, col]
    past_list.append(file[:train_end])
    future_list.append(file[:val_end])
    val_past_list.append(file[val_start:val_end])
    val_future_list.append(file[val_start:val_end])
    pred_list.append(file[pred_start:pred_end])

    # file_counter += 1
    # if file_counter == 690:
    #     break

    # pred_slice = file[future_pred_cov_start:future_pred_cov_end]
past_df = pd.concat(past_list, axis=0).reset_index(drop=False)
del past_list
future_df = pd.concat(future_list, axis=0).reset_index(drop=False)
del future_list
val_past_df = pd.concat(val_past_list, axis=0).reset_index(drop=False)
val_future_df = pd.concat(val_future_list, axis=0).reset_index(drop=False)
# plot_val_df = deepcopy(val_df)
# val_df.loc[:, "q_mm_day"] = 0.
del val_past_list
del val_future_list
pred_df = pd.concat(pred_list, axis=0).reset_index(drop=False)
del pred_list


# create timeseries and scaler
train_target_scaler, train_cov_scaler, train_static_scaler = (
    Scaler(scaler=StandardScaler()),
    Scaler(scaler=StandardScaler()),
    StaticCovariatesTransformer(),
)
# train
train_target = target_creator(
    data_frame=past_df, target_input=hydro_target, static_parameters=static_parameters
)
past_cov = covariate_creator(
    data_frame=past_df, static_parameters=static_parameters, meteo_input=meteo_input
)
future_cov = covariate_creator(
    data_frame=future_df, static_parameters=static_parameters, meteo_input=meteo_input
)
# val
val_target = target_creator(
    data_frame=val_past_df,
    target_input=hydro_target,
    static_parameters=static_parameters,
)
# val_plot = target_creator(
#     data_frame=plot_val_df, target_input=hydro_target, static_parameters=static_parameters)
val_past_cov = covariate_creator(
    data_frame=val_past_df, static_parameters=static_parameters, meteo_input=meteo_input
)
val_future_cov = covariate_creator(
    data_frame=val_future_df,
    static_parameters=static_parameters,
    meteo_input=meteo_input,
)
# pred
pred_target = target_creator(
    data_frame=pred_df, target_input=hydro_target, static_parameters=static_parameters
)
pred_cov = covariate_creator(
    data_frame=pred_df, static_parameters=static_parameters, meteo_input=meteo_input
)


def type32_converter(series_list: TimeSeries | list[TimeSeries]) -> list[TimeSeries]:
    return [s.astype(dtype="float32") for s in series_list]


def scale_with_static(
    series: TimeSeries | list[TimeSeries],
    scaler: Scaler,
    static_scaler: StaticCovariatesTransformer,
) -> list[TimeSeries]:
    series = scaler.transform(series=series)
    series = static_scaler.transform(series=series)

    return type32_converter(series_list=series)


# scale target
train_target = train_target_scaler.fit_transform(train_target)
train_target = train_static_scaler.fit_transform(train_target)
train_target = type32_converter(train_target)
# scale covariate
past_cov = train_cov_scaler.fit_transform(past_cov)
past_cov = train_static_scaler.transform(past_cov)
past_cov = type32_converter(past_cov)
# prepare new data
future_cov = scale_with_static(
    series=future_cov, scaler=train_cov_scaler, static_scaler=train_static_scaler
)
# val
val_target = scale_with_static(
    series=val_target, scaler=train_target_scaler, static_scaler=train_static_scaler
)
val_past_cov = scale_with_static(
    series=val_past_cov, scaler=train_cov_scaler, static_scaler=train_static_scaler
)
val_future_cov = scale_with_static(
    series=val_future_cov, scaler=train_cov_scaler, static_scaler=train_static_scaler
)
# pred
pred_target = scale_with_static(
    series=pred_target, scaler=train_target_scaler, static_scaler=train_static_scaler
)
pred_cov = scale_with_static(
    series=pred_cov, scaler=train_cov_scaler, static_scaler=train_static_scaler
)


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


loss_logger = LossLogger()
# early stop callback
my_stopper = EarlyStopping(
    monitor="val_loss",  # "val_MeanAbsolutePercentageError"
    patience=6,
    min_delta=0.1,
    mode="min",
)

pl_trainer_kwargs = {
    "accelerator": "gpu",
    "devices": [0],
    "callbacks": [my_stopper, loss_logger],
    "enable_progress_bar": True,
}
# default quantiles for QuantileRegression
quantiles = [0.01, 0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95, 0.99]

my_model = TFTModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=output_chunk_length,
    hidden_size=256,
    lstm_layers=1,
    num_attention_heads=4,
    dropout=0.42,
    batch_size=128,
    n_epochs=20,
    use_static_covariates=True,
    add_relative_index=False,
    add_encoders=None,
    # QuantileRegression is set per default
    likelihood=QuantileRegression(
        quantiles=quantiles),
    # likelihood=None,
    # loss_fn=torch.nn.MSELoss(),
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs={"lr": 1e-3},
    lr_scheduler_cls=torch.optim.lr_scheduler.StepLR,
    lr_scheduler_kwargs={"step_size": 5, "gamma": 1e-1},
    pl_trainer_kwargs=pl_trainer_kwargs,
    random_state=42,
    model_name="q_mm_day_hs256_bs128_do042",
    # model_name="test_tft",
    work_dir="./model",
    save_checkpoints=True,
    force_reset=True,
)

my_model.fit(
    series=train_target,
    past_covariates=past_cov,
    future_covariates=future_cov,
    val_series=val_target,
    val_past_covariates=val_past_cov,
    val_future_covariates=val_future_cov,
    verbose=True,
    num_loader_workers=4,
)
