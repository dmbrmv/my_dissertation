import logging
import pathlib
import warnings
from copy import deepcopy
from functools import reduce
import gc

import numpy as np
import pandas as pd
import torch
import xarray as xr
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer
from darts.models import TFTModel
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from scripts.metric_definers import metric_df
from scripts.tft_data_creators import (
    covariate_creator,
    scale_with_static,
    target_creator,
    type32_converter,
)

warnings.filterwarnings("ignore", ".*does not have many workers.*")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
torch.set_float32_matmul_precision("medium")


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

gauges = list(i.stem for i in pathlib.Path("./model/single_gauge").glob("./*"))

input_chunk_length = 365
output_chunk_length = 7

train_start = "2008-01-01"
train_end = "2016-12-31"

val_start = pd.to_datetime("2017-01-01")
val_end = pd.to_datetime("2018-12-31")

pred_start = pd.to_datetime("2019-01-01") - pd.to_timedelta(f"{input_chunk_length}D")
pred_end = "2020-12-31"

pred_start = pd.to_datetime("2019-01-01") - pd.to_timedelta(f"{input_chunk_length}D")
pred_end = "2020-12-31"

with_static = True
index_col = "gauge_id"
static_path = "../geo_data/attributes/static_with_height.csv"
nc_files = list(
    pathlib.Path(f"../geo_data/ws_related_meteo/{nc_variable}").glob("./*.nc")
)

if with_static:
    static_attributes = pd.read_csv(static_path, index_col=index_col)
    static_attributes.index = static_attributes.index.astype(str)
    static_attributes = static_attributes.replace(np.NaN, 0.0)
else:
    static_attributes = None


tft_single_gauge_stat = list()

for gauge_id in tqdm(gauges, desc="I'm Scatman !"):
    tft_model = TFTModel.load_from_checkpoint(
        model_name="q_mm_day_hs256_bs256_do042",
        work_dir=f"./model/single_gauge/{gauge_id}",
    )

    if isinstance(static_attributes, pd.DataFrame):
        static_attributes = static_attributes[[*static_parameters]]
        gauge_static = static_attributes.loc[[gauge_id], :]
    else:
        gauge_static = None
    with xr.open_dataset(
        f"../geo_data/ws_related_meteo/{nc_variable}/{gauge_id}.nc"
    ) as f:
        gauge_file = f.to_dataframe()
    gauge_file = gauge_file[[*meteo_input, hydro_target, "gauge_id"]]

    if isinstance(gauge_static, pd.DataFrame):
        for col in gauge_static.columns:
            gauge_file[col] = gauge_static.loc[gauge_id, col]

    past_df = gauge_file[:train_end].reset_index(drop=False)
    future_df = gauge_file[:val_end].reset_index(drop=False)
    val_past_df = gauge_file[val_start:val_end].reset_index(drop=False)
    val_future_df = gauge_file[val_start:val_end].reset_index(drop=False)
    pred_df = gauge_file[train_start:pred_end].reset_index(drop=False)

    # create timeseries and scaler
    train_target_scaler, train_cov_scaler, train_static_scaler = (
        Scaler(scaler=StandardScaler()),
        Scaler(scaler=StandardScaler()),
        StaticCovariatesTransformer(),
    )
    # train
    train_target = target_creator(
        data_frame=past_df,
        target_input=hydro_target,
        static_parameters=static_parameters,
    )
    past_cov = covariate_creator(
        data_frame=past_df, static_parameters=static_parameters, meteo_input=meteo_input
    )
    future_cov = covariate_creator(
        data_frame=future_df,
        static_parameters=static_parameters,
        meteo_input=meteo_input,
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
        data_frame=val_past_df,
        static_parameters=static_parameters,
        meteo_input=meteo_input,
    )
    val_future_cov = covariate_creator(
        data_frame=val_future_df,
        static_parameters=static_parameters,
        meteo_input=meteo_input,
    )
    # pred
    # pred_target = target_creator(
    #     data_frame=pred_df,
    #     target_input=hydro_target,
    #     static_parameters=static_parameters,
    # )
    # pred_cov = covariate_creator(
    #     data_frame=pred_df, static_parameters=static_parameters, meteo_input=meteo_input
    # )

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
        series=val_future_cov,
        scaler=train_cov_scaler,
        static_scaler=train_static_scaler,
    )
    # pred
    # pred_target = scale_with_static(
    #     series=pred_target,
    #     scaler=train_target_scaler,
    #     static_scaler=train_static_scaler,
    # )
    # pred_cov = scale_with_static(
    #     series=pred_cov, scaler=train_cov_scaler, static_scaler=train_static_scaler
    # )


    pred_len = pd.date_range(start=pred_start, end=pred_end).__len__()

    test_pred = list()

    target_index_col = pred_df.columns.get_loc(f"{hydro_target}")

    target_obs = deepcopy(pred_df).iloc[
        -(pred_len - input_chunk_length) :, target_index_col
    ]

    for i in range(0, (pred_len - input_chunk_length - output_chunk_length) + 1):
        slice_limit = -pred_len + input_chunk_length + output_chunk_length + i

        if slice_limit == 0:
            pred_step_df = pred_df.iloc[:, :]
        else:
            pred_step_df = pred_df.iloc[:slice_limit, :]
        # pred
        pred_target = scale_with_static(
            series=target_creator(
                # data_frame=pred_df.iloc[: (-pred_len + input_chunk_length + i), :],
                data_frame=pred_step_df.iloc[:(-output_chunk_length), :],
                target_input=hydro_target,
                static_parameters=static_parameters,
            ),
            scaler=train_target_scaler,
            static_scaler=train_static_scaler,
        )
        pred_cov = scale_with_static(
            series=covariate_creator(
                # data_frame=pred_df.iloc[
                #     : (-pred_len + input_chunk_length + i), :
                # ],
                data_frame=pred_step_df.iloc[:(-output_chunk_length), :],
                static_parameters=static_parameters,
                meteo_input=meteo_input,
            ),
            scaler=train_cov_scaler,
            static_scaler=train_static_scaler,
        )

        pred_cov_future = scale_with_static(
            series=covariate_creator(
                data_frame=pred_step_df,
                static_parameters=static_parameters,
                meteo_input=meteo_input,
            ),
            scaler=train_cov_scaler,
            static_scaler=train_static_scaler,
        )

        some_pred = tft_model.predict(
            n=output_chunk_length,
            series=pred_target,
            past_covariates=pred_cov,
            future_covariates=pred_cov_future,
            verbose=False,
            num_loader_workers=15,
            num_samples=100
        )

        temp_df = pd.DataFrame()
        temp_df["date"] = some_pred[0].time_index
        temp_df = temp_df.set_index("date")

        temp_df[f"{hydro_target}"] = (
            train_target_scaler.inverse_transform(some_pred[0]).values().squeeze()
        )
        test_pred.append(temp_df)

        if slice_limit == 0:
            pred_df.iloc[
                -(pred_len - input_chunk_length - i) :,
                target_index_col,
            ] = train_target_scaler.inverse_transform(some_pred[0]).values().squeeze()
        else:
            pred_df.iloc[
                -(pred_len - input_chunk_length - i) : -(
                    pred_len - input_chunk_length - output_chunk_length - i
                ),
                target_index_col,
            ] = train_target_scaler.inverse_transform(some_pred[0]).values().squeeze()
    pred_df_multi = reduce(lambda df1, df2: df2.combine_first(df1), test_pred)
    pred_df_multi = pred_df_multi.rename(columns={f"{hydro_target}": "pred"})
    pred_df_multi["obs"] = target_obs.values

    metric_res_df = metric_df(
        gauge_id=gauge_id, predictions=pred_df_multi["pred"], targets=pred_df_multi["obs"]
    )
    metric_res_df.index.name = "gauge_id"
    metric_res_df.to_csv(f"./res/single_predict/{gauge_id}.csv")

    tft_single_gauge_stat.append(metric_res_df)    
    
    tft_model = None
    gc.collect()
    torch.cuda.empty_cache()

tft_single_gauge_stat = pd.concat(tft_single_gauge_stat)
tft_single_gauge_stat.to_csv("./single_gauge_predict.csv")
