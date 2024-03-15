from scripts.tft_data import open_for_tft, train_val_split
from scripts.model_eval import pred_res_builder
import glob
import geopandas as gpd
import pandas as pd
import torch

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
# era meteo input
meteo_input = ["prcp_e5l", "t_max_e5l", "t_min_e5l"]
# q_mm_day or lvl_mbs
hydro_target = "lvl_mbs"
# If we predict level we'll use height of gauge as extra static
if hydro_target == "lvl_mbs":
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
    ]  # 'height_bs'
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
# read geometry file with index for available gauges
ws_file = gpd.read_file("../geo_data/great_db/geometry/russia_ws.gpkg")
ws_file = ws_file.set_index("gauge_id")
###############################################################################
file = open_for_tft(
    nc_files=glob.glob(f"../geo_data/great_db/{nc_variable}/*.nc"),
    static_path="../geo_data/attributes/static_with_height.csv",
    area_index=ws_file.index,
    meteo_input=meteo_input,
    hydro_target=hydro_target,
    shuffle_static=False,
    with_static=True,
)

(train_ds, train_loader, val_ds, val_loader, val_df, _, _, _, scaler) = train_val_split(
    file, hydro_target=hydro_target, with_static=True
)
res = list()
# get checkpoint
counter = 0
for gauge in val_df.gauge_id.unique():
    g_df = pred_res_builder(
        gauge_id=gauge,
        res_storage="./result/tft_level_multi_256/",
        model_checkpoint="./lvl_prediction_multi_gauge/lightning_logs/version_0/checkpoints/epoch=3-step=63988.ckpt",
        hydro_target=hydro_target,
        meteo_input=meteo_input,
        static_parameters=static_parameters,
        val_ts_ds=val_ds,
        val_df=val_df,
        scaler=scaler,
        with_plot=False,
    )[0]
    res.append(g_df)

res = pd.concat(res)
res.to_csv("./result/tft_level_multi_256.csv", index=False)
