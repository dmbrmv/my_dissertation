import gc
import glob
from pathlib import Path

import geopandas as gpd
import torch
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config
from scripts.file_manipulator import train_rewriter

# setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

gpd.options.io_engine = "pyogrio"

# Additional Info when using cuda
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print("Memory Usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
    print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")

# era_input = ["prcp_e5l", "t_max_e5l", "t_min_e5l"]
era_input = ["prcp_mswep", "t_max_e5l", "t_min_e5l"]
# q_mm_day or lvl_sm
hydro_target = "q_mm_day"
q_h_relation = False

if hydro_target == "lvl_sm":
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
        "ws_area",
        "ele_mt_sav",
        "height_bs",
    ]
    nc_variable = "nc_all_h"
    if q_h_relation:
        nc_variable = "nc_all_q_h"
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
        "ws_area",
        "ele_mt_sav",
    ]
    nc_variable = "nc_all_q"

ws_file = gpd.read_file(filename="../geo_data/geometry/russia_ws.gpkg")
ws_file = ws_file.set_index("gauge_id")

# time series directory
ts_dir = Path("../geo_data/time_series")
ts_dir.mkdir(exist_ok=True, parents=True)
# write files for train procedure
print(f"train data for {hydro_target} with {nc_variable} initial data")

trained_gauges = list(
    str(i).split("_")[-3] for i in Path("../geo_data/lstm_single_cfg/").glob("*.yml")
)
for gauge_id in ws_file.index:
    if gauge_id in trained_gauges:
        continue
    train_rewriter(
        era_paths=glob.glob(f"../geo_data/ws_related_meteo/{nc_variable}/*.nc"),
        ts_dir=ts_dir,
        hydro_target=hydro_target,
        area_index=[gauge_id],
        predictors=[*era_input],
        possible_nans=1,
    )

    # define variables require to perform hindcast
    gauges = [file.split("/")[-1][:-3] for file in glob.glob(f"{ts_dir}/*.nc")]
    gauge_size = len(gauges)
    if gauge_size == 0:
        continue
    else:
        cfg = Config(Path("./model_config.yml"))
        # base model type
        # [cudalstm, customlstm, ealstm, embcudalstm, mtslstm, gru, transformer]
        # (has to match the if statement in modelzoo/__init__.py)
        model_name = "cudalstm"
        cfg.update_config(
            yml_path_or_dict={
                # define storage and experiment
                "experiment_name": f"{model_name}_{hydro_target}_{gauge_id}_mswep_mo_static",
                "model": f"{model_name}",
                "run_dir": f"../geo_data/lstm_single_run/{gauge_id}",
                "data_dir": "../geo_data/",
                "device": f"{device}:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cpu",
                # define inner parameters
                # "static_attributes": static_parameters,
                "dynamic_inputs": [*era_input],
                # 'hindcast_inputs': era_input,
                # 'forecast_inputs': era_input,
                "target_variables": [hydro_target],
                # 'dynamics_embedding': {'type': 'fc', 'hiddens': [128, 64, 256],
                #                       'activation': 'tanh', 'dropout': 0.2},
                # 'statics_embedding': {'type': 'fc', 'hiddens': [128, 64, 256],
                #                      'activation': 'tanh', 'dropout': 0.2},
                # define files with gauge data
                "train_basin_file": "./every_basin.txt",
                "validate_n_random_basins": gauge_size,
                "validation_basin_file": "./every_basin.txt",
                "test_basin_file": "./every_basin.txt",
                "loss": "NSE",
                # define time periods
                "seq_length": 365,
                "hidden_size": 256,
                # 'forecast_seq_length': 10,
                "train_start_date": "01/01/2008",
                "train_end_date": "31/12/2016",
                "validation_start_date": "01/01/2017",
                "validation_end_date": "31/12/2018",
                "test_start_date": "01/01/2019",
                "test_end_date": "31/12/2020",
            }
        )
        cfg.dump_config(
            folder=Path("../geo_data/lstm_single_cfg"),
            filename=f"{model_name}_{hydro_target}_{gauge_id}_mswep_static.yml",
        )

        gc.collect()

        start_run(
            config_file=Path(
                f"../geo_data/lstm_single_cfg/{model_name}_{hydro_target}_{gauge_id}_mswep_static.yml"
            )
        )