import gc
from pathlib import Path
import sys

from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config
import torch

sys.path.append("./")
from src.readers.geom_reader import load_geodata
from src.utils.logger import setup_logger

LOG = setup_logger(
    "lstm_train_simple", log_file="logs/lstm_train_simple.log", level="INFO"
)
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


test_periods = {
    "forward_1": {
        "train_start_date": "01/01/2008",
        "train_end_date": "31/12/2014",
        "validation_start_date": "01/01/2015",
        "validation_end_date": "31/12/2016",
        "test_start_date": "01/01/2017",
        "test_end_date": "31/12/2018",
    },
    "forward_2": {
        "train_start_date": "01/01/2008",
        "train_end_date": "31/12/2012",
        "validation_start_date": "01/01/2013",
        "validation_end_date": "31/12/2014",
        "test_start_date": "01/01/2015",
        "test_end_date": "31/12/2018",
    },
    "forward_3": {
        "train_start_date": "01/01/2008",
        "train_end_date": "31/12/2010",
        "validation_start_date": "01/01/2011",
        "validation_end_date": "31/12/2012",
        "test_start_date": "01/01/2013",
        "test_end_date": "31/12/2018",
    },
    "back_1": {
        "train_start_date": "01/01/2012",
        "train_end_date": "31/12/2018",
        "validation_start_date": "01/01/2010",
        "validation_end_date": "31/12/2011",
        "test_start_date": "01/01/2008",
        "test_end_date": "31/12/2009",
    },
    "back_2": {
        "train_start_date": "01/01/2014",
        "train_end_date": "31/12/2018",
        "validation_start_date": "01/01/2012",
        "validation_end_date": "31/12/2013",
        "test_start_date": "01/01/2008",
        "test_end_date": "31/12/2011",
    },
    "back_3": {
        "train_start_date": "01/01/2016",
        "train_end_date": "31/12/2018",
        "validation_start_date": "01/01/2014",
        "validation_end_date": "31/12/2015",
        "test_start_date": "01/01/2008",
        "test_end_date": "31/12/2013",
    },
}

hydro_target = "q_mm_day"


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

# Load watershed geometries and gauge locations
ws, gauges = load_geodata(folder_depth="./")
gauge_size = len(gauges)

model_folder = Path("data/models/full")
model_folder.mkdir(parents=True, exist_ok=True)

with open(model_folder / "full_gauges.txt", "w") as the_file:
    for gauge_name in gauges.index:
        the_file.write(f"{int(gauge_name)}\n")
# base model type
# [cudalstm, customlstm, ealstm, embcudalstm, mtslstm, gru, transformer]
# (has to match the if statement in modelzoo/__init__.py)
model_name = "cudalstm"
hidden_size = 256
seq_length = 365

for test_period, test_period_params in test_periods.items():
    cfg = Config(Path("data/models/lstm_config.yml"))
    cfg.update_config(
        yml_path_or_dict={
            # define storage and experiment
            "experiment_name": f"FULL_{model_name}_{test_period}.yml",
            "model": f"{model_name}",
            "run_dir": "data/lstm_configs/model_runs",
            "data_dir": "data",
            # define inner parameters
            "hidden_size": hidden_size,
            "static_attributes": static_parameters,
            "dynamic_inputs": ["prcp_e5l", "t_max_e5l", "t_min_e5l"],
            "target_variables": [hydro_target],
            "train_basin_file": "data/models/full/full_gauges.txt",
            "validate_n_random_basins": gauge_size,
            "validation_basin_file": "data/models/full/full_gauges.txt",
            "test_basin_file": "data/models/full/full_gauges.txt",
            # specify loss [MSE, NSE, RMSE]
            "loss": "NSE",
            # define time periodsj
            "seq_length": seq_length,
            # 'forecast_seq_length': 10,
            "train_start_date": test_period_params["train_start_date"],
            "train_end_date": test_period_params["train_end_date"],
            "validation_start_date": test_period_params["validation_start_date"],
            "validation_end_date": test_period_params["validation_end_date"],
            "test_start_date": test_period_params["test_start_date"],
            "test_end_date": test_period_params["test_end_date"],
        }
    )
    cfg.dump_config(
        folder=Path("data/lstm_configs/launch_configs"),
        filename=f"FULL_{model_name}_{test_period}.yml",
    )

    gc.collect()
    if torch.cuda.is_available():
        start_run(
            config_file=Path(
                f"data/lstm_configs/launch_configs/FULL_{model_name}_{test_period}.yml"
            )
        )
