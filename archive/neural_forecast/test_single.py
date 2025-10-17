import glob
from pathlib import Path

import geopandas as gpd
import torch
from neuralhydrology.nh_run import eval_run
from neuralhydrology.utils.config import Config
from scripts.file_manipulator import train_rewriter

gpd.options.io_engine = "pyogrio"

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


def best_epoch_finder(log_file: Path) -> int:
    with open(f"{log_file}", "r") as f:
        lines = f.readlines()
    full_lines = [line for line in lines if ("NSE" in line) & ("Epoch" in line)]

    epoch_nse = {
        int(line.split(" Epoch ")[1].split(" ")[0]): float(line.split(" NSE: ")[1].split(",")[0])
        for line in full_lines
    }

    max_epoch = max(epoch_nse, key=epoch_nse.get)

    return max_epoch


ws_file = gpd.read_file(filename="../data/geometry/russia_ws.gpkg")
ws_file = ws_file.set_index("gauge_id")

available_gauges = list(i.stem for i in Path("../data/lstm_single_run").glob("*"))

# q_mm_day or lvl_sm
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
nc_variable = "nc_all_q"
# time series directory
ts_dir = Path("../data/time_series")
for gauge_id in available_gauges:
    configs = list(
        Path(f"../data/lstm_single_run/{gauge_id}").glob("cudalstm_q_mm_day*mswep*/config.yml")
    )
    configs_names = [i.parents[0].stem for i in configs]
    logs = list(Path(f"../data/lstm_single_run/{gauge_id}").glob("cudalstm_q_mm_day*mswep*/output.log"))
    try:
        best_epochs = list(best_epoch_finder(log) for log in logs)
    except ValueError:
        best_epochs = [24]

    meteo_inputs = [
        # ["prcp_gpcp", "t_max_e5l", "t_min_e5l"],
        ["prcp_mswep", "t_max_e5l", "t_min_e5l"],
        # ["prcp_e5", "t_max_e5", "t_min_e5"],
        # ["prcp_e5l", "t_max_e5l", "t_min_e5l"],
        # ["prcp_mswep", "t_max_e5", "t_min_e5"],
    ] * len(best_epochs)

    cfg_path, cfg_name, epoch, met_in = (
        configs[0],
        configs_names[0],
        best_epochs[0],
        meteo_inputs[0],
    )

    # write files for train procedure
    print(f"test data for {cfg_name}")
    if "mswep" in cfg_name:
        thresh = 1
    else:
        thresh = 0
    train_rewriter(
        era_paths=glob.glob(f"../data/ws_related_meteo/{nc_variable}/*.nc"),
        ts_dir=ts_dir,
        hydro_target=hydro_target,
        area_index=[gauge_id],
        predictors=[*met_in],
        possible_nans=thresh,
    )
    cfg = Config(cfg_path)
    eval_run(run_dir=cfg.run_dir, period="test", epoch=epoch)
