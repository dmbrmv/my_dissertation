import glob
import torch
import geopandas as gpd
from neuralhydrology.nh_run import start_run
from scripts.file_manipulator import file_rewriter
from pathlib import Path

import random

random.seed(42)
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

ts_dir = Path("../geo_data/time_series")

# gpcp
file_rewriter(
    q_pathes=glob.glob("../geo_data/great_db/nc_all_q/*.nc"),
    ts_dir=ts_dir,
    hydro_target="q_cms_s",
    meteo_predictors=["t_max_e5", "t_min_e5", "prcp_gpcp"],
)
if torch.cuda.is_available():
    start_run(config_file=Path("./configs/gpcp_prcp_qms.yml"))
# imerg
file_rewriter(
    q_pathes=glob.glob("../geo_data/great_db/nc_all_q/*.nc"),
    ts_dir=ts_dir,
    hydro_target="q_cms_s",
    meteo_predictors=["t_max_e5", "t_min_e5", "prcp_imerg"],
)
if torch.cuda.is_available():
    start_run(config_file=Path("./configs/imerg_prcp_qms.yml"))
# mswep
file_rewriter(
    q_pathes=glob.glob("../geo_data/great_db/nc_all_q/*.nc"),
    ts_dir=ts_dir,
    hydro_target="q_cms_s",
    meteo_predictors=["t_max_e5", "t_min_e5", "prcp_mswep"],
)
if torch.cuda.is_available():
    start_run(config_file=Path("./configs/mswep_prcp_qms.yml"))
