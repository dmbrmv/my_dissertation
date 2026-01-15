from pathlib import Path
import sys

from neuralhydrology.evaluation import get_tester
from neuralhydrology.utils.config import Config
import pandas as pd

sys.path.append("./")
from src.readers.geom_reader import load_geodata
from src.utils.logger import setup_logger

LOG = setup_logger("lstm_predict", log_file="logs/lstm_predict.log", level="INFO")

ws, gauges = load_geodata(folder_depth="./")
gauge_size = len(gauges)


def best_epoch_finder(validation_dir: Path) -> int:
    """Find the best epoch based on median NSE across validation metrics.

    Args:
        validation_dir: Path to the validation directory containing epoch subdirectories

    Returns:
        The epoch number with the highest median NSE
    """
    epoch_median_nse = {}

    # Iterate through all epoch directories
    for epoch_dir in sorted(validation_dir.glob("model_epoch*")):
        metrics_file = epoch_dir / "validation_metrics.csv"

        if not metrics_file.exists():
            continue

        # Read validation metrics CSV
        df = pd.read_csv(metrics_file)

        # Extract epoch number from directory name (e.g., "model_epoch030" -> 30)
        epoch_num = int(epoch_dir.name.split("model_epoch")[1])

        # Calculate median NSE across all basins
        median_nse = df["NSE"].median()
        epoch_median_nse[epoch_num] = median_nse

    if not epoch_median_nse:
        raise ValueError(f"No validation metrics found in {validation_dir}")

    # Return epoch with highest median NSE
    best_epoch = max(epoch_median_nse, key=lambda x: epoch_median_nse[x])

    return best_epoch


cfg_pathes = {
    "gpcp": {
        "path": Path(
            "data/lstm_configs/model_runs/FULL_cudalstm_q_mm_day_256_365_gpcp_1201_212718/config.yml"
        ),
        "epoch": 24,
    },
    "mswep": {
        "path": Path(
            "data/lstm_configs/model_runs/FULL_cudalstm_q_mm_day_256_365_mswep_1301_013021/config.yml"
        ),
        "epoch": 24,
    },
    "e5l": {
        "path": Path(
            "data/lstm_configs/model_runs/FULL_cudalstm_q_mm_day_256_365_e5l_1201_193434/config.yml"
        ),
        "epoch": 26,
    },
    "e5": {
        "path": Path(
            "data/lstm_configs/model_runs/FULL_cudalstm_q_mm_day_256_365_e5_1201_233905/config.yml"
        ),
        "epoch": 20,
    },
}

model_results = {}

for model in ["gpcp", "mswep", "e5l", "e5"]:
    LOG.info(f"Evaluating {model}...")
    lstm_cfg = cfg_pathes[model]["path"]
    cfg_run = Config(lstm_cfg)
    epoch = best_epoch_finder(cfg_run.run_dir / "validation")

    cfg_run.update_config(
        {
            "train_basin_file": "data/models/full/full_gauges.txt",
            "validate_n_random_basins": gauge_size,
            "validation_basin_file": "data/models/full/full_gauges.txt",
            "test_basin_file": "data/models/full/full_gauges.txt",
            "test_start_date": "01/01/2017",
            "test_end_date": "31/12/2018",
        }
    )
    tester = get_tester(
        cfg=cfg_run, run_dir=cfg_run.run_dir, period="test", init_model=True
    )
    pred_results = tester.evaluate(epoch=epoch, save_results=True)
    model_results[model] = pred_results

    save_folder = Path("data/predictions/lstm_poor_gauges")
    save_folder.mkdir(parents=True, exist_ok=True)
    for gauge_id, results in pred_results.items():
        save_path = save_folder / gauge_id
        save_path.mkdir(parents=True, exist_ok=True)
        results["1D"]["xr"].to_dataframe().droplevel(1).rename(
            columns={"q_mm_day_obs": "q_obs", "q_mm_day_sim": "q_sim"}
        ).to_csv(
            save_path / f"{gauge_id}_{model}_predictions.csv"
        )
