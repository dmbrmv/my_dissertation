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

test_periods = {
    "forward_1": Path(
        "data/lstm_configs/model_runs/FULL_cudalstm_forward_1.yml_1601_014943/config.yml"
    ),
    "forward_2": Path(
        "data/lstm_configs/model_runs/FULL_cudalstm_forward_2.yml_1601_033748/config.yml"
    ),
    "forward_3": Path(
        "data/lstm_configs/model_runs/FULL_cudalstm_forward_3.yml_1601_045804/config.yml"
    ),
    "back_1": Path(
        "data/lstm_configs/model_runs/FULL_cudalstm_back_1.yml_1601_055253/config.yml"
    ),
    "back_2": Path(
        "data/lstm_configs/model_runs/FULL_cudalstm_back_2.yml_1601_073830/config.yml"
    ),
    "back_3": Path(
        "data/lstm_configs/model_runs/FULL_cudalstm_back_3.yml_1601_085848/config.yml"
    ),
}

for test_period, test_period_cfg in test_periods.items():
    cfg = Config(test_period_cfg)
    epoch = best_epoch_finder(cfg.run_dir / "validation")
    tester = get_tester(cfg=cfg, run_dir=cfg.run_dir, period="test", init_model=True)
    pred_results = tester.evaluate(epoch=epoch, save_results=True)

    save_folder = Path("data/predictions/lstm_different_test_periods")
    save_folder.mkdir(parents=True, exist_ok=True)
    for gauge_id, results in pred_results.items():
        save_path = save_folder / gauge_id
        save_path.mkdir(parents=True, exist_ok=True)
        results["1D"]["xr"].to_dataframe().droplevel(1).rename(
            columns={"q_mm_day_obs": "q_obs", "q_mm_day_sim": "q_sim"}
        ).to_csv(
            save_path / f"{gauge_id}_{test_period}_predictions.csv"
        )