"""Run gr4j calibration for single gauge."""

import pathlib
import sys

sys.path.append("/Users/dmbrmv/Development/ESG/my_dissertation")

import numpy as np
import spotpy
import xarray as xr
from spotpy.objectivefunctions import kge, lognashsutcliffe, mae, nashsutcliffe, rmse

from conceptual_runs.scripts.logger import logging

from .model_setups import gr4j_setup

# Configure logging

logger = logging.getLogger(pathlib.Path(__file__).name)

gauges = [i.stem for i in pathlib.Path("/Users/dmbrmv/Development/data/great_db/nc_all_q").glob("*.nc")]

calibration_path = pathlib.Path("./gr4j_calibration_mle_logNSE")
calibration_path.mkdir(exist_ok=True, parents=True)
gauges = [i for i in gauges if i not in [i.stem for i in calibration_path.glob("*.npy")]]


def gr4j_single_core(g_id: str) -> None:
    """GR4J model implementation for one process.

    Args:
    ----
        g_id (str) : name of the gauge to use

    Returns:
    -------
        None

    """
    # global exit_flag
    try:
        logging.info(f"Processing gauge ID: {g_id}")
        with xr.open_dataset(f"/Users/dmbrmv/Development/data/great_db/nc_all_q/{g_id}.nc") as f:
            example_df = f.to_pandas()
            example_df = example_df.drop("gauge_id", axis=1)
            train_df = example_df[:"2018-12-31"]

        sampler = spotpy.algorithms.mle(
            gr4j_setup(data_file=train_df, obj_func=lognashsutcliffe),  # type: ignore
            dbname=f"{calibration_path}/{g_id}",
            dbformat="csv",
            random_state=42,
            optimization_direction="maximize",
        )
        sampler.sample(6000)

        gauge_results = spotpy.analyser.load_csv_results(f"{calibration_path}/{g_id}")
        best_gr4j_params = np.array(spotpy.analyser.get_best_parameterset(gauge_results))
        with open(f"{calibration_path}/{g_id}.npy", "wb") as f:
            np.save(file=f, arr=best_gr4j_params)
        with pathlib.Path(f"{calibration_path}/{g_id}.csv") as f:
            f.unlink()

    except EOFError as e:
        # exit_flag.value = True
        logger.error(f"EOFError occurred while reading data for gauge {g_id}: {e}")
    except (ValueError, RuntimeWarning) as e:
        # exit_flag.value = True
        logger.error(f"Exception occurred while calibrating the {g_id}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    return None
