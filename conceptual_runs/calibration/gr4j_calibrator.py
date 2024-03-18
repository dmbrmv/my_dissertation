"""
Run gr4j calibration for single gauge
"""

import logging
import pathlib

import numpy as np
import spotpy
import xarray as xr
from spotpy.objectivefunctions import kge

from .model_setups import gr4j_setup

# Configure logging

logging.basicConfig(filename="gr4j_log.log", level=logging.INFO)

gauges = [
    i.stem
    for i in pathlib.Path("/Users/dmbrmv/Development/geo_data/great_db/nc_all_q").glob(
        "*.nc"
    )
]
gauges = [
    i
    for i in gauges
    if i not in [i.stem for i in pathlib.Path("./gr4j_calibrated").glob("*.npy")]
]


def gr4j_single_core(g_id: str) -> None:
    """

    Parameters
    ----------
    gauge_id (str) : name of the gauge to use

    Returns
    -------
    None
    """
    # global exit_flag
    try:
        logging.info(f"Processing gauge ID: {g_id}")
        with xr.open_dataset(
            f"/Users/dmbrmv/Development/geo_data/great_db/nc_all_q/{g_id}.nc"
        ) as f:
            example_df = f.to_pandas()
            example_df = example_df.drop("gauge_id", axis=1)
            train_df = example_df[:"2018-12-31"]

        gr4j_calibrated = pathlib.Path("./gr4j_calibrated")
        gr4j_calibrated.mkdir(exist_ok=True, parents=True)
        sampler = spotpy.algorithms.mle(
            gr4j_setup(data_file=train_df, obj_func=kge),
            dbname=f"{gr4j_calibrated}/{g_id}",
            dbformat="csv",
            random_state=42,
        )
        sampler.sample(6000)

        gauge_results = spotpy.analyser.load_csv_results(f"{gr4j_calibrated}/{g_id}")
        best_gr4j_params = np.array(
            spotpy.analyser.get_best_parameterset(gauge_results)
        )
        with open(f"{gr4j_calibrated}/{g_id}.npy", "wb") as f:
            np.save(file=f, arr=best_gr4j_params)

        with pathlib.Path(f"{gr4j_calibrated}/{g_id}.csv") as f:
            f.unlink()
    except EOFError as e:
        # exit_flag.value = True
        logging.error(f"EOFError occurred while reading data for gauge {g_id}: {e}")
    except (ValueError, RuntimeWarning) as e:
        # exit_flag.value = True
        logging.error(f"Exception occurred while calibrating the {g_id}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    return None
