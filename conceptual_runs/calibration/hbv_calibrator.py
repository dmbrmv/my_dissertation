"""
Run HBV calibration for single gauge
"""

import pathlib

import spotpy
import xarray as xr
import numpy as np
from calibration.model_setups import hbv_setup
from spotpy.objectivefunctions import kge
from scripts.logger import logging

gauges = [
    i.stem
    for i in pathlib.Path(
        "/workspaces/my_dissertation/geo_data/ws_related_meteo/nc_all_q"
    ).glob("*.nc")
]
gauges = [
    i
    for i in gauges
    if i not in [i.stem for i in pathlib.Path("../hbv_calibrated/").glob("*.npy")]
]


def hbv_single_core(g_id: str) -> None:
    """

    Parameters
    ----------
    gauge_id (str) : name of the gauge to use

    Returns
    -------
    None
    """
    try:
        logging.info(f"Processing guage ID: {g_id}")
        with xr.open_dataset(
            f"/workspaces/my_dissertation/geo_data/ws_related_meteo/nc_all_q/{g_id}.nc"
        ) as f:
            example_df = f.to_pandas()
            example_df = example_df.drop("gauge_id", axis=1)
            train_df = example_df[:"2018-12-31"]

        hbv_calibrated = pathlib.Path("./hbv_calibrated")
        hbv_calibrated.mkdir(exist_ok=True, parents=True)

        sampler = spotpy.algorithms.mle(
            hbv_setup(data_file=train_df, obj_func=kge),
            dbname=f"{hbv_calibrated}/{g_id}",
            dbformat="csv",
            random_state=42,
        )

        sampler.sample(repetitions=6000)

        gauge_results = spotpy.analyser.load_csv_results(f"{hbv_calibrated}/{g_id}")
        best_hbv_params = np.array(spotpy.analyser.get_best_parameterset(gauge_results))

        with open(f"{hbv_calibrated}/{g_id}.npy", "wb") as f:
            np.save(file=f, arr=best_hbv_params)
        with pathlib.Path(f"{hbv_calibrated}/{g_id}.csv") as f:
            f.unlink()

    except EOFError as e:
        logging.error(f"EOFError occurred while reading data for gauge {g_id}: {e}")
    except (ValueError, RuntimeWarning) as e:
        logging.error(f"Exception occured while calibrating for {g_id}: {e}")
    except Exception as e:
        logging.error(f"Some bullshit for {g_id}: {e}")
    return None
