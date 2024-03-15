"""
Run gr4j calibration for single gauge
"""

import pathlib
from pathlib import Path

import spotpy
import xarray as xr
from calibration.model_setups import gr4j_setup
from spotpy.objectivefunctions import kge

gauges = [
    i.stem
    for i in pathlib.Path("/Users/dmbrmv/Development/geo_data/great_db/nc_all_q").glob(
        "*.nc"
    )
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
    with xr.open_dataset(
        f"/Users/dmbrmv/Development/geo_data/great_db/nc_all_q/{g_id}.nc"
    ) as f:
        example_df = f.to_pandas()
        example_df = example_df.drop("gauge_id", axis=1)
        train_df = example_df[:"2018-12-31"]

    gr4j_calibrated: Path = pathlib.Path("../gr4j_calibrated")
    gr4j_calibrated.mkdir(exist_ok=True, parents=True)

    sampler = spotpy.algorithms.mle(
        gr4j_setup(data_file=train_df, obj_func=kge),
        dbname=f"{gr4j_calibrated}/{g_id}",
        dbformat="csv",
        random_state=42,
    )

    sampler.sample(repetitions=6000)

    return None
