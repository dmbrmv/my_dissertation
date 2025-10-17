"""Potential Evapotranspiration (PET) calculation utilities."""

from collections.abc import Sequence

import numpy as np


def pet_oudin(mean_temp: Sequence[float], day_of_year: Sequence[int], latitude_deg: float) -> np.ndarray:
    """Compute daily PET using the Oudin (2005) formula.

    Args:
        mean_temp: Mean air temperature [°C] for each day.
        day_of_year: Julian day (1-365/366) for each day.
        latitude_deg: Latitude in decimal degrees (positive north).

    Returns:
        Daily PET in mm/day (same length as mean_temp).
    """
    g_sc = 0.0820  # MJ m-2 min-1
    lambda_ = 2.45  # MJ kg-1
    phi = np.deg2rad(latitude_deg)
    doy = np.asarray(day_of_year, dtype=float)
    tmean = np.asarray(mean_temp, dtype=float)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    delta = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    omega_s = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0))
    ra = (
        (24.0 * 60.0 / np.pi)
        * g_sc
        * dr
        * (omega_s * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(omega_s))
    )
    pet = (ra / lambda_) * (tmean + 5.0) / 100.0
    pet = np.maximum(pet, 0.0)
    return pet
