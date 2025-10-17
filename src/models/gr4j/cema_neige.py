# filepath: /home/dmbrmv/Development/MeteoSources/src/models/cema_neige_optimized.py
"""CemaNeige snow model for hydrological modeling.

This module implements the CemaNeige snow accumulation and melt model.
"""

import numpy as np
import pandas as pd

# Import logger setup
from ...utils.logger import setup_logger

logger = setup_logger(function_name="gr4j_cema_neige", level="INFO", log_file="logs/gr4j_optuna.log")


def simulation(
    data: pd.DataFrame, params: list[float], verbose: bool = False
) -> np.ndarray | tuple[np.ndarray, list[float]]:
    """Run the CemaNeige snow model simulation.

    This model simulates snow accumulation and melt processes based on
    temperature and precipitation inputs.

    Args:
        data: DataFrame with temperature and precipitation timeseries
            - 't_mean': mean daily temperature (Celsius degrees)
            - 'prcp': mean daily precipitation (mm/day)
        params: List of model parameters:
            - [0] 'ctg': dimensionless weighting coefficient of snow pack thermal state [0, 1]
            - [1] 'kf': day-degree rate of melting (mm/(day*celsius degree)) [1, 10]
            - [2] 'tt': temperature threshold separating rain/snow (optional) [-1.5, 2.5]
        verbose: If True, returns both liquid water and snowpack, otherwise just liquid water

    Returns:
        If verbose=False: np.ndarray with total liquid water (rain + melt)
        If verbose=True: Tuple of (liquid water array, snowpack timeseries list)
    """
    logger.debug(f"Running CemaNeige simulation with parameters: {params}")

    # Parameter extraction with validation
    if len(params) == 3:
        ctg, kf, tt = params
    elif len(params) == 2:
        ctg, kf = params
        tt = -0.2
        logger.info(f"TT parameter not set. Setting to default value of {tt}")
    else:
        error_msg = f"Expected 2 or 3 parameters, got {len(params)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Input data extraction
    temp = data["t_mean"].values
    prec = data["prcp"].values

    # Constants
    tmelt = 0.0  # Melting temperature threshold
    min_speed = 0.1  # Minimum melt speed

    # Calculate fraction of solid precipitation
    fraq_solid_precip = np.where(temp < tt, 1, 0)

    # Calculate mean annual solid precipitation
    def mean_annual_solid_precip(data: pd.DataFrame) -> float:
        """Calculate mean annual solid precipitation.

        Args:
            data: DataFrame with temperature and precipitation data

        Returns:
            Mean annual solid precipitation value
        """
        years = np.unique(data.index.year)
        annual_vals = [data.prcp[data.prcp.index.year == year][data.t_mean < tt].sum() for year in years]
        return np.mean(annual_vals)

    # Initialize model states
    masp = mean_annual_solid_precip(data)

    g_threshold = 0.9 * masp

    # State variables
    g = 0.0  # Snow pack volume
    etg = 0.0  # Snow pack thermal state

    # Output arrays
    snowpack = []
    pliq_and_melt = np.zeros(len(temp))

    # Main simulation loop
    for t in range(len(temp)):
        # Solid and liquid precipitation accounting
        pliq = (1 - fraq_solid_precip[t]) * prec[t]  # Liquid precipitation
        psol = fraq_solid_precip[t] * prec[t]  # Solid precipitation

        # Update snow pack volume before melt
        g += psol

        # Update snow pack thermal state before melt
        # Prevent overflow by limiting extreme values
        if abs(etg) > 1e10:
            etg = -1e10 if etg < 0 else 1e10
        etg = ctg * etg + (1 - ctg) * temp[t]

        # Control thermal state (can't be positive)
        if etg > 0:
            etg = 0

        # Calculate potential melt
        if (etg == 0) and (temp[t] > tmelt):
            pot_melt = kf * (temp[t] - tmelt)
            # Can't melt more than available
            pot_melt = min(pot_melt, g)
        else:
            pot_melt = 0

        # Calculate ratio of snow pack cover
        g_ratio = g / g_threshold if g < g_threshold else 1.0

        # Calculate actual melt with coverage factor
        melt = ((1 - min_speed) * g_ratio + min_speed) * pot_melt

        # Update snow pack volume
        g -= melt

        # Store results
        pliq_and_melt[t] = pliq + melt
        snowpack.append(g)

    if verbose:
        return pliq_and_melt, snowpack
    else:
        return pliq_and_melt


def bounds() -> tuple[tuple[float, float], ...]:
    """Return parameter bounds for model calibration.

    Returns:
        Tuple of parameter bounds: ((min1, max1), (min2, max2), ...)
        - 'ctg': dimensionless weighting coefficient [0, 1]
        - 'kf': day-degree rate of melting [1, 10]
        - 'tt': temperature threshold for snow/rain [-1.5, 2.5]
    """
    bounds_val = ((0.0, 3.0), (1.0, 10.0), (-1.5, 3.0))
    return bounds_val
