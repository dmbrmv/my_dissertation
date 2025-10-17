"""HBV rainfall-runoff model implementation (Bergström, 1986).

This module provides the HBV (Hydrologiska Byråns Vattenbalansavdelning)
conceptual hydrological model with snow routine and soil moisture accounting.
"""

import numpy as np
import pandas as pd
import scipy.signal as ss


def simulation(
    data: pd.DataFrame,
    params: tuple[float, ...] | list[float] = (
        1.0,
        0.1,
        150.0,
        0.2,
        0.2,
        0.05,
        0.5,
        2.0,
        2.0,
        100.0,
        1.0,
        0.0,
        2.0,
        0.6,
        0.05,
        0.1,
    ),
) -> np.ndarray:
    """Run HBV rainfall-runoff simulation with snow module.

    Args:
        data: DataFrame with columns 'Temp' (°C), 'Prec' (mm/day), 'Evap' (mm/day).
        params: List of 16 HBV parameters:
            [0] beta: shape parameter for runoff contribution [1, 6]
            [1] cet: evaporation correction factor [0, 0.3]
            [2] fc: maximum soil moisture storage (mm) [50, 500]
            [3] k0: recession coefficient for surface runoff [0.01, 0.4]
            [4] k1: recession coefficient for upper groundwater [0.01, 0.4]
            [5] k2: recession coefficient for lower groundwater [0.001, 0.15]
            [6] lp: evaporation threshold (SM/FC) [0.3, 1]
            [7] maxbas: Butterworth filter order [1, 7]
            [8] perc: percolation rate (mm/day) [0, 3]
            [9] uzl: threshold for surface runoff (mm) [0, 500]
            [10] pcorr: precipitation correction factor [0.5, 2]
            [11] tt: temperature threshold for snow/rain (°C) [-1.5, 2.5]
            [12] cfmax: snow melt rate (mm/day/°C) [1, 10]
            [13] sfcf: snowfall correction factor [0.4, 1]
            [14] cfr: refreezing coefficient [0, 0.1]
            [15] cwh: water holding capacity in snow [0, 0.2]

    Returns:
        Array of simulated runoff (mm/day).
    """
    temp = data["Temp"].values
    prec = data["Prec"].values
    evap = data["Evap"].values

    # Unpack parameters with descriptive names
    (
        beta,
        cet,
        fc,
        k0,
        k1,
        k2,
        lp,
        maxbas,
        perc_rate,
        uzl,
        pcorr,
        tt,
        cfmax,
        sfcf,
        cfr,
        cwh,
    ) = params

    # Initialize state arrays
    n_steps = len(prec)
    snowpack = np.zeros(n_steps)
    snowpack[0] = 0.0001
    meltwater = np.zeros(n_steps)
    meltwater[0] = 0.0001
    soil_moisture = np.zeros(n_steps)
    soil_moisture[0] = 0.0001
    upper_zone = np.zeros(n_steps)
    upper_zone[0] = 0.0001
    lower_zone = np.zeros(n_steps)
    lower_zone[0] = 0.0001
    et_actual = np.zeros(n_steps)
    et_actual[0] = 0.0001
    q_sim = np.zeros(n_steps)
    q_sim[0] = 0.0001

    # Apply precipitation correction
    prec = pcorr * prec

    # Separate precipitation into rain and snow
    rain = np.where(temp > tt, prec, 0.0)
    snow = np.where(temp <= tt, prec, 0.0)
    snow = sfcf * snow

    # Control evaporation (non-negative)
    evap = np.where(evap > 0, evap, 0.0)

    # Main simulation loop
    for t in range(1, n_steps):
        # Snow routine
        snowpack[t], meltwater[t], tosoil = _snow_routine(
            snowpack[t - 1],
            meltwater[t - 1],
            snow[t],
            temp[t],
            tt,
            cfmax,
            cfr,
            cwh,
        )

        # Soil and evaporation routine
        soil_moisture[t], recharge, et_actual[t] = _soil_routine(
            soil_moisture[t - 1],
            rain[t],
            tosoil,
            evap[t],
            fc,
            beta,
            lp,
        )

        # Groundwater routine
        upper_zone[t], lower_zone[t], q_sim[t] = _groundwater_routine(
            upper_zone[t - 1],
            lower_zone[t - 1],
            recharge,
            perc_rate,
            k0,
            k1,
            k2,
            uzl,
        )

    # Apply routing (scale effect)
    q_sim = _apply_routing(q_sim, maxbas)

    return q_sim


def _snow_routine(
    snowpack_prev: float,
    meltwater_prev: float,
    snow: float,
    temp: float,
    tt: float,
    cfmax: float,
    cfr: float,
    cwh: float,
) -> tuple[float, float, float]:
    """Execute snow accumulation, melting, and refreezing.

    Returns:
        Tuple of (snowpack, meltwater, recharge_to_soil).
    """
    snowpack = snowpack_prev + snow

    # Calculate melting
    melt = cfmax * (temp - tt)
    melt = max(0.0, melt)
    melt = min(melt, snowpack)

    meltwater = meltwater_prev + melt
    snowpack = snowpack - melt

    # Calculate refreezing
    refreeze = cfr * cfmax * (tt - temp)
    refreeze = max(0.0, refreeze)
    refreeze = min(refreeze, meltwater)

    snowpack = snowpack + refreeze
    meltwater = meltwater - refreeze

    # Calculate recharge to soil
    tosoil = meltwater - (cwh * snowpack)
    tosoil = max(0.0, tosoil)
    meltwater = meltwater - tosoil

    return snowpack, meltwater, tosoil


def _soil_routine(
    sm_prev: float,
    rain: float,
    tosoil: float,
    evap: float,
    fc: float,
    beta: float,
    lp: float,
) -> tuple[float, float, float]:
    """Execute soil moisture accounting and evapotranspiration.

    Returns:
        Tuple of (soil_moisture, recharge, actual_evap).
    """
    # Calculate soil wetness
    soil_wetness = (sm_prev / fc) ** beta
    soil_wetness = np.clip(soil_wetness, 0.0, 1.0)

    # Calculate recharge
    recharge = (rain + tosoil) * soil_wetness
    sm = sm_prev + rain + tosoil - recharge

    # Handle excess water
    excess = sm - fc
    excess = max(0.0, excess)
    sm = sm - excess
    recharge = recharge + excess

    # Calculate actual evapotranspiration
    evap_factor = sm / (lp * fc)
    evap_factor = np.clip(evap_factor, 0.0, 1.0)
    et_actual = evap * evap_factor
    et_actual = min(sm, et_actual)

    sm = sm - et_actual

    return sm, recharge, et_actual


def _groundwater_routine(
    suz_prev: float,
    slz_prev: float,
    recharge: float,
    perc_rate: float,
    k0: float,
    k1: float,
    k2: float,
    uzl: float,
) -> tuple[float, float, float]:
    """Execute upper and lower groundwater box dynamics.

    Returns:
        Tuple of (upper_zone, lower_zone, total_runoff).
    """
    # Update upper zone
    suz = suz_prev + recharge

    # Percolation
    perc = min(suz, perc_rate)
    suz = suz - perc

    # Surface runoff (Q0)
    q0 = k0 * max(suz - uzl, 0.0)
    suz = suz - q0

    # Upper zone runoff (Q1)
    q1 = k1 * suz
    suz = suz - q1

    # Update lower zone
    slz = slz_prev + perc

    # Lower zone runoff (Q2)
    q2 = k2 * slz
    slz = slz - q2

    # Total runoff
    q_total = q0 + q1 + q2

    return suz, slz, q_total


def _apply_routing(q_sim: np.ndarray, maxbas: float) -> np.ndarray:
    """Apply Butterworth filter for routing (scale effect).

    Args:
        q_sim: Simulated runoff array.
        maxbas: Butterworth filter order (1-7).

    Returns:
        Smoothed runoff array.
    """
    maxbas_int = int(maxbas)
    if maxbas_int == 1:
        return q_sim

    b, a = ss.butter(maxbas_int, 1.0 / maxbas_int)
    q_smoothed = ss.lfilter(b, a, q_sim)
    q_smoothed = np.where(q_smoothed > 0, q_smoothed, 0.0)

    return q_smoothed


def bounds() -> tuple[tuple[float, float], ...]:
    """Return parameter bounds for HBV model.

    Returns:
        Tuple of (min, max) bounds for 16 parameters:
        beta, cet, fc, k0, k1, k2, lp, maxbas, perc, uzl,
        pcorr, tt, cfmax, sfcf, cfr, cwh.
    """
    return (
        (1.0, 6.0),  # beta
        (0.0, 0.3),  # cet
        (50.0, 500.0),  # fc
        (0.01, 0.4),  # k0
        (0.01, 0.4),  # k1
        (0.001, 0.15),  # k2
        (0.3, 1.0),  # lp
        (1.0, 7.0),  # maxbas
        (0.0, 3.0),  # perc
        (0.0, 500.0),  # uzl
        (0.5, 2.0),  # pcorr
        (-1.5, 2.5),  # tt
        (1.0, 10.0),  # cfmax
        (0.4, 1.0),  # sfcf
        (0.0, 1.0),  # cfr
        (0.0, 0.2),  # cwh
    )


__all__ = ["simulation", "bounds"]
