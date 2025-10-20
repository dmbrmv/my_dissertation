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
        params: List of 16 HBV parameters (aligned with bounds()):
            [0] beta: runoff contribution parameter [1, 6]
            [1] cet: evaporation correction factor [0, 0.3]
            [2] fc: maximum soil moisture storage (mm) [50, 500]
            [3] k0: recession coefficient (surface) [0.01, 0.4]
            [4] k1: recession coefficient (upper GW) [0.01, 0.4]
            [5] k2: recession coefficient (lower GW) [0.001, 0.15]
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
    day_of_year = data.index.dayofyear  # type: ignore[attr-defined]

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

    # Evaporation correction based on temperature deviation from long-term mean
    if cet > 0:
        # Calculate long-term daily temperature averages
        temp_df = pd.DataFrame({"temp": temp, "doy": day_of_year})
        temp_mean_by_doy = temp_df.groupby("doy")["temp"].transform("mean").values
        # Apply correction: Evap_corrected = (1 + CET * (T - T_mean)) * Evap
        evap = (1 + cet * (temp - temp_mean_by_doy)) * evap

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
        soil_moisture[t], recharge, excess, et_actual[t] = _soil_routine(
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
            excess,
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
) -> tuple[float, float, float, float]:
    """Execute soil moisture accounting and evapotranspiration.

    Follows Bergström (1986) HBV structure:
    1. Calculate soil wetness from previous SM: soil_wetness = (SM/FC)^BETA
    2. Calculate recharge: recharge = (rain + tosoil) * soil_wetness
    3. Update SM: SM = SM_prev + rain + tosoil - recharge
    4. Handle excess (overflow, kept SEPARATE from recharge)
    5. Calculate and apply actual ET
    6. Ensure non-negative storage

    Returns:
        Tuple of (soil_moisture, recharge, excess, actual_evap).
    """
    # Step 1: Calculate soil wetness from PREVIOUS soil moisture
    if fc > 0:
        soil_wetness = (sm_prev / fc) ** beta
        # Constrain to [0, 1] range
        soil_wetness = max(0.0, min(1.0, soil_wetness))
    else:
        soil_wetness = 0.0

    # Step 2: Calculate groundwater recharge from both rain and snowmelt
    recharge = (rain + tosoil) * soil_wetness

    # Step 3: Update soil moisture (before evaporation)
    sm = sm_prev + rain + tosoil - recharge

    # Step 4: Handle excess water (overflow, kept SEPARATE from recharge!)
    excess = sm - fc
    if excess > 0:
        sm = fc
    else:
        excess = 0.0

    # Step 5: Calculate evaporation factor based on current soil moisture
    if lp * fc > 0:
        evap_factor = sm / (lp * fc)
        evap_factor = max(0.0, min(1.0, evap_factor))
    else:
        evap_factor = 0.0

    # Step 6: Calculate actual evaporation
    et_actual = evap * evap_factor
    # Cannot evaporate more than available
    et_actual = min(et_actual, sm)

    # Step 7: Apply evaporation to soil moisture
    sm = sm - et_actual

    # Step 8: Ensure non-negative soil moisture
    sm = max(0.0, sm)

    return sm, recharge, excess, et_actual


def _groundwater_routine(
    suz_prev: float,
    slz_prev: float,
    recharge: float,
    excess: float,
    perc_rate: float,
    k0: float,
    k1: float,
    k2: float,
    uzl: float,
) -> tuple[float, float, float]:
    """Execute upper and lower groundwater box dynamics.

    Follows Bergström (1986) HBV structure:
    1. Add recharge AND excess to upper zone: SUZ = SUZ + recharge + excess
    2. Percolation to lower zone: perc = min(SUZ, PERC)
    3. Update SUZ: SUZ = SUZ - perc
    4. Calculate Q0 from upper zone: Q0 = K0 * max(SUZ - UZL, 0)
    5. Update SUZ: SUZ = SUZ - Q0
    6. Calculate Q1 from remaining upper zone: Q1 = K1 * SUZ
    7. Update SUZ: SUZ = SUZ - Q1
    8. Calculate Q2 from lower zone: Q2 = K2 * SLZ

    Returns:
        Tuple of (upper_zone, lower_zone, total_runoff).
    """
    # Step 1: Add both recharge AND excess to upper zone
    suz = suz_prev + recharge + excess

    # Step 2: Percolation from upper to lower zone (before runoff)
    perc = min(perc_rate, suz)
    suz = suz - perc

    # Step 3: Calculate Q0 (surface/quick runoff from upper part of SUZ)
    q0 = k0 * max(suz - uzl, 0.0)
    suz = suz - q0

    # Step 4: Calculate Q1 (intermediate runoff from remaining SUZ)
    q1 = k1 * suz
    suz = suz - q1

    # Step 5: Update lower zone and calculate Q2 (baseflow)
    slz = slz_prev + perc

    q2 = k2 * slz
    slz = slz - q2

    # Step 6: Ensure non-negative storages
    suz = max(0.0, suz)
    slz = max(0.0, slz)

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
    if maxbas_int <= 1:
        return q_sim

    # Safe division with explicit check
    cutoff_freq = 1.0 / max(maxbas_int, 1)
    b, a = ss.butter(maxbas_int, cutoff_freq)
    q_smoothed = ss.lfilter(b, a, q_sim)
    q_smoothed = np.where(q_smoothed > 0, q_smoothed, 0.0)

    return q_smoothed


def bounds() -> tuple[tuple[float, float], ...]:
    """Return parameter bounds for HBV model (Bergström, 1986).

    Returns:
        Tuple of (min, max) bounds for 16 parameters:
        beta, cet, fc, k0, k1, k2, lp, maxbas, perc, uzl,
        pcorr, tt, cfmax, sfcf, cfr, cwh.
    """
    return (
        (1.0, 6.0),  # beta - runoff contribution parameter
        (0.0, 0.3),  # cet - evaporation correction factor
        (50.0, 500.0),  # fc - maximum soil moisture storage (mm)
        (0.01, 0.4),  # k0 - recession coefficient (surface)
        (0.01, 0.4),  # k1 - recession coefficient (upper GW)
        (0.001, 0.15),  # k2 - recession coefficient (lower GW)
        (0.3, 1.0),  # lp - evaporation threshold (SM/FC)
        (1.0, 7.0),  # maxbas - Butterworth filter order
        (0.0, 3.0),  # perc - percolation rate (mm/day)
        (0.0, 500.0),  # uzl - threshold for surface runoff (mm)
        (0.5, 2.0),  # pcorr - precipitation correction factor
        (-1.5, 2.5),  # tt - temperature threshold for snow/rain (°C)
        (1.0, 10.0),  # cfmax - snow melt rate (mm/day/°C)
        (0.4, 1.0),  # sfcf - snowfall correction factor
        (0.0, 0.1),  # cfr - refreezing coefficient
        (0.0, 0.2),  # cwh - water holding capacity in snow
    )


__all__ = ["simulation", "bounds"]
