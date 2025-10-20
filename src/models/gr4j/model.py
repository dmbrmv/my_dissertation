# filepath: /home/dmbrmv/Development/MeteoSources/src/models/gr4j_optimized.py
"""GR4J rainfall-runoff model implementation.

This module implements the GR4J (modèle du Génie Rural à 4 paramètres Journalier)
hydrological model with optional CemaNeige snow module.
"""

import numpy as np
import pandas as pd

# Import the optimized snow model
from . import cema_neige


def simulation(data: pd.DataFrame, params: list[float]) -> np.ndarray:
    """Run the GR4J rainfall-runoff simulation, optionally with snow module.

    Args:
        data: DataFrame with meteorological data:
            - 't_mean': Temperature (°C)
            - 'prcp': Precipitation (mm/day)
            - 'evap': Potential evapotranspiration (mm/day)
        params: List of GR4J and optionally CemaNeige parameters:
            - [0] 'x1': production store capacity [mm], range [0, 1500]
            - [1] 'x2': intercatchment exchange coefficient [mm/day], range [-10, 5]
            - [2] 'x3': routing store capacity [mm], range [1, 500]
            - [3] 'x4': time constant of unit hydrograph [day], range [0.5, 4.0]
            - [4] 'ctg': snow thermal state coefficient, range [0, 1]
            - [5] 'kf': melting rate coefficient, range [1, 10]
            - [6] 'tt': temperature threshold, range [-1.5, 2.5]

    Returns:
        Array of simulated runoff values (mm/day)
    """
    # Extract parameters
    x1, x2, x3, x4 = params[:4]
    snow_params = params[4:7] if len(params) >= 7 else None

    # Get effective precipitation considering snow if snow params provided
    if snow_params is not None:
        prcp = cema_neige.simulation(data, snow_params)
    else:
        prcp = data["prcp"].values

    # Get potential evapotranspiration
    evap = data["pet_mm_day"].values

    # Parameter for unit hydrograph length
    nh = 20

    # Initialize model states
    # Store states: [production_store, routing_store]
    state = np.array([x1 / 2.0, x3 / 2.0])

    # Initialize output array
    q_out = np.zeros(len(prcp))

    # Initialize unit hydrograph states
    uh1_state = np.zeros(nh)
    uh2_state = np.zeros(2 * nh)

    # Compute unit hydrograph ordinates
    ord_uh1 = _compute_uh1(x4, nh)
    ord_uh2 = _compute_uh2(x4, nh)

    # Main simulation loop
    for t in range(len(prcp)):
        # Process rainfall vs. evaporation interactions
        if prcp[t] <= evap[t]:
            # Case: Evaporation dominates
            net_evap = evap[t] - prcp[t]
            evap_factor = min(13.0, net_evap / x1)  # Cap to avoid extreme values
            tws = np.tanh(evap_factor)
            store_ratio = state[0] / x1

            # Calculate evaporation from production store
            actual_evap = (
                state[0] * (2.0 - store_ratio) * tws / (1.0 + (1.0 - store_ratio) * tws)
            )

            # Update states
            state[0] -= actual_evap
            effective_rainfall = 0.0
        else:
            # Case: Rainfall dominates
            net_rainfall = prcp[t] - evap[t]
            rainfall_factor = min(13.0, net_rainfall / x1)  # Cap to avoid extreme values
            tws = np.tanh(rainfall_factor)
            store_ratio = state[0] / x1

            # Calculate production store addition
            store_addition = (
                x1 * (1.0 - store_ratio * store_ratio) * tws / (1.0 + store_ratio * tws)
            )
            effective_rainfall = net_rainfall - store_addition
            state[0] += store_addition

        # Apply percolation from production store
        store_ratio = (state[0] / x1) ** 4
        percolation = state[0] * (
            1.0 - 1.0 / np.sqrt(np.sqrt(1.0 + store_ratio / 25.62891))
        )
        state[0] -= percolation
        effective_rainfall += percolation

        # Split effective rainfall between two routing paths
        rain_to_uh1 = 0.9 * effective_rainfall
        rain_to_uh2 = 0.1 * effective_rainfall

        # Route through UH1
        _route_uh1(uh1_state, ord_uh1, rain_to_uh1, x4, nh)

        # Route through UH2
        _route_uh2(uh2_state, ord_uh2, rain_to_uh2, x4, nh)

        # Calculate groundwater exchange
        routing_ratio = state[1] / x3
        exchange = x2 * routing_ratio**3.5

        # Process routing store
        routing_input = uh1_state[0] + exchange
        state[1] = max(0.0, state[1] + routing_input)
        routing_ratio = (state[1] / x3) ** 4
        q_routing = state[1] * (1.0 - 1.0 / np.sqrt(np.sqrt(1.0 + routing_ratio)))
        state[1] -= q_routing

        # Process direct runoff
        q_direct = max(0.0, uh2_state[0] + exchange)

        # Calculate total runoff
        q_out[t] = q_routing + q_direct

    return q_out


def bounds() -> tuple[tuple[float, float], ...]:
    """Return parameter bounds for model calibration.

    Returns:
        Tuple of parameter bounds for all GR4J and CemaNeige parameters
    """
    # GR4J bounds
    gr4j_bounds = (
        (10.0, 3000.0),  # x1: production store capacity [mm]
        (-20.0, 10.0),  # x2: intercatchment exchange coefficient [mm/day]
        (1.0, 4000.0),  # x3: routing store capacity [mm]
        (0.05, 20.0),  # x4: time constant of unit hydrograph [day]
    )

    # CemaNeige bounds
    snow_bounds = (
        (0.0, 3.0),  # ctg: thermal state coefficient
        (1.0, 10.0),  # kf: melting rate coefficient
        (-1.5, 3.0),  # tt: temperature threshold
    )

    return gr4j_bounds + snow_bounds


def _compute_uh1(c: float, nh: int) -> np.ndarray:
    """Compute ordinates of the first unit hydrograph.

    Args:
        c: Time constant parameter
        nh: Number of ordinates

    Returns:
        Array of UH1 ordinates
    """
    ord_uh1 = np.zeros(nh)
    for i in range(nh):
        ord_uh1[i] = _ss1(i, c, 2.5) - _ss1(i - 1, c, 2.5)
    return ord_uh1


def _compute_uh2(c: float, nh: int) -> np.ndarray:
    """Compute ordinates of the second unit hydrograph.

    Args:
        c: Time constant parameter
        nh: Number of ordinates

    Returns:
        Array of UH2 ordinates
    """
    ord_uh2 = np.zeros(2 * nh)
    for i in range(2 * nh):
        ord_uh2[i] = _ss2(i, c, 2.5) - _ss2(i - 1, c, 2.5)
    return ord_uh2


def _ss1(t: int, c: float, d: float) -> float:
    """Calculate S-curve value for UH1.

    Args:
        t: Time step
        c: Time constant
        d: Exponent

    Returns:
        S-curve value
    """
    fi = t + 1
    if fi <= 0:
        return 0.0
    elif fi < c:
        return (fi / c) ** d
    else:
        return 1.0


def _ss2(t: int, c: float, d: float) -> float:
    """Calculate S-curve value for UH2.

    Args:
        t: Time step
        c: Time constant
        d: Exponent

    Returns:
        S-curve value
    """
    fi = t + 1
    if fi <= 0:
        return 0.0
    elif fi <= c:
        return 0.5 * (fi / c) ** d
    elif c < fi <= 2 * c:
        return 1.0 - 0.5 * (2.0 - fi / c) ** d
    else:
        return 1.0


def _route_uh1(
    state: np.ndarray, ordinates: np.ndarray, input_val: float, x4: float, nh: int
) -> None:
    """Route water through the first unit hydrograph.

    Args:
        state: State array to update
        ordinates: UH ordinates
        input_val: Input value to route
        x4: Time parameter
        nh: Length of hydrograph
    """
    uh_size = max(1, min(nh - 1, int(x4 + 1)))

    # Shift states and apply new input
    for k in range(uh_size):
        state[k] = state[k + 1] + ordinates[k] * input_val

    # Last ordinate
    state[nh - 1] = ordinates[nh - 1] * input_val


def _route_uh2(
    state: np.ndarray, ordinates: np.ndarray, input_val: float, x4: float, nh: int
) -> None:
    """Route water through the second unit hydrograph.

    Args:
        state: State array to update
        ordinates: UH ordinates
        input_val: Input value to route
        x4: Time parameter
        nh: Length of hydrograph
    """
    uh_size = max(1, min(2 * nh - 1, 2 * int(x4 + 1)))

    # Shift states and apply new input
    for k in range(uh_size):
        state[k] = state[k + 1] + ordinates[k] * input_val

    # Last ordinate
    state[2 * nh - 1] = ordinates[2 * nh - 1] * input_val
