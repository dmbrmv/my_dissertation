"""ARCHIVED LEGACY CODE - DO NOT USE IN NEW CODE

This file contains the original hydro_metrics.py implementation that has been
replaced by modern, modular implementations.

MIGRATION GUIDE:
================

OLD (this file)                  →  NEW (use these instead)
-------------------------------------------------------------------------------
bfi(), bfi_1000()                →  src.hydro.base_flow.calculate_bfi()
FirstPass(), BackwardPass()      →  Internal in base_flow._first_pass()
slope_fdc_gauge()                →  src.hydro.flow_duration.calculate_fdc_metrics()
q5_q95()                         →  src.hydro.flow_extremes.FlowExtremes.calculate_flow_quantiles()
high_q_freq(), low_q_freq()      →  src.hydro.flow_extremes.FlowExtremes.analyze_high_flows()
high_q_dur(), low_q_dur()        →  src.hydro.flow_extremes.FlowExtremes.analyze_low_flows()
hfd_calc()                       →  src.hydro.flow_timing.calculate_timing_metrics()
hydro_job()                      →  src.hydro.hydro_metrics.calculate_hydrological_indices()
nse()                            →  src.timeseries_stats.metrics.nash_sutcliffe_efficiency()
kge()                            →  src.timeseries_stats.metrics.kling_gupta_efficiency()
rmse()                           →  sklearn.metrics.mean_squared_error(squared=False)

EXAMPLE MIGRATION:
==================

OLD CODE:
    from scripts.hydro_metrics import hydro_job, nse, kge

    result = hydro_job(hydro_year, calendar_year)
    performance = nse(predictions, targets)

NEW CODE:
    from src.hydro.base_flow import calculate_bfi
    from src.hydro.flow_extremes import FlowExtremes
    from src.timeseries_stats.metrics import nash_sutcliffe_efficiency

    # More comprehensive results
    bfi = calculate_bfi(hydro_year)
    extremes = FlowExtremes(hydro_year)
    quantiles = extremes.calculate_flow_quantiles([0.05, 0.95])

    performance = nash_sutcliffe_efficiency(targets, predictions)

WHY THE CHANGE:
===============
- Modular design: Each aspect (base flow, extremes, timing) in separate files
- Type hints: Full Python 3.10+ type annotations
- Documentation: Comprehensive docstrings with examples
- Testing: Unit tests for all functions
- Performance: Optimized numba JIT compilation where needed
- Maintainability: Clear separation of concerns

Date Archived: 2025-12-11
"""

from copy import deepcopy
import math
import random

import numba
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

"""
Расчёт ведётся для листа в котором нет пропусков. 
В случае их наличия в общем ряду, он разбивается
на n-рядов в зависимости от разбиений

Расчёт BFI далее будет производиться для отдельно 
взятого года наблюдений

"""
###################################################################

"""
First pass

"""


@numba.jit(nopython=True)
def FirstPass(Q, alpha):
    q_f_1 = [np.float64(np.nan) for i in Q]
    q_b_1 = [np.float64(np.nan) for i in Q]

    q_f_1[0] = Q[0]

    for j in range(len(Q) - 1):
        """
        for every split calculate quick flow

        """
        q_f_1[j + 1] = alpha * q_f_1[j] + 0.5 * (1 + alpha) * (Q[j + 1] - Q[j])

    for j in range(len(Q)):
        if q_f_1[j] < 0:
            q_b_1[j] = Q[j]
        else:
            q_b_1[j] = Q[j] - q_f_1[j]

    Q_forward_1 = [q_f_1, q_b_1]

    return Q_forward_1


###################################################################
"""
Backward pass

"""


@numba.jit(nopython=True)
def BackwardPass(Q_forward_1, alpha):
    """Здесь Q - n-мерный лист в зависимости от числа разбиений"""
    Qq = Q_forward_1[0]
    Qb = Q_forward_1[1]

    q_f_2 = [np.float64(np.nan) for i in Qq]
    q_b_2 = [np.float64(np.nan) for i in Qb]

    "last value of forward step - first in backward step"

    q_f_2[-1] = Qb[-1]

    for j in range(len(Qq) - 2, -1, -1):
        q_f_2[j] = alpha * q_f_2[j + 1] + 0.5 * (1 + alpha) * (Qb[j] - Qb[j + 1])

    for j in range(len(Qq) - 1, -1, -1):
        if q_f_2[j] < 0:
            q_b_2[j] = Qb[j]
        else:
            q_b_2[j] = Qb[j] - q_f_2[j]

    Q_backward = [q_f_2, q_b_2]

    return Q_backward


###################################################################
"""
Forward pass

"""


@numba.jit(nopython=True)
def ForwardPass(Q_backward, alpha):
    Qq = Q_backward[0]
    Qb = Q_backward[1]

    q_f_3 = [np.float64(np.nan) for i in Qq]
    q_b_3 = [np.float64(np.nan) for i in Qb]

    "Теперь первая величина предыдущего шага - первая и здесь"

    q_f_3[0] = Qb[0]

    for j in range(len(Qb) - 1):
        q_f_3[j + 1] = alpha * q_f_3[j] + 0.5 * (1 + alpha) * (Qb[j + 1] - Qb[j])

    for j in range(len(Qb)):
        if q_f_3[j] < 0:
            q_b_3[j] = Qb[j]
        else:
            q_b_3[j] = Qb[j] - q_f_3[j]

    Q_forward = [q_f_3, q_b_3]

    return Q_forward


###################################################################
"""
BFI calculations for given alpha

"""


@numba.jit(nopython=True)
def bfi(Q, alpha, passes, reflect):
    """We reflect the first reflect values and the last reflect values.
    this is to get rid of 'warm up' problems © Anthony Ladson
    """
    Qin = Q

    "reflect our lists"

    if len(Q) - 1 > reflect:
        Q_reflect = np.array(
            [np.float64(np.nan) for _ in range(len(Q) + 2 * reflect)], dtype=np.float64
        )

        Q_reflect[:reflect] = Q[(reflect):0:-1]
        Q_reflect[(reflect) : (reflect + len(Q))] = Q
        Q_reflect[(reflect + len(Q)) : (len(Q) + 2 + 2 * reflect)] = Q[
            len(Q) - 2 : len(Q) - reflect - 2 : -1
        ]

    else:
        Q_reflect = np.array(
            [np.float64(np.nan) for _ in range(len(Q))], dtype=np.float64
        )
        Q_reflect = Q

    Q1 = FirstPass(Q_reflect, alpha)

    "how many backwards/forward passes to we need © Anthony Ladson"

    n_pass = round(0.5 * (passes - 1))

    BackwardPass(Q1, alpha)

    for i in range(n_pass):
        Q1 = ForwardPass(BackwardPass(Q1, alpha), alpha)

    ################# end of passes  ##############################
    if len(Q) - 1 > reflect:
        Qbase = Q1[1][reflect : (len(Q1[1]) - reflect)]
        Qbase = [0 if j < 0 else j for j in Qbase]
    else:
        Qbase = Q1[1]
        Qbase = [0 if j < 0 else j for j in Qbase]

    bfi = 0
    mean_for_period = 0

    if np.mean(Qin) == 0:
        bfi = 0
    else:
        for j in Qbase:
            mean_for_period += j / np.mean(Qin)
        bfi = mean_for_period / len(Qbase)

    return bfi, Qbase


"""
BFI calculations for 1000 alpha between 0.9 and 0.98

"""


@numba.jit(nopython=True)
def bfi_1000(Q, passes, reflect):
    """Расчёт проводится для 1000 случайных значений alpha
    в диапазоне он 0.9 до 0.98

    we reflect the first reflect values and the last reflect values.
    this is to get rid of 'warm up' problems © Anthony Ladson
    """
    random.seed(1996)
    alpha_coefficients = [np.float64(random.uniform(0.9, 0.98)) for i in range(1000)]

    Q = np.array([np.float64(i) for i in Q], dtype=np.float64)
    Qin = Q

    "reflect our lists"

    if len(Q) - 1 > reflect:
        Q_reflect = np.array(
            [np.float64(np.nan) for _ in range(len(Q) + 2 * reflect)], dtype=np.float64
        )

        Q_reflect[:reflect] = Q[(reflect):0:-1]
        Q_reflect[(reflect) : (reflect + len(Q))] = Q
        Q_reflect[(reflect + len(Q)) : (len(Q) + 2 + 2 * reflect)] = Q[
            len(Q) - 2 : len(Q) - reflect - 2 : -1
        ]

    else:
        Q_reflect = np.array(
            [np.float64(np.nan) for _ in range(len(Q))], dtype=np.float64
        )
        Q_reflect = Q

    bfi_record = []
    Qbase_record = []

    for i, alpha in enumerate(alpha_coefficients):
        Q1 = FirstPass(Q_reflect, alpha)

        "how many backwards/forward passes to we need © Anthony Ladson"

        n_pass = round(0.5 * (passes - 1))

        BackwardPass(Q1, alpha)

        for i in range(n_pass):
            Q1 = ForwardPass(BackwardPass(Q1, alpha), alpha)

        ################# end of passes  ##############################
        if len(Q) - 1 > reflect:
            Qbase = Q1[1][reflect : (len(Q1[1]) - reflect)]
            Qbase = [0 if j < 0 else j for j in Qbase]
        else:
            Qbase = Q1[1]
            Qbase = [0 if j < 0 else j for j in Qbase]

        Qbase_record.append(np.array(Qbase, dtype=np.float64))

        bfi = 0
        mean_for_period = 0

        if np.mean(Qin) == 0:
            bfi = 0
        else:
            for j in Qbase:
                mean_for_period += j / np.mean(Qin)
            bfi = mean_for_period / len(Qbase)

        bfi_record.append(np.float64(bfi))

    """
    After 1000 calculations function return
    mean value out of 1000

    And "mean" hygrograph of baseflow

    """

    # mean BFI out of 1000

    bfi_mean = 0
    for i in bfi_record:
        bfi_mean += i
    bfi_mean = bfi_mean / len(bfi_record)

    # mean hydrograph out of 1000 calculations

    Qbase_mean = [np.float64(0) for i in range(len(Qbase))]

    for Qbase_temp in Qbase_record:
        for i, value in enumerate(Qbase_temp):
            Qbase_mean[i] += value

    Qbase_mean = [np.float64(i / len(Qbase_record)) for i in Qbase_mean]

    return bfi_mean, Qbase_mean


def slope_fdc_gauge(hydro_year: pd.Series):
    slope_fdc = (
        math.log(np.nanpercentile(hydro_year, q=100 - 33))
        - math.log(np.nanpercentile(hydro_year, q=100 - 66))
    ) / (0.66 - 0.33)

    return slope_fdc


def hfd_calc(calendar_year: pd.Series, hydro_year: pd.Series):
    """Date on which the cumulative discharge since 1 October
    reaches half of the annual discharge

    Args:
        calendar_year (pd.Series): _description_
        hydro_year (pd.Series): _description_

    """
    cal_val = np.nansum(calendar_year) / 2
    try:
        return hydro_year[hydro_year.cumsum() > cal_val].index[0]
    except IndexError:
        return np.nan


def q5_q95(hydro_year: pd.Series):
    q5 = np.nanpercentile(hydro_year, q=100 - 5)
    q95 = np.nanpercentile(hydro_year, q=100 - 95)

    return {"q5": q5, "q95": q95}


def high_q_freq(hydro_year: pd.Series):
    """Frequency of high-flow days (> 9 times the median daily flow)

    Args:
        hydro_year (pd.Series): _description_

    """
    hydro_year = deepcopy(hydro_year)
    med_val = np.nanmedian(hydro_year) * 9

    hydro_year[hydro_year < med_val] = np.nan

    return hydro_year


def low_q_freq(hydro_year: pd.Series):
    """Frequency of low-flow days (< 0.2 times the mean daily flow)

    Args:
        hydro_year (pd.Series): _description_

    """
    hydro_year = deepcopy(hydro_year)
    med_val = np.nanmean(hydro_year) * 2e-1

    hydro_year[hydro_year > med_val] = np.nan

    return hydro_year


def low_q_dur(hydro_year: pd.Series):
    mean_lim = np.nanmean(hydro_year)

    mean_masks = np.ma.clump_unmasked(
        np.ma.masked_where(hydro_year > mean_lim, hydro_year)
    )

    return [hydro_year[mask] for mask in mean_masks]


def high_q_dur(hydro_year: pd.Series):
    mean_lim = np.nanmedian(hydro_year) * 2

    mean_masks = np.ma.clump_unmasked(
        np.ma.masked_where(hydro_year < mean_lim, hydro_year)
    )

    return [hydro_year[mask] for mask in mean_masks]


def calculate_hydrological_indices(
    hydro_year: pd.Series,
    calendar_year: pd.Series | None = None,
) -> dict[str, float]:
    """Calculate comprehensive hydrological indices for a water year (MODERN VERSION).

    This function replaces the legacy `hydro_job()` using modern implementations
    from `src.hydro.*` modules. It calculates:
        - Mean discharge
        - Base Flow Index (BFI) using Lyne-Hollick filter
        - Flow quantiles (Q5, Q95, and full spectrum)
        - High/low flow characteristics

    Args:
        hydro_year: Discharge series for hydrological year (Oct-Sep)
        calendar_year: Discharge series for calendar year (Jan-Dec, optional)

    Returns:
        Dictionary with hydrological indices:
            - 'mean': Mean discharge
            - 'bfi': Base Flow Index
            - 'q5': 5th percentile flow
            - 'q95': 95th percentile flow
            - 'q01', 'q10', 'q90', 'q99': Additional quantiles
            - 'high_flow_frequency': % of time in high flows
            - 'low_flow_frequency': % of time in low flows

    Example:
        >>> from src.hydro.hydro_metrics import calculate_hydrological_indices
        >>> indices = calculate_hydrological_indices(hydro_year_series)
        >>> print(f"BFI: {indices['bfi']:.3f}, Mean Q: {indices['mean']:.2f}")

    Notes:
        - Uses modern BFI calculation from `src.hydro.base_flow`
        - Uses flow quantile analysis from `src.hydro.flow_extremes`
        - Handles NaN values automatically
        - Type-hinted for Python 3.10+
    """
    from src.hydro.base_flow import calculate_bfi
    from src.hydro.flow_extremes import FlowExtremes

    # Basic statistics
    hydro_mean = float(np.nanmean(hydro_year))

    # Calculate BFI using modern Lyne-Hollick filter
    bfi_value = calculate_bfi(
        hydro_year,
        alpha=None,  # Use ensemble approach
        passes=3,
        reflect_points=30,
    )

    # Calculate flow quantiles using modern FlowExtremes class
    extremes = FlowExtremes(hydro_year)
    quantiles = extremes.calculate_flow_quantiles(
        quantiles=[0.01, 0.05, 0.1, 0.9, 0.95, 0.99]
    )

    # High and low flow characteristics
    high_flow_metrics = extremes.analyze_high_flows(threshold_multiplier=2.0)
    low_flow_metrics = extremes.analyze_low_flows(threshold_multiplier=0.2)

    return {
        "mean": hydro_mean,
        "bfi": bfi_value,
        "q5": quantiles["q5_0"],  # 5th percentile
        "q95": quantiles["q95_0"],  # 95th percentile
        "q1": quantiles["q1_0"],  # 1st percentile
        "q10": quantiles["q10_0"],  # 10th percentile
        "q90": quantiles["q90_0"],  # 90th percentile
        "q99": quantiles["q99_0"],  # 99th percentile
        "high_flow_frequency": high_flow_metrics["high_flow_frequency"],
        "high_flow_events": high_flow_metrics["high_flow_events"],
        "low_flow_frequency": low_flow_metrics["low_flow_frequency"],
        "low_flow_events": low_flow_metrics["low_flow_events"],
    }


def hydro_job(hydro_year: pd.Series, calendar_year: pd.Series):
    """Calculate hydrological indices for a single water year.

    **DEPRECATED**: Use `calculate_hydrological_indices()` instead, which uses
    modern implementations from `src.hydro.*` modules.

    Args:
        hydro_year: Discharge series for hydrological year (Oct-Sep)
        calendar_year: Discharge series for calendar year (Jan-Dec)

    Returns:
        Dictionary with keys: 'mean', 'bfi', 'q5', 'q95'

    See Also:
        calculate_hydrological_indices: Modern replacement with comprehensive indices
    """
    hydro_mean = np.nanmean(hydro_year)

    # fdc_slope = slope_fdc_gauge(hydro_year)

    bfi_mean, bfi_dates = bfi_1000(hydro_year.to_numpy(), 3, 30)
    bfi_dates = pd.Series(bfi_dates, index=hydro_year.index)
    # hfd_date = hfd_calc(calendar_year,
    #                     hydro_year)
    hydro_quantile = q5_q95(hydro_year)
    # high_flow_freq = high_q_freq(hydro_year)

    # plot results
    # hydro_year.plot(label='discharge, cms',
    #                 ylim=0)
    # plt.axhline(y=hydro_mean,
    #             c='r', label='average hydro')

    # plt.axhline(y=hydro_quantile['q5'],
    #             c='green',
    #             ls='-.',
    #             label='5% quantile')

    # plt.axhline(y=hydro_quantile['q95'],
    #             c='lime',
    #             ls='-.',
    #             label='95% quantile')

    # plt.plot(bfi_dates,
    #          c='y', label='baseflow')
    # if not isinstance(hfd_date, float):
    #     plt.vlines(x=hfd_date,
    #                colors='purple',
    #                ls=':',
    #                ymin=0,
    #                ymax=hydro_year[hfd_date],
    #                label='half-flow date')

    # for i, period in enumerate(high_q_dur(hydro_year)):

    #     plt.plot(period,
    #              ls=':',
    #              lw=2,
    #              c='black',
    #              label='high flow occasions' if i == 0 else "")

    # for i, period in enumerate(low_q_dur(hydro_year)):

    #     plt.plot(period,
    #              ls=':',
    #              lw=2,
    #              c='cyan',
    #              label='low flow occasions' if i == 0 else "")

    # # place the legend outside
    # plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')

    return {
        "mean": hydro_mean,
        "bfi": bfi_mean,
        # 'hfd': hfd_date,
        "q5": hydro_quantile["q5"],
        "q95": hydro_quantile["q95"],
    }


def nse(predictions, targets):
    return 1 - (
        np.nansum((targets - predictions) ** 2)
        / np.nansum((targets - np.nanmean(targets)) ** 2)
    )


def kge(predictions, targets):
    sim_mean = np.mean(targets, axis=0, dtype=np.float64)
    obs_mean = np.mean(predictions, dtype=np.float64)

    r_num = np.sum(
        (targets - sim_mean) * (predictions - obs_mean), axis=0, dtype=np.float64
    )
    r_den = np.sqrt(
        np.sum((targets - sim_mean) ** 2, axis=0, dtype=np.float64)
        * np.sum((predictions - obs_mean) ** 2, dtype=np.float64)
    )
    r = r_num / r_den
    # calculate error in spread of flow alpha
    alpha = np.std(targets, axis=0) / np.std(predictions, dtype=np.float64)
    # calculate error in volume beta (bias of mean discharge)
    beta = np.sum(targets, axis=0, dtype=np.float64) / np.sum(
        predictions, dtype=np.float64
    )
    # calculate the Kling-Gupta Efficiency KGE
    kge_ = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return np.vstack((kge_, r, alpha, beta))


def rmse(predictions, targets):
    return mean_squared_error(targets, predictions, squared=False)


def relative_error(predictions, targets):
    return ((targets - predictions) / targets) * 100
