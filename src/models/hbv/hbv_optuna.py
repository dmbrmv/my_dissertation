"""Improved HBV Optuna multi-objective optimization with warm-up support.

This module mirrors the enhanced structure used for the GR4J calibration,
adapting it for the HBV conceptual hydrological model. Key features:

1. Multi-year warm-up period for state stabilization.
2. Composite and detailed objective suites covering full flow regimes.
3. Parameter suggestions constrained by literature-derived bounds.
4. Early-stopping callback for efficient composite optimization runs.
"""

from __future__ import annotations

import numpy as np
import optuna  # type: ignore[import-untyped]
import pandas as pd

from src.models.hbv import hbv
from src.utils.logger import setup_logger
from src.utils.metrics import (
    evaluate_model,
    kling_gupta_efficiency,
    percent_bias,
)
from src.utils.metrics_enhanced import (
    composite_high_flow_metric,
    composite_low_flow_metric,
    sqrt_nse,
)

logger = setup_logger("main_hbv_optuna", log_file="logs/hbv_optuna.log")


# --------------------------------------------------------------------------- #
# Objective functions
# --------------------------------------------------------------------------- #


def multi_objective_composite(
    trial: optuna.Trial,
    data: pd.DataFrame,
    calibration_period: tuple[str, str],
    warmup_years: int = 2,
) -> tuple[float, float, float, float]:
    """Composite 4-objective optimization covering all flow regimes.

    The objectives mirror the GR4J implementation to ensure comparability:
        1. Kling-Gupta Efficiency (overall performance).
        2. Composite low-flow metric (logNSE & inverse NSE).
        3. Composite high-flow metric (NSE & peak flow error).
        4. Volume conservation (negative absolute PBIAS).
    """
    params = _suggest_hbv_parameters(trial)

    q_obs, q_sim = _simulate_with_warmup(
        data=data,
        params=params,
        calibration_period=calibration_period,
        warmup_years=warmup_years,
    )

    if q_obs is None or q_sim is None:
        return -999.0, -999.0, -999.0, -999.0

    kge = kling_gupta_efficiency(q_obs, q_sim)
    low_flow = composite_low_flow_metric(q_obs, q_sim, weights=(0.5, 0.5))
    high_flow = composite_high_flow_metric(q_obs, q_sim, weights=(0.7, 0.3))
    pbias_abs = abs(percent_bias(q_obs, q_sim))

    # Replace NaNs with poor scores to keep Pareto front stable.
    kge = _nan_guard(kge, -999.0)
    low_flow = _nan_guard(low_flow, -999.0)
    high_flow = _nan_guard(high_flow, -999.0)
    pbias_abs = _nan_guard(pbias_abs, 999.0)

    return kge, low_flow, high_flow, -pbias_abs


def multi_objective_detailed(
    trial: optuna.Trial,
    data: pd.DataFrame,
    calibration_period: tuple[str, str],
    warmup_years: int = 2,
) -> tuple[float, float, float, float, float, float]:
    """Detailed 6-objective optimization for diagnostics.

    Returns: (KGE, NSE, sqrtNSE, logNSE, invNSE, -abs(PBIAS))
    """
    params = _suggest_hbv_parameters(trial)

    q_obs, q_sim = _simulate_with_warmup(
        data=data,
        params=params,
        calibration_period=calibration_period,
        warmup_years=warmup_years,
    )

    if q_obs is None or q_sim is None:
        return tuple([-999.0] * 6)  # type: ignore[return-value]

    metrics = evaluate_model(q_obs, q_sim)
    kge = _nan_guard(metrics.get("KGE"), -999.0)
    nse = _nan_guard(metrics.get("NSE"), -999.0)
    log_nse = _nan_guard(metrics.get("logNSE"), -999.0)

    sqrt_metric = sqrt_nse(q_obs, q_sim)

    from src.utils.metrics_enhanced import inverse_nse  # avoid circular import

    inv_metric = inverse_nse(q_obs, q_sim)
    pbias_abs = abs(_nan_guard(metrics.get("PBIAS"), 999.0))

    sqrt_metric = _nan_guard(sqrt_metric, -999.0)
    inv_metric = _nan_guard(inv_metric, -999.0)

    return kge, nse, sqrt_metric, log_nse, inv_metric, -pbias_abs


# --------------------------------------------------------------------------- #
# Helper routines
# --------------------------------------------------------------------------- #


def _suggest_hbv_parameters(trial: optuna.Trial) -> list[float]:
    """Suggest HBV parameter set following literature bounds."""
    par_beta = trial.suggest_float("parBETA", 1.0, 6.0)
    par_cet = trial.suggest_float("parCET", 0.0, 0.3)
    par_fc = trial.suggest_float("parFC", 50.0, 500.0, log=True)
    par_k0 = trial.suggest_float("parK0", 0.01, 0.4)
    par_k1 = trial.suggest_float("parK1", 0.01, 0.4)
    par_k2 = trial.suggest_float("parK2", 0.001, 0.15, log=True)
    par_lp = trial.suggest_float("parLP", 0.3, 1.0)
    par_maxbas = trial.suggest_float("parMAXBAS", 1.0, 7.0)
    par_perc = trial.suggest_float("parPERC", 0.0, 3.0)
    par_uzl = trial.suggest_float("parUZL", 1e-6, 500.0, log=True)
    par_pcorr = trial.suggest_float("parPCORR", 0.5, 2.0)
    par_tt = trial.suggest_float("parTT", -1.5, 2.5)
    par_cfmax = trial.suggest_float("parCFMAX", 1.0, 10.0)
    par_sfcf = trial.suggest_float("parSFCF", 0.4, 1.0)
    par_cfr = trial.suggest_float("parCFR", 0.0, 0.1)
    par_cwh = trial.suggest_float("parCWH", 0.0, 0.2)

    return [
        par_beta,
        par_cet,
        par_fc,
        par_k0,
        par_k1,
        par_k2,
        par_lp,
        par_maxbas,
        par_perc,
        par_uzl,
        par_pcorr,
        par_tt,
        par_cfmax,
        par_sfcf,
        par_cfr,
        par_cwh,
    ]


def _simulate_with_warmup(
    data: pd.DataFrame,
    params: list[float],
    calibration_period: tuple[str, str],
    warmup_years: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Run HBV simulation including warm-up period."""
    calib_start = pd.to_datetime(calibration_period[0])
    calib_end = pd.to_datetime(calibration_period[1])
    warmup_start = calib_start - pd.DateOffset(years=warmup_years)

    if warmup_start < data.index[0]:
        logger.warning(
            "Insufficient data for warm-up period. "
            "Using earliest available date %s instead of %s.",
            data.index[0],
            warmup_start,
        )
        warmup_start = data.index[0]

    try:
        warmup_data = data.loc[warmup_start:calib_end]
        q_sim_full = hbv.simulation(warmup_data, params)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("HBV simulation failed: %s", exc)
        return None, None

    warmup_count = len(data.loc[warmup_start:calib_start]) - 1

    q_sim = np.asarray(q_sim_full[warmup_count:], dtype=float)
    q_obs_series = data.loc[calibration_period[0] : calibration_period[1], "q_mm_day"]
    q_obs = np.asarray(q_obs_series.values, dtype=float)

    min_len = min(len(q_obs), len(q_sim))
    if min_len <= 0:
        logger.error("Warm-up produced empty simulation/observation arrays.")
        return None, None

    return q_obs[:min_len], q_sim[:min_len]


def _nan_guard(value: float | None, fallback: float) -> float:
    """Return fallback when value is None or NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    return float(value)


# --------------------------------------------------------------------------- #
# Optuna utilities
# --------------------------------------------------------------------------- #


def early_stopping_callback_composite(
    thresholds: dict[str, float],
    patience: int = 1000,
) -> callable:
    """Early stopping callback for composite optimization."""
    history: list[int] = []

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        history.append(trial.number)

        if _meets_composite_thresholds(trial, thresholds):
            logger.info(
                "Early stopping triggered at trial %s. Thresholds satisfied.",
                trial.number,
            )
            study.stop()
        elif len(history) > patience:
            logger.info(
                "Early stopping triggered after %s trials (patience exceeded).",
                patience,
            )
            study.stop()

    return callback


def _meets_composite_thresholds(
    trial: optuna.trial.FrozenTrial, thresholds: dict[str, float]
) -> bool:
    """Check whether a trial satisfies composite thresholds."""
    if len(trial.values) < 3:
        return False

    kge, low_flow, high_flow = trial.values[:3]
    return (
        kge >= thresholds.get("KGE", 0.0)
        and low_flow >= thresholds.get("low_flow", 0.0)
        and high_flow >= thresholds.get("high_flow", 0.0)
    )


def progress_callback(_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Log progress every 1000 trials."""
    if (trial.number + 1) % 1000 == 0 and len(trial.values) >= 3:
        logger.info(
            "Trial %s :: KGE=%.3f LowFlow=%.3f HighFlow=%.3f",
            trial.number + 1,
            trial.values[0],
            trial.values[1],
            trial.values[2],
        )


def run_optimization(
    data: pd.DataFrame,
    calibration_period: tuple[str, str],
    study_name: str | None = None,
    n_trials: int = 100,
    timeout: int = 600,
    verbose: bool = False,
    warmup_years: int = 2,
    use_detailed: bool = False,
) -> optuna.Study:
    """Execute HBV multi-objective Optuna study."""
    if study_name is None:
        study_name = f"HBV_improved_{pd.Timestamp.now():%Y%m%d_%H%M%S}"

    if use_detailed:

        def objective(trial: optuna.Trial) -> tuple[float, ...]:
            return multi_objective_detailed(
                trial,
                data=data,
                calibration_period=calibration_period,
                warmup_years=warmup_years,
            )

        n_objectives = 6
    else:

        def objective(trial: optuna.Trial) -> tuple[float, ...]:
            return multi_objective_composite(
                trial,
                data=data,
                calibration_period=calibration_period,
                warmup_years=warmup_years,
            )

        n_objectives = 4

    original_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    sampler = optuna.samplers.NSGAIISampler(seed=42)
    study = optuna.create_study(
        directions=["maximize"] * n_objectives,
        sampler=sampler,
        study_name=study_name,
        load_if_exists=True,
    )

    optuna.logging.set_verbosity(original_verbosity)

    callbacks: list[callable] = [progress_callback]
    if not use_detailed:
        callbacks.append(
            early_stopping_callback_composite(
                thresholds={"KGE": 0.85, "low_flow": 0.75, "high_flow": 0.80},
                patience=max(1000, n_trials // 2),
            )
        )

    try:
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=verbose,
            callbacks=callbacks,
            gc_after_trial=True,
        )
    finally:
        optuna.logging.set_verbosity(original_verbosity)

    return study


__all__ = [
    "multi_objective_composite",
    "multi_objective_detailed",
    "run_optimization",
    "early_stopping_callback_composite",
    "progress_callback",
]
