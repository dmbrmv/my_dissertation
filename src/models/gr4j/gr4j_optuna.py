"""Improved GR4J Optuna multi-objective optimization with enhanced flow coverage.

Key improvements:
1. Warm-up period for model initialization (critical for snowpack)
2. Enhanced objective functions targeting specific flow regimes
3. Corrected CemaNeige parameter bounds (per airGR standards)
4. Composite metrics for balanced low/high flow performance
"""

import numpy as np
import optuna  # type: ignore[import-untyped]
import pandas as pd

from src.models.gr4j import model as gr4j
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

logger = setup_logger("main_gr4j_optuna", log_file="logs/gr4j_optuna.log")


def multi_objective_composite(
    trial: optuna.Trial,
    data: pd.DataFrame,
    calibration_period: tuple[str, str],
    warmup_years: int = 2,
) -> tuple[float, float, float, float]:
    """Multi-objective optimization with composite flow regime metrics.

    This approach uses 4 objectives that comprehensively cover all flow regimes:
    1. KGE - Overall balanced performance
    2. Composite low-flow metric (logNSE + invNSE)
    3. Composite high-flow metric (NSE + PFE)
    4. Volume conservation (PBIAS)

    Args:
        trial: Optuna trial object
        data: Hydrometeorological data with warm-up period included
        calibration_period: (start_date, end_date) for calibration (excluding warm-up)
        warmup_years: Number of years to use for warm-up (default 2)

    Returns:
        Tuple of (KGE, low_flow_composite, high_flow_composite, -abs(PBIAS))
    """
    # --- Parameter suggestions with corrected bounds ---
    # GR4J parameters (standard bounds)
    x1 = trial.suggest_float("x1", 10.0, 5000.0, log=True)
    x2 = trial.suggest_float("x2", -30.0, 30.0)
    x3 = trial.suggest_float("x3", 1.0, 6000.0, log=True)
    x4 = trial.suggest_float("x4", 0.05, 30.0)

    # CemaNeige parameters (CORRECTED per airGR standards)
    ctg = trial.suggest_float("ctg", 0.0, 1.0)  # Was 0-5, now 0-1
    kf = trial.suggest_float("kf", 1.0, 10.0)  # Was 1-15, now 1-10
    tt = trial.suggest_float("tt", -2.0, 2.5)  # Was -5-5, now -2-2.5

    params = [x1, x2, x3, x4, ctg, kf, tt]

    # --- Calculate warm-up period ---
    calib_start = pd.to_datetime(calibration_period[0])
    calib_end = pd.to_datetime(calibration_period[1])
    warmup_start = calib_start - pd.DateOffset(years=warmup_years)

    # Ensure warm-up data is available
    if warmup_start < data.index[0]:
        logger.warning(
            f"Insufficient data for {warmup_years}-year warm-up. "
            f"Using available data from {data.index[0]}"
        )
        warmup_start = data.index[0]

    # --- Run simulation with warm-up ---
    try:
        warmup_data = data[warmup_start:calib_end]
        q_sim_full = gr4j.simulation(warmup_data, params)

        # Extract calibration period (exclude warm-up)
        n_warmup_days = len(data[warmup_start:calib_start]) - 1
        q_sim = q_sim_full[n_warmup_days:]
        q_obs = data[calib_start:calib_end]["q_mm_day"].values

        # Ensure arrays are same length (handle edge cases)
        min_len = min(len(q_obs), len(q_sim))
        q_obs = q_obs[:min_len]
        q_sim = q_sim[:min_len]

    except Exception as e:
        logger.error(f"Simulation failed for trial {trial.number}: {e}")
        # Return worst possible values
        return -999.0, -999.0, -999.0, -999.0

    # --- Calculate composite objective functions ---
    kge = kling_gupta_efficiency(np.asarray(q_obs), np.asarray(q_sim))

    # Low-flow composite (equal weight to log and inverse NSE)
    low_flow_comp = composite_low_flow_metric(
        np.asarray(q_obs), np.asarray(q_sim), weights=(0.5, 0.5)
    )

    # High-flow composite (70% NSE, 30% peak error)
    high_flow_comp = composite_high_flow_metric(
        np.asarray(q_obs), np.asarray(q_sim), weights=(0.7, 0.3)
    )

    # Volume conservation
    pbias_abs = abs(percent_bias(np.asarray(q_obs), np.asarray(q_sim)))

    # Handle NaN values (assign worst score)
    kge = kge if not pd.isna(kge) else -999.0
    low_flow_comp = low_flow_comp if not pd.isna(low_flow_comp) else -999.0
    high_flow_comp = high_flow_comp if not pd.isna(high_flow_comp) else -999.0
    pbias_abs = pbias_abs if not pd.isna(pbias_abs) else 999.0

    return kge, low_flow_comp, high_flow_comp, -pbias_abs


def multi_objective_detailed(
    trial: optuna.Trial,
    data: pd.DataFrame,
    calibration_period: tuple[str, str],
    warmup_years: int = 2,
) -> tuple[float, float, float, float, float, float]:
    """Multi-objective optimization with detailed transformation pyramid.

    This approach uses 6 objectives covering the full flow spectrum:
    1. KGE - Overall balanced performance
    2. NSE (normal) - High flows
    3. NSE (sqrt) - Medium flows
    4. logNSE - Low flows
    5. invNSE - Very low flows
    6. Volume conservation (PBIAS)

    Use this for detailed diagnosis when composite approach doesn't suffice.

    Args:
        trial: Optuna trial object
        data: Hydrometeorological data with warm-up period included
        calibration_period: (start_date, end_date) for calibration (excluding warm-up)
        warmup_years: Number of years to use for warm-up (default 2)

    Returns:
        Tuple of (KGE, NSE, NSE_sqrt, logNSE, invNSE, -abs(PBIAS))
    """
    # --- Parameter suggestions (same as composite) ---
    x1 = trial.suggest_float("x1", 10.0, 5000.0, log=True)
    x2 = trial.suggest_float("x2", -30.0, 30.0)
    x3 = trial.suggest_float("x3", 1.0, 6000.0, log=True)
    x4 = trial.suggest_float("x4", 0.05, 30.0)
    ctg = trial.suggest_float("ctg", 0.0, 1.0)
    kf = trial.suggest_float("kf", 1.0, 10.0)
    tt = trial.suggest_float("tt", -2.0, 2.5)
    params = [x1, x2, x3, x4, ctg, kf, tt]

    # --- Simulation with warm-up (same as composite) ---
    calib_start = pd.to_datetime(calibration_period[0])
    calib_end = pd.to_datetime(calibration_period[1])
    warmup_start = calib_start - pd.DateOffset(years=warmup_years)

    if warmup_start < data.index[0]:
        warmup_start = data.index[0]

    try:
        warmup_data = data[warmup_start:calib_end]
        q_sim_full = gr4j.simulation(warmup_data, params)
        n_warmup_days = len(data[warmup_start:calib_start]) - 1
        q_sim = q_sim_full[n_warmup_days:]
        q_obs = data[calib_start:calib_end]["q_mm_day"].values
        min_len = min(len(q_obs), len(q_sim))
        q_obs = q_obs[:min_len]
        q_sim = q_sim[:min_len]
    except Exception as e:
        logger.error(f"Simulation failed for trial {trial.number}: {e}")
        return -999.0, -999.0, -999.0, -999.0, -999.0, -999.0

    # --- Calculate detailed metrics ---
    metrics = evaluate_model(q_obs, q_sim)
    kge = metrics["KGE"]
    nse = metrics["NSE"]
    lognse = metrics["logNSE"]

    # Additional transformations
    nse_sqrt = sqrt_nse(np.asarray(q_obs), np.asarray(q_sim))

    # Import here to avoid circular dependency
    from src.utils.metrics_enhanced import inverse_nse

    inv_nse = inverse_nse(np.asarray(q_obs), np.asarray(q_sim))

    pbias_abs = abs(metrics["PBIAS"])

    # Handle NaN values
    kge = kge if not pd.isna(kge) else -999.0
    nse = nse if not pd.isna(nse) else -999.0
    nse_sqrt = nse_sqrt if not pd.isna(nse_sqrt) else -999.0
    lognse = lognse if not pd.isna(lognse) else -999.0
    inv_nse = inv_nse if not pd.isna(inv_nse) else -999.0
    pbias_abs = pbias_abs if not pd.isna(pbias_abs) else 999.0

    return kge, nse, nse_sqrt, lognse, inv_nse, -pbias_abs


def early_stopping_callback_composite(thresholds: dict | None = None):
    """Optuna callback to stop if composite metric thresholds are met.

    Args:
        thresholds: Dict with keys 'KGE', 'low_flow', 'high_flow'

    Returns:
        Callback function for Optuna
    """
    if thresholds is None:
        thresholds = {"KGE": 0.85, "low_flow": 0.75, "high_flow": 0.80}

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if len(trial.values) < 3:
            return

        kge, low_comp, high_comp = trial.values[0], trial.values[1], trial.values[2]

        if (
            kge is not None
            and low_comp is not None
            and high_comp is not None
            and kge > thresholds["KGE"]
            and low_comp > thresholds["low_flow"]
            and high_comp > thresholds["high_flow"]
        ):
            logger.info(
                f"Early stopping: KGE={kge:.3f}, LowFlow={low_comp:.3f}, "
                f"HighFlow={high_comp:.3f} exceed thresholds."
            )
            study.stop()

    return callback


def progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Log progress every 500 trials with composite metrics."""
    if (trial.number + 1) % 500 == 0 and len(trial.values) >= 3:
        logger.info(
            f"Trial {trial.number + 1}: KGE={trial.values[0]:.2f} "
            f"LowFlow={trial.values[1]:.2f} HighFlow={trial.values[2]:.2f}"
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
    """Run improved Optuna multi-objective optimization for GR4J model.

    Args:
        data: Hydrometeorological data (should include warm-up period)
        calibration_period: (start_date, end_date) for calibration
        study_name: Name for the optimization study
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds
        verbose: Whether to show progress bar
        warmup_years: Years to use for warm-up (default 2)
        use_detailed: If True, use 6-objective detailed approach,
                      else use 4-objective composite (default False)

    Returns:
        Completed Optuna study with Pareto front
    """
    if study_name is None:
        study_name = f"GR4J_improved_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

    # Select objective function
    if use_detailed:
        n_objectives = 6

        def obj_func(trial: optuna.Trial) -> tuple:
            return multi_objective_detailed(trial, data, calibration_period, warmup_years)

        logger.info(f"Using detailed 6-objective optimization: {study_name}")
    else:
        n_objectives = 4

        def obj_func(trial: optuna.Trial) -> tuple:
            return multi_objective_composite(
                trial, data, calibration_period, warmup_years
            )

        logger.info(f"Using composite 4-objective optimization: {study_name}")

    sampler = optuna.samplers.NSGAIISampler(seed=42)
    study = optuna.create_study(
        directions=["maximize"] * n_objectives,
        sampler=sampler,
        study_name=study_name,
        load_if_exists=True,
    )

    try:
        start_time = pd.Timestamp.now()
        logger.info(
            f"Optimization started at {start_time} with {warmup_years}-year warm-up"
        )

        original_verbosity = optuna.logging.get_verbosity()
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        callbacks = [progress_callback]
        if not use_detailed:
            callbacks.append(
                early_stopping_callback_composite(
                    {"KGE": 0.85, "low_flow": 0.75, "high_flow": 0.80}
                )
            )

        study.optimize(
            obj_func,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=verbose,
            callbacks=callbacks,
            gc_after_trial=True,
        )

        optuna.logging.set_verbosity(original_verbosity)

        end_time = pd.Timestamp.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(
            f"Optimization completed in {duration:.1f}s "
            f"with {len(study.best_trials)} Pareto-optimal solutions"
        )

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise

    return study
