"""Random Forest Regressor Optuna multi-objective optimization.

Key features:
1. Multi-objective optimization covering all flow regimes
2. Hyperparameter bounds based on literature and scikit-learn best practices
3. Feature engineering with rolling windows for temporal dependencies
4. Composite metrics for balanced low/high flow performance
"""

import numpy as np
import optuna  # type: ignore[import-untyped]
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]

from src.utils.logger import setup_logger
from src.utils.metrics import (
    kling_gupta_efficiency,
    percent_bias,
)
from src.utils.metrics_enhanced import (
    composite_high_flow_metric,
    composite_low_flow_metric,
)

logger = setup_logger("rfr_optuna", log_file="logs/rfr_optuna.log")


def _add_pet_feature(
    result: pd.DataFrame, pet_values: np.ndarray, base_features: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Add PET as feature if provided.

    Args:
        result: DataFrame to add PET to
        pet_values: PET values [mm/day]
        base_features: Current base features list

    Returns:
        Tuple of (updated DataFrame, updated base_features list)
    """
    if len(pet_values) == len(result):
        result["pet_mm_day"] = pet_values
        return result, base_features + ["pet_mm_day"]

    logger.warning(
        f"PET array length {len(pet_values)} != data length {len(result)}, skipping PET"
    )
    return result, base_features


def _add_cyclic_temporal_features(result: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic day-of-year encoding (sin/cos).

    Args:
        result: DataFrame with DatetimeIndex

    Returns:
        DataFrame with doy_sin and doy_cos features
    """
    if not hasattr(result.index, "dayofyear"):
        logger.warning(
            "Index does not support dayofyear, skipping cyclic temporal features"
        )
        return result

    doy = result.index.dayofyear  # type: ignore[attr-defined]
    result["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    result["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
    return result


def create_temporal_features(
    data: pd.DataFrame,
    rolling_windows: list[int] | None = None,
    base_features: list[str] | None = None,
    pet_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """Create temporal rolling features for Random Forest.

    Captures temporal dependencies through rolling statistics. For hydrological
    modeling, this is crucial as discharge responds to cumulative precipitation
    and temperature patterns over days/weeks.

    Args:
        data: Input dataframe with meteorological variables (must have DatetimeIndex)
        rolling_windows: List of window sizes in days (default: [1, 2, 4, 8, 16, 32])
        base_features: Base feature columns to create rolling stats from
                      (default: ["prcp_e5l", "t_min_e5l", "t_max_e5l"])
        pet_values: Optional pre-computed PET values [mm/day] to include as feature

    Returns:
        DataFrame with original + rolling + temporal features
    """
    if rolling_windows is None:
        rolling_windows = [2**n for n in range(6)]  # [1, 2, 4, 8, 16, 32]

    if base_features is None:
        base_features = ["prcp_e5l", "t_min_e5l", "t_max_e5l"]

    result = data.copy()

    # Add PET if provided
    if pet_values is not None:
        result, base_features = _add_pet_feature(result, pet_values, base_features)

    # Add cyclic temporal features
    result = _add_cyclic_temporal_features(result)

    # Create rolling features for each base variable
    for feature in base_features:
        if feature not in result.columns:
            logger.warning(f"Feature {feature} not in dataframe, skipping")
            continue

        for window in rolling_windows:
            if "prcp" in feature:
                # Precipitation: sum over window (cumulative)
                result[f"{feature}_sum_{window}d"] = (
                    result[feature].rolling(window=window, min_periods=1).sum()
                )
            elif "pet" in feature:
                # PET: sum over window (cumulative potential water loss)
                result[f"{feature}_sum_{window}d"] = (
                    result[feature].rolling(window=window, min_periods=1).sum()
                )
            else:
                # Temperature: mean over window
                result[f"{feature}_mean_{window}d"] = (
                    result[feature].rolling(window=window, min_periods=1).mean()
                )

    # Drop rows with any NaN (from rolling windows)
    result = result.dropna()

    return result


def multi_objective_composite(
    trial: optuna.Trial,
    X_train: np.ndarray,  # noqa: N803
    y_train: np.ndarray,
    X_val: np.ndarray,  # noqa: N803
    y_val: np.ndarray,
    n_jobs: int = -1,
) -> tuple[float, float, float, float]:
    """Multi-objective Random Forest optimization with composite flow regime metrics.

    This approach uses 4 objectives comprehensively covering all flow regimes:
    1. KGE - Overall balanced performance
    2. Composite low-flow metric (logNSE + invNSE)
    3. Composite high-flow metric (NSE + PFE)
    4. Volume conservation (PBIAS)

    Hyperparameter ranges are based on:
    - Scikit-learn documentation and defaults
    - Recent ML literature for hydrological applications
    - Computational efficiency considerations

    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training targets (discharge)
        X_val: Validation features
        y_val: Validation targets (discharge)
        n_jobs: Number of parallel jobs for RF (-1 = all cores)

    Returns:
        Tuple of (KGE, low_flow_composite, high_flow_composite, -abs(PBIAS))
    """
    # --- Hyperparameter suggestions with optimized bounds ---

    # Number of trees: More is better up to diminishing returns
    # Literature: 200-2000 trees typical for hydrology applications
    n_estimators = trial.suggest_int("n_estimators", 200, 1000, step=100)

    # Max depth: Controls overfitting
    # None = full depth (often best for RF), or limit to 10-50
    max_depth_choice = trial.suggest_categorical("max_depth_choice", ["None", "limited"])
    if max_depth_choice == "None":
        max_depth = None
    else:
        max_depth = trial.suggest_int("max_depth", 10, 50, step=5)

    # Min samples to split: Controls tree granularity
    # Higher values = more regularization
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

    # Min samples per leaf: Prevents tiny leaves
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    # Max features: Number of features to consider per split
    # "sqrt", "log2", or fraction of features
    max_features_choice = trial.suggest_categorical(
        "max_features", ["sqrt", "log2", "0.3", "0.5", "0.7", "1.0"]
    )
    if max_features_choice in ["sqrt", "log2"]:
        max_features = max_features_choice
    else:
        max_features = float(max_features_choice)

    # Bootstrap: Whether to use bootstrap samples
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])

    # Max samples: Fraction of samples to use per tree (if bootstrap=True)
    # Reduces overfitting and speeds training
    if bootstrap:
        max_samples_frac = trial.suggest_float("max_samples", 0.5, 1.0)
        max_samples = max_samples_frac
    else:
        max_samples = None

    # Min impurity decrease: Minimum improvement required to split
    # Acts as regularization
    min_impurity_decrease = trial.suggest_float(
        "min_impurity_decrease", 1e-6, 0.01, log=True
    )

    # --- Build and train Random Forest ---
    try:
        rfr = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            max_samples=max_samples,
            min_impurity_decrease=min_impurity_decrease,
            n_jobs=n_jobs,
            random_state=42,  # Reproducibility
            verbose=0,
        )

        # Train on training data
        rfr.fit(X_train, y_train)

        # Predict on validation data
        y_pred = rfr.predict(X_val)

        # Ensure no negative predictions (physically impossible for discharge)
        y_pred = np.maximum(y_pred, 0.0)

    except Exception as e:
        logger.error(f"Random Forest training failed for trial {trial.number}: {e}")
        # Return worst possible values
        return -999.0, -999.0, -999.0, -999.0

    # --- Calculate composite objective functions ---
    try:
        # 1. KGE - Overall balanced performance
        kge = kling_gupta_efficiency(y_val, y_pred)

        # 2. Low-flow composite (logNSE + invNSE)
        low_flow = composite_low_flow_metric(y_val, y_pred)

        # 3. High-flow composite (NSE + PFE)
        high_flow = composite_high_flow_metric(y_val, y_pred)

        # 4. Volume conservation (minimize absolute bias)
        pbias = percent_bias(y_val, y_pred)
        pbias_abs = abs(pbias)

        # Handle NaN values (return poor performance)
        if np.isnan(kge):
            kge = -999.0
        if np.isnan(low_flow):
            low_flow = -999.0
        if np.isnan(high_flow):
            high_flow = -999.0
        if np.isnan(pbias_abs):
            pbias_abs = 999.0

    except Exception as e:
        logger.error(f"Metric calculation failed for trial {trial.number}: {e}")
        return -999.0, -999.0, -999.0, -999.0

    # Log trial performance
    logger.debug(
        f"Trial {trial.number}: KGE={kge:.3f}, LowFlow={low_flow:.3f}, "
        f"HighFlow={high_flow:.3f}, |PBIAS|={pbias_abs:.2f}%"
    )

    # Return objectives (all to maximize)
    # PBIAS is negated so maximizing -|PBIAS| minimizes absolute bias
    return (kge, low_flow, high_flow, -pbias_abs)


def run_optimization(
    X_train: np.ndarray,  # noqa: N803
    y_train: np.ndarray,
    X_val: np.ndarray,  # noqa: N803
    y_val: np.ndarray,
    study_name: str = "RFR_multiobj",
    n_trials: int = 100,
    timeout: int | None = 3600,
    n_jobs: int = -1,
    verbose: bool = True,
) -> optuna.Study:
    """Run multi-objective Optuna optimization for Random Forest Regressor.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        study_name: Name for the Optuna study
        n_trials: Number of optimization trials
        timeout: Maximum optimization time in seconds (None = no limit)
        n_jobs: Number of parallel jobs for RF training
        verbose: Whether to show progress

    Returns:
        Completed Optuna study with Pareto-optimal trials
    """
    # Configure logging based on verbosity
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    else:
        optuna.logging.set_verbosity(optuna.logging.INFO)

    logger.info(f"Starting optimization: {study_name}")
    logger.info(
        f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, "
        f"n_trials={n_trials}, timeout={timeout}s"
    )

    # Create multi-objective study (maximize all 4 objectives)
    study = optuna.create_study(
        study_name=study_name,
        directions=["maximize", "maximize", "maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Create objective function with fixed data
    def objective(trial: optuna.Trial) -> tuple[float, float, float, float]:
        return multi_objective_composite(
            trial, X_train, y_train, X_val, y_val, n_jobs=n_jobs
        )

    # Run optimization
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=verbose,
        )

        logger.info(f"Optimization complete: {len(study.trials)} trials completed")
        logger.info(f"Pareto-optimal solutions: {len(study.best_trials)}")

        if study.best_trials:
            # Log best trial by first objective (KGE)
            best_kge_trial = max(study.best_trials, key=lambda t: t.values[0])
            logger.info(
                f"Best KGE trial {best_kge_trial.number}: "
                f"KGE={best_kge_trial.values[0]:.3f}, "
                f"LowFlow={best_kge_trial.values[1]:.3f}, "
                f"HighFlow={best_kge_trial.values[2]:.3f}, "
                f"-|PBIAS|={best_kge_trial.values[3]:.3f}"
            )

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise

    return study


def train_final_model(
    X_train: np.ndarray,  # noqa: N803
    y_train: np.ndarray,
    params: dict[str, float | int | str | None],
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Train final Random Forest model with optimized hyperparameters.

    Args:
        X_train: Training features
        y_train: Training targets
        params: Hyperparameter dictionary from Optuna trial
        n_jobs: Number of parallel jobs

    Returns:
        Trained RandomForestRegressor model
    """
    # Convert Optuna params to sklearn format with proper type handling
    max_depth_choice = params.get("max_depth_choice", "None")
    max_depth: int | None
    if max_depth_choice == "None":
        max_depth = None
    else:
        max_depth_val = params.get("max_depth", 20)
        max_depth = int(max_depth_val) if max_depth_val is not None else 20

    max_features_val = params.get("max_features", "sqrt")
    max_features: str | float
    if max_features_val in ["sqrt", "log2"]:
        max_features = str(max_features_val)
    else:
        if max_features_val is not None and max_features_val not in ["sqrt", "log2"]:
            max_features = float(max_features_val)
        else:
            max_features = "sqrt"

    bootstrap_val = params.get("bootstrap", True)
    bootstrap = bool(bootstrap_val) if bootstrap_val is not None else True

    max_samples_val = params.get("max_samples")
    max_samples: float | None = (
        float(max_samples_val) if bootstrap and max_samples_val is not None else None
    )

    n_estimators_val = params.get("n_estimators", 500)
    n_estimators = int(n_estimators_val) if n_estimators_val is not None else 500

    min_samples_split_val = params.get("min_samples_split", 2)
    min_samples_split = (
        int(min_samples_split_val) if min_samples_split_val is not None else 2
    )

    min_samples_leaf_val = params.get("min_samples_leaf", 1)
    min_samples_leaf = (
        int(min_samples_leaf_val) if min_samples_leaf_val is not None else 1
    )

    min_impurity_decrease_val = params.get("min_impurity_decrease", 0.0)
    min_impurity_decrease = (
        float(min_impurity_decrease_val) if min_impurity_decrease_val is not None else 0.0
    )

    # Build model
    rfr = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,  # type: ignore[arg-type]
        bootstrap=bootstrap,
        max_samples=max_samples,
        min_impurity_decrease=min_impurity_decrease,
        n_jobs=n_jobs,
        random_state=42,
        verbose=0,
    )

    # Train
    logger.info("Training final Random Forest model...")
    rfr.fit(X_train, y_train)
    logger.info("Training complete")

    return rfr
