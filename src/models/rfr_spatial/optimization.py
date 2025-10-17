"""Optimization utilities for regional RFR spatial models.

Handles hyperparameter optimization for universal Random Forest models
using Optuna with multi-objective optimization.
"""

import numpy as np
import optuna  # type: ignore[import-untyped]
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

logger = setup_logger("rfr_spatial_optimization", log_file="logs/rfr_spatial.log")


def objective_universal(
    trial: optuna.Trial,
    x_train: np.ndarray,  # noqa: N803
    y_train: np.ndarray,
    x_val: np.ndarray,  # noqa: N803
    y_val: np.ndarray,
    n_jobs: int = -1,
) -> tuple[float, float, float, float]:
    """Multi-objective function for universal spatial model optimization.

    Same objective as gauge-specific model: KGE, low-flow, high-flow, PBIAS.

    Args:
        trial: Optuna trial
        x_train: Training features (all basins except held-out, 2008-2015)
        y_train: Training targets
        x_val: Validation features (all basins except held-out, 2016-2018)
        y_val: Validation targets
        n_jobs: Number of parallel jobs

    Returns:
        Tuple of (KGE, low_flow_composite, high_flow_composite, -|PBIAS|)
    """
    # Same hyperparameter space as gauge-specific model
    n_estimators = trial.suggest_int("n_estimators", 200, 1000, step=100)

    max_depth_choice = trial.suggest_categorical("max_depth_choice", ["None", "limited"])
    if max_depth_choice == "None":
        max_depth = None
    else:
        max_depth = trial.suggest_int("max_depth", 10, 50, step=5)

    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    max_features_choice = trial.suggest_categorical(
        "max_features", ["sqrt", "log2", "0.3", "0.5", "0.7", "1.0"]
    )
    if max_features_choice in ["sqrt", "log2"]:
        max_features = max_features_choice
    else:
        max_features = float(max_features_choice)

    bootstrap = trial.suggest_categorical("bootstrap", [True, False])

    if bootstrap:
        max_samples_frac = trial.suggest_float("max_samples", 0.5, 1.0)
        max_samples = max_samples_frac
    else:
        max_samples = None

    min_impurity_decrease = trial.suggest_float(
        "min_impurity_decrease", 1e-6, 0.01, log=True
    )

    # Build and train model
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
            random_state=42,
            verbose=0,
        )

        rfr.fit(x_train, y_train)
        y_pred = rfr.predict(x_val)
        y_pred = np.maximum(y_pred, 0.0)

    except Exception as e:
        logger.error(f"Model training failed for trial {trial.number}: {e}")
        return -999.0, -999.0, -999.0, -999.0

    # Calculate metrics
    try:
        kge = kling_gupta_efficiency(y_val, y_pred)
        low_flow = composite_low_flow_metric(y_val, y_pred)
        high_flow = composite_high_flow_metric(y_val, y_pred)
        pbias = percent_bias(y_val, y_pred)
        pbias_abs = abs(pbias)

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

    logger.debug(
        f"Trial {trial.number}: KGE={kge:.3f}, LowFlow={low_flow:.3f}, "
        f"HighFlow={high_flow:.3f}, |PBIAS|={pbias_abs:.2f}%"
    )

    return (kge, low_flow, high_flow, -pbias_abs)


def train_universal_model(
    x_train: np.ndarray,  # noqa: N803
    y_train: np.ndarray,
    params: dict[str, float | int | str | None],
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Train universal Random Forest model with optimized hyperparameters.

    Args:
        x_train: Training features (pooled from multiple basins)
        y_train: Training targets
        params: Hyperparameter dictionary from Optuna trial
        n_jobs: Number of parallel jobs

    Returns:
        Trained RandomForestRegressor model
    """
    # Convert Optuna params to sklearn format
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

    # Build and train model
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

    logger.info("Training universal Random Forest model...")
    rfr.fit(x_train, y_train)
    logger.info("Training complete")

    return rfr
