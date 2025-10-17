"""I/O utilities for saving and loading LOBO cross-validation results."""

from pathlib import Path

import optuna  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]

from src.utils.logger import setup_logger

logger = setup_logger("rfr_spatial_io", log_file="logs/rfr_spatial.log")


def save_fold_results(
    fold_dir: Path,
    final_model: RandomForestRegressor,
    test_metrics: dict,
    best_params: dict,
    study: optuna.Study,
) -> None:
    """Save results for a single LOBO fold.

    Args:
        fold_dir: Directory to save results
        final_model: Trained universal model
        test_metrics: Test metrics dictionary
        best_params: Best hyperparameters
        study: Optuna study object
    """
    import json

    import joblib

    fold_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = fold_dir / "universal_model.joblib"
    joblib.dump(final_model, model_path)

    # Save metrics
    metrics_path = fold_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Save best parameters
    params_path = fold_dir / "best_parameters.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    # Save study
    study_path = fold_dir / "optimization_study.pkl"
    joblib.dump(study, study_path)

    logger.info(f"Saved fold results to {fold_dir}")
