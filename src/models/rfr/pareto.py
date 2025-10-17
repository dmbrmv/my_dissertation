"""Pareto front analysis and result saving utilities for RFR Optuna optimization."""

from datetime import datetime
import json
from pathlib import Path
import pickle

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]

from src.utils.logger import setup_logger

logger = setup_logger("rfr_optuna", log_file="logs/rfr_optuna.log")


def select_best_trial_weighted(
    pareto_trials: list[optuna.trial.FrozenTrial],
    weights: dict[str, float] | None = None,
    method: str = "weighted_sum",
) -> optuna.trial.FrozenTrial:
    """Select the best trial from Pareto-optimal trials using a weighting strategy.

    Uses 4-objective optimization: KGE, low_flow, high_flow, PBIAS.

    Args:
        pareto_trials: List of Pareto-optimal trials.
        weights: Metric weights {KGE, low_flow, high_flow, PBIAS}.
                Default: balanced weighting emphasizing flow regimes.
        method: 'weighted_sum', 'topsis', or 'compromise'.

    Returns:
        Best trial according to the method.
    """
    if not pareto_trials:
        raise ValueError("pareto_trials list is empty.")

    if weights is None:
        # Composite optimization defaults (balanced)
        weights = {
            "KGE": 0.25,
            "low_flow": 0.35,
            "high_flow": 0.30,
            "PBIAS": 0.10,
        }

    values_matrix = np.array([trial.values for trial in pareto_trials])

    # Build weight vector (4 objectives)
    weight_vector = np.array(
        [
            weights.get("KGE", 0.25),
            weights.get("low_flow", 0.35),
            weights.get("high_flow", 0.30),
            weights.get("PBIAS", 0.10),
        ]
    )

    if method == "weighted_sum":
        scores = np.dot(values_matrix, weight_vector)
        best_idx = np.argmax(scores)
    elif method == "topsis":
        # TOPSIS: Technique for Order of Preference by Similarity to Ideal Solution
        norm_matrix = values_matrix / np.sqrt(np.sum(values_matrix**2, axis=0))
        weighted_matrix = norm_matrix * weight_vector
        ideal_solution = np.max(weighted_matrix, axis=0)
        negative_ideal = np.min(weighted_matrix, axis=0)
        dist_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution) ** 2, axis=1))
        dist_to_negative = np.sqrt(
            np.sum((weighted_matrix - negative_ideal) ** 2, axis=1)
        )
        topsis_scores = dist_to_negative / (dist_to_ideal + dist_to_negative + 1e-10)
        best_idx = np.argmax(topsis_scores)
    elif method == "compromise":
        # Compromise programming
        max_vals = np.max(values_matrix, axis=0)
        min_vals = np.min(values_matrix, axis=0)
        norm_matrix = (values_matrix - min_vals) / (max_vals - min_vals + 1e-10)
        distances = np.sqrt(np.sum(weight_vector * (1 - norm_matrix) ** 2, axis=1))
        best_idx = np.argmin(distances)
    else:
        raise ValueError(f"Unknown method: {method}")

    return pareto_trials[best_idx]


def analyze_pareto_front(pareto_trials: list[optuna.trial.FrozenTrial]) -> pd.DataFrame:
    """Analyze the Pareto front and return a summary DataFrame.

    Structures the DataFrame for 4-objective optimization:
    KGE, low_flow_composite, high_flow_composite, -abs(PBIAS).

    Args:
        pareto_trials: List of Pareto-optimal trials

    Returns:
        DataFrame with trial metrics and hyperparameters
    """
    if not pareto_trials:
        return pd.DataFrame()

    data = []
    for trial in pareto_trials:
        data.append(
            {
                "trial_number": trial.number,
                "KGE": trial.values[0],
                "low_flow_composite": trial.values[1],
                "high_flow_composite": trial.values[2],
                "neg_PBIAS_abs": trial.values[3],
                "PBIAS": -trial.values[3],
                **trial.params,
            }
        )

    df = pd.DataFrame(data)
    df = df.sort_values("KGE", ascending=False)
    return df


def save_optimization_results(
    study: optuna.Study,
    dataset_name: str,
    gauge_id: str,
    best_parameters: dict | None = None,
    best_model: RandomForestRegressor | None = None,
    metrics: dict | None = None,
    output_dir: Path | str = Path("data/optimization/rfr_results"),
) -> Path:
    """Save RFR optimization results including Pareto front and best model.

    Args:
        study: Completed Optuna study
        dataset_name: Name of meteorological dataset used
        gauge_id: Gauge identifier
        best_parameters: Best hyperparameter set (optional)
        best_model: Trained RandomForestRegressor model (optional)
        metrics: Performance metrics dictionary (optional)
        output_dir: Root directory for saving results

    Returns:
        Path to results directory
    """
    output_dir = Path(output_dir)
    result_dir = output_dir / f"{gauge_id}_{dataset_name}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save Pareto front analysis
    pareto_df = analyze_pareto_front(study.best_trials)
    pareto_df.to_csv(result_dir / "pareto_front.csv", index=False)

    # Save Pareto trials (for later re-analysis)
    with open(result_dir / "pareto_trials.pkl", "wb") as f:
        pickle.dump(study.best_trials, f)

    # Save best hyperparameters
    if best_parameters:
        with open(result_dir / "best_parameters.json", "w") as f:
            json.dump(best_parameters, f, indent=2)

    # Save trained model
    if best_model:
        model_path = result_dir / "best_model.joblib"
        joblib.dump(best_model, model_path, compress=3)
        logger.info(f"Saved best model to {model_path}")

        # Save feature importances
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
            importance_df = pd.DataFrame(
                {"feature_idx": range(len(importances)), "importance": importances}
            )
            importance_df = importance_df.sort_values("importance", ascending=False)
            importance_df.to_csv(result_dir / "feature_importances.csv", index=False)

    # Save performance metrics
    if metrics:
        with open(result_dir / "metrics.json", "w") as f:
            clean_metrics = {
                k: float(v) if hasattr(v, "item") else v for k, v in metrics.items()
            }
            json.dump(clean_metrics, f, indent=2)

    # Save study metadata
    study_stats = {
        "n_trials": len(study.trials),
        "study_name": study.study_name,
        "directions": [str(d) for d in study.directions],
        "datetime": datetime.now().isoformat(),
        "n_pareto_trials": len(study.best_trials),
        "gauge_id": gauge_id,
        "dataset": dataset_name,
    }
    with open(result_dir / "study_info.json", "w") as f:
        json.dump(study_stats, f, indent=2)

    logger.info(f"Saved optimization results to {result_dir}")
    return result_dir


def load_best_model(
    gauge_id: str,
    dataset_name: str,
    output_dir: Path | str = Path("data/optimization/rfr_results"),
) -> RandomForestRegressor:
    """Load a previously trained RFR model.

    Args:
        gauge_id: Gauge identifier
        dataset_name: Dataset name
        output_dir: Root directory where results are saved

    Returns:
        Trained RandomForestRegressor model
    """
    output_dir = Path(output_dir)
    model_path = output_dir / f"{gauge_id}_{dataset_name}" / "best_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    return model


def compare_pareto_fronts(
    study_list: list[tuple[str, optuna.Study]],
) -> pd.DataFrame:
    """Compare Pareto fronts from multiple optimization runs.

    Useful for comparing performance across different datasets or configurations.

    Args:
        study_list: List of (name, study) tuples

    Returns:
        DataFrame comparing best trials across studies
    """
    comparison_data = []

    for name, study in study_list:
        if not study.best_trials:
            logger.warning(f"Study {name} has no Pareto-optimal trials")
            continue

        # Get trial with best KGE
        best_kge_trial = max(study.best_trials, key=lambda t: t.values[0])

        comparison_data.append(
            {
                "study_name": name,
                "n_trials": len(study.trials),
                "n_pareto": len(study.best_trials),
                "best_KGE": best_kge_trial.values[0],
                "best_low_flow": best_kge_trial.values[1],
                "best_high_flow": best_kge_trial.values[2],
                "best_PBIAS": -best_kge_trial.values[3],
                "trial_number": best_kge_trial.number,
            }
        )

    return pd.DataFrame(comparison_data)
