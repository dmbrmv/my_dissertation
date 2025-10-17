"""Pareto front analysis and result saving utilities for GR4J Optuna optimization."""

from datetime import datetime
import json
import os
import pickle

import numpy as np
import optuna
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("main_gr4j_optuna", log_file="logs/gr4j_optuna.log")


def select_best_trial_weighted(
    pareto_trials: list[optuna.trial.FrozenTrial],
    weights: dict[str, float] | None = None,
    method: str = "weighted_sum",
) -> optuna.trial.FrozenTrial:
    """Select the best trial from Pareto-optimal trials using a weighting strategy.

    Args:
        pareto_trials: List of Pareto-optimal trials.
        weights: Metric weights.
        method: 'weighted_sum', 'topsis', or 'compromise'.

    Returns:
        Best trial according to the method.
    """
    if not pareto_trials:
        raise ValueError("pareto_trials list is empty.")
    if weights is None:
        weights = {"KGE": 0.4, "NSE": 0.3, "logNSE": 0.2, "PBIAS": 0.05, "RMSE": 0.05}
    values_matrix = np.array([trial.values for trial in pareto_trials])
    if method == "weighted_sum":
        weight_vector = np.array(
            [weights["KGE"], weights["NSE"], weights["logNSE"], weights["PBIAS"], weights["RMSE"]]
        )
        scores = np.dot(values_matrix, weight_vector)
        best_idx = np.argmax(scores)
    elif method == "topsis":
        norm_matrix = values_matrix / np.sqrt(np.sum(values_matrix**2, axis=0))
        weight_vector = np.array(
            [weights["KGE"], weights["NSE"], weights["logNSE"], weights["PBIAS"], weights["RMSE"]]
        )
        weighted_matrix = norm_matrix * weight_vector
        ideal_solution = np.max(weighted_matrix, axis=0)
        negative_ideal = np.min(weighted_matrix, axis=0)
        dist_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution) ** 2, axis=1))
        dist_to_negative = np.sqrt(np.sum((weighted_matrix - negative_ideal) ** 2, axis=1))
        topsis_scores = dist_to_negative / (dist_to_ideal + dist_to_negative)
        best_idx = np.argmax(topsis_scores)
    elif method == "compromise":
        max_vals = np.max(values_matrix, axis=0)
        min_vals = np.min(values_matrix, axis=0)
        norm_matrix = (values_matrix - min_vals) / (max_vals - min_vals + 1e-10)
        weight_vector = np.array(
            [weights["KGE"], weights["NSE"], weights["logNSE"], weights["PBIAS"], weights["RMSE"]]
        )
        distances = np.sqrt(np.sum(weight_vector * (1 - norm_matrix) ** 2, axis=1))
        best_idx = np.argmin(distances)
    else:
        raise ValueError(f"Unknown method: {method}")
    return pareto_trials[best_idx]


def analyze_pareto_front(pareto_trials: list[optuna.trial.FrozenTrial]) -> pd.DataFrame:
    """Analyze the Pareto front and return a summary DataFrame."""
    if not pareto_trials:
        return pd.DataFrame()
    data = []
    for trial in pareto_trials:
        data.append(
            {
                "trial_number": trial.number,
                "KGE": trial.values[0],
                "NSE": trial.values[1],
                "logNSE": trial.values[2],
                "neg_PBIAS": trial.values[3],
                "neg_RMSE_norm": trial.values[4],
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
    metrics: dict | None = None,
    output_dir: str = "../data/optimization/results",
) -> str:
    """Save optimization results including Pareto front and best parameters."""
    result_dir = f"{output_dir}/{gauge_id}_{dataset_name}"

    os.makedirs(result_dir, exist_ok=True)
    pareto_df = analyze_pareto_front(study.best_trials)
    pareto_df.to_csv(f"{result_dir}/pareto_front.csv", index=False)
    with open(f"{result_dir}/pareto_trials.pkl", "wb") as f:
        pickle.dump(study.best_trials, f)
    if best_parameters:
        with open(f"{result_dir}/best_parameters.json", "w") as f:
            json.dump(best_parameters, f, indent=2)
    if metrics:
        with open(f"{result_dir}/metrics.json", "w") as f:
            clean_metrics = {k: float(v) if hasattr(v, "item") else v for k, v in metrics.items()}
            json.dump(clean_metrics, f, indent=2)
    study_stats = {
        "n_trials": len(study.trials),
        "study_name": study.study_name,
        "directions": [str(d) for d in study.directions],
        "datetime": datetime.now().isoformat(),
        "n_pareto_trials": len(study.best_trials),
    }
    with open(f"{result_dir}/study_info.json", "w") as f:
        json.dump(study_stats, f, indent=2)
    logger.info(f"Saved optimization results to {result_dir}")
    return result_dir
