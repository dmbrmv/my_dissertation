"""Pareto utilities for HBV Optuna optimization.

These mirror the GR4J helpers to provide consistent result handling across
conceptual models. Functions are thin wrappers around generic analysis logic.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import pickle

import numpy as np
import optuna  # type: ignore[import-untyped]
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("main_hbv_optuna", log_file="logs/hbv_optuna.log")


def select_best_trial_weighted(
    pareto_trials: list[optuna.trial.FrozenTrial],
    weights: dict[str, float] | None = None,
    method: str = "weighted_sum",
) -> optuna.trial.FrozenTrial:
    """Select Pareto-optimal trial using weighted aggregation."""
    if not pareto_trials:
        raise ValueError("pareto_trials list is empty.")

    n_objectives = len(pareto_trials[0].values)

    if weights is None:
        if n_objectives == 4:
            weights = {
                "KGE": 0.25,
                "low_flow": 0.35,
                "high_flow": 0.30,
                "PBIAS": 0.10,
            }
        elif n_objectives == 6:
            weights = {
                "KGE": 0.35,
                "NSE": 0.2,
                "NSE_sqrt": 0.1,
                "logNSE": 0.2,
                "invNSE": 0.1,
                "PBIAS": 0.05,
            }
        else:
            weights = {
                "KGE": 0.4,
                "NSE": 0.3,
                "logNSE": 0.2,
                "PBIAS": 0.05,
                "RMSE": 0.05,
            }

    values_matrix = np.array([trial.values for trial in pareto_trials])

    if n_objectives == 4:
        weight_vector = np.array(
            [
                weights.get("KGE", 0.25),
                weights.get("low_flow", 0.35),
                weights.get("high_flow", 0.30),
                weights.get("PBIAS", 0.10),
            ]
        )
    elif n_objectives == 6:
        weight_vector = np.array(
            [
                weights.get("KGE", 0.35),
                weights.get("NSE", 0.2),
                weights.get("NSE_sqrt", 0.1),
                weights.get("logNSE", 0.2),
                weights.get("invNSE", 0.1),
                weights.get("PBIAS", 0.05),
            ]
        )
    else:
        weight_vector = np.array(
            [
                weights.get("KGE", 0.4),
                weights.get("NSE", 0.3),
                weights.get("logNSE", 0.2),
                weights.get("PBIAS", 0.05),
                weights.get("RMSE", 0.05),
            ]
        )

    if method == "weighted_sum":
        scores = np.dot(values_matrix, weight_vector)
        best_idx = int(np.argmax(scores))
    elif method == "topsis":
        norm_matrix = values_matrix / np.sqrt(np.sum(values_matrix**2, axis=0))
        weighted_matrix = norm_matrix * weight_vector
        ideal = np.max(weighted_matrix, axis=0)
        anti_ideal = np.min(weighted_matrix, axis=0)
        dist_ideal = np.sqrt(np.sum((weighted_matrix - ideal) ** 2, axis=1))
        dist_anti = np.sqrt(np.sum((weighted_matrix - anti_ideal) ** 2, axis=1))
        scores = dist_anti / (dist_ideal + dist_anti + 1e-12)
        best_idx = int(np.argmax(scores))
    elif method == "compromise":
        max_vals = np.max(values_matrix, axis=0)
        min_vals = np.min(values_matrix, axis=0)
        norm_matrix = (values_matrix - min_vals) / (max_vals - min_vals + 1e-12)
        distances = np.sqrt(np.sum(weight_vector * (1 - norm_matrix) ** 2, axis=1))
        best_idx = int(np.argmin(distances))
    else:
        raise ValueError(f"Unknown method: {method}")

    return pareto_trials[best_idx]


def analyze_pareto_front(
    pareto_trials: list[optuna.trial.FrozenTrial],
) -> pd.DataFrame:
    """Return DataFrame summarizing Pareto front trials."""
    if not pareto_trials:
        return pd.DataFrame()

    n_objectives = len(pareto_trials[0].values)
    records: list[dict[str, float | int]] = []

    for trial in pareto_trials:
        base = {"trial_number": trial.number, **trial.params}
        if n_objectives == 4:
            base.update(
                {
                    "KGE": trial.values[0],
                    "low_flow_composite": trial.values[1],
                    "high_flow_composite": trial.values[2],
                    "neg_PBIAS_abs": trial.values[3],
                    "PBIAS": -trial.values[3],
                }
            )
        elif n_objectives == 6:
            base.update(
                {
                    "KGE": trial.values[0],
                    "NSE": trial.values[1],
                    "NSE_sqrt": trial.values[2],
                    "logNSE": trial.values[3],
                    "invNSE": trial.values[4],
                    "neg_PBIAS_abs": trial.values[5],
                    "PBIAS": -trial.values[5],
                }
            )
        else:
            base.update(
                {
                    "KGE": trial.values[0],
                    "NSE": trial.values[1],
                    "logNSE": trial.values[2],
                    "neg_PBIAS": trial.values[3],
                    "neg_RMSE_norm": trial.values[4],
                    "PBIAS": -trial.values[3],
                }
            )
        records.append(base)

    df = pd.DataFrame(records)
    return df.sort_values("KGE", ascending=False)


def save_optimization_results(
    study: optuna.Study,
    dataset_name: str,
    gauge_id: str,
    best_parameters: dict | None = None,
    metrics: dict | None = None,
    output_dir: str | Path = "data/res/hbv_optuna",
) -> Path:
    """Persist Pareto assets, best parameters, and diagnostics to disk.

    Args:
        study: Completed Optuna study.
        dataset_name: Name of meteorological dataset used.
        gauge_id: Gauge identifier.
        best_parameters: Best parameter set (optional).
        metrics: Performance metrics dictionary (optional).
        output_dir: Root directory for saving results.

    Returns:
        Path to results directory.
    """
    output_dir = Path(output_dir)
    result_dir = output_dir / f"{gauge_id}_{dataset_name}"
    result_dir.mkdir(parents=True, exist_ok=True)

    pareto_df = analyze_pareto_front(study.best_trials)
    pareto_df.to_csv(result_dir / "pareto_front.csv", index=False)

    with open(result_dir / "pareto_trials.pkl", "wb") as handle:
        pickle.dump(study.best_trials, handle)

    if best_parameters:
        with open(result_dir / "best_parameters.json", "w") as handle:
            json.dump(best_parameters, handle, indent=2)

    if metrics:
        cleaned = {
            key: float(value) if hasattr(value, "item") else value
            for key, value in metrics.items()
        }
        with open(result_dir / "metrics.json", "w") as handle:
            json.dump(cleaned, handle, indent=2)

    study_stats = {
        "n_trials": len(study.trials),
        "study_name": study.study_name,
        "directions": [str(direction) for direction in study.directions],
        "datetime": datetime.now().isoformat(),
        "n_pareto_trials": len(study.best_trials),
        "gauge_id": gauge_id,
        "dataset": dataset_name,
    }

    with open(result_dir / "study_info.json", "w") as handle:
        json.dump(study_stats, handle, indent=2)

    logger.info("Saved HBV optimization results to %s", result_dir)
    return result_dir


__all__ = [
    "select_best_trial_weighted",
    "analyze_pareto_front",
    "save_optimization_results",
]
