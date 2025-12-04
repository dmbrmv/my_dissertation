"""Leave-One-Basin-Out (LOBO) cross-validation coordinator.

Main entry point for nested LOBO cross-validation for regional spatial models.
Coordinates data loading, optimization, training, and evaluation across folds.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import optuna  # type: ignore[import-untyped]
import pandas as pd

from src.models.rfr_spatial.data_loader import (
    collect_full_training_data,
    collect_inner_split_data,
    load_gauge_data,
)
from src.models.rfr_spatial.io import save_fold_results
from src.models.rfr_spatial.optimization import (
    objective_universal,
    train_universal_model,
)
from src.utils.logger import setup_logger
from timeseries_stats.metrics import evaluate_model

logger = setup_logger("rfr_lobo", log_file="logs/rfr_lobo.log")


def run_lobo_optimization(
    gauge_ids: list[str],
    dataset: str,
    gauges_gdf: gpd.GeoDataFrame,
    static_file: Path,
    static_columns: list[str],
    rolling_windows: list[int],
    n_trials: int = 100,
    timeout: int = 3600,
    n_jobs: int = -1,
    output_dir: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Run nested LOBO cross-validation for universal spatial model.

    Outer loop: Leave-One-Basin-Out
    Inner loop: Temporal split (2008-2015 train, 2016-2018 val)
    Test: 2019-2020 on held-out basin

    Args:
        gauge_ids: List of gauge identifiers
        dataset: Meteorological dataset name (e.g., "e5l")
        gauges_gdf: GeoDataFrame with gauge geometries
        static_file: Path to static attributes CSV
        static_columns: List of static attribute column names
        rolling_windows: Rolling window sizes in days
        n_trials: Number of Optuna trials per fold
        timeout: Optimization timeout per fold (seconds)
        n_jobs: Number of parallel jobs for RF training
        output_dir: Directory to save results (None = don't save)

    Returns:
        Dictionary mapping gauge_id -> metrics for test period (2019-2020)
    """
    logger.info(f"Starting LOBO cross-validation with {len(gauge_ids)} basins")
    logger.info(f"Dataset: {dataset}, n_trials: {n_trials}, timeout: {timeout}s")

    # Load all gauge data first
    gauge_data_dict = _load_all_gauges(
        gauge_ids, dataset, gauges_gdf, static_file, static_columns, rolling_windows
    )

    available_gauges = list(gauge_data_dict.keys())
    logger.info(f"Successfully loaded {len(available_gauges)}/{len(gauge_ids)} gauges")

    if len(available_gauges) < 2:
        logger.error("Need at least 2 gauges for LOBO cross-validation")
        return {}

    # Create output directory
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Run LOBO outer loop
    lobo_results = {}
    for fold_idx, test_gauge_id in enumerate(available_gauges):
        fold_result = _run_single_fold(
            fold_idx,
            test_gauge_id,
            available_gauges,
            gauge_data_dict,
            n_trials,
            timeout,
            n_jobs,
            output_dir,
        )
        if fold_result is not None:
            lobo_results[test_gauge_id] = fold_result

    # Save summary statistics
    if lobo_results and output_dir is not None:
        _save_summary(lobo_results, available_gauges, output_dir)

    return lobo_results


def _load_all_gauges(
    gauge_ids: list[str],
    dataset: str,
    gauges_gdf: gpd.GeoDataFrame,
    static_file: Path,
    static_columns: list[str],
    rolling_windows: list[int],
) -> dict:
    """Load data for all gauges.

    Args:
        gauge_ids: List of gauge identifiers
        dataset: Meteorological dataset name
        gauges_gdf: GeoDataFrame with gauge geometries
        static_file: Path to static attributes CSV
        static_columns: List of static attribute column names
        rolling_windows: Rolling window sizes

    Returns:
        Dictionary mapping gauge_id -> data dict
    """
    logger.info("Loading data for all gauges...")
    gauge_data_dict = {}

    for gauge_id in gauge_ids:
        result = load_gauge_data(
            gauge_id,
            dataset,
            gauges_gdf,
            static_file,
            static_columns,
            rolling_windows,
        )
        if result is not None:
            features_df, latitude, static_attrs = result
            gauge_data_dict[gauge_id] = {
                "features": features_df,
                "latitude": latitude,
                "static_attrs": static_attrs,
            }

    return gauge_data_dict


def _run_single_fold(
    fold_idx: int,
    test_gauge_id: str,
    available_gauges: list[str],
    gauge_data_dict: dict,
    n_trials: int,
    timeout: int,
    n_jobs: int,
    output_dir: Path | None,
) -> dict[str, float] | None:
    """Run a single LOBO fold.

    Args:
        fold_idx: Fold index
        test_gauge_id: Gauge ID to hold out for testing
        available_gauges: List of all available gauge IDs
        gauge_data_dict: Dictionary of gauge data
        n_trials: Number of Optuna trials
        timeout: Optimization timeout (seconds)
        n_jobs: Number of parallel jobs
        output_dir: Output directory (None = don't save)

    Returns:
        Test metrics dictionary or None if fold failed
    """
    logger.info(
        f"\n{'=' * 80}\n"
        f"LOBO Fold {fold_idx + 1}/{len(available_gauges)}: "
        f"Testing on {test_gauge_id}\n"
        f"{'=' * 80}"
    )

    # Train gauges: all except test gauge
    train_gauge_ids = [g for g in available_gauges if g != test_gauge_id]
    logger.info(
        f"Training on {len(train_gauge_ids)} basins: {', '.join(train_gauge_ids)}"
    )

    # Collect and prepare training data
    x_train_all, y_train_all, x_val_all, y_val_all = _prepare_inner_data(
        train_gauge_ids, gauge_data_dict
    )

    if (
        x_train_all is None
        or y_train_all is None
        or x_val_all is None
        or (y_val_all is None)
    ):
        logger.warning(f"No training data for fold {test_gauge_id}, skipping")
        return None

    # Run hyperparameter optimization
    best_params = _optimize_hyperparameters(
        fold_idx,
        test_gauge_id,
        x_train_all,
        y_train_all,
        x_val_all,
        y_val_all,
        n_trials,
        timeout,
        n_jobs,
    )

    if best_params is None:
        return None

    # Train final model and evaluate
    return _train_and_evaluate(
        test_gauge_id,
        train_gauge_ids,
        gauge_data_dict,
        best_params,
        n_jobs,
        fold_idx,
        output_dir,
    )


def _prepare_inner_data(
    train_gauge_ids: list[str], gauge_data_dict: dict
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Prepare training/validation data for inner loop.

    Args:
        train_gauge_ids: List of training gauge IDs
        gauge_data_dict: Dictionary of gauge data

    Returns:
        Tuple of (X_train, y_train, X_val, y_val) or (None, None, None, None)
    """
    x_train_list, y_train_list, x_val_list, y_val_list = collect_inner_split_data(
        train_gauge_ids, gauge_data_dict
    )

    if not x_train_list or not x_val_list:
        return None, None, None, None

    x_train_all = np.vstack(x_train_list)
    y_train_all = np.concatenate(y_train_list)
    x_val_all = np.vstack(x_val_list)
    y_val_all = np.concatenate(y_val_list)

    logger.info(
        f"Inner loop data shapes - Train: {x_train_all.shape}, Val: {x_val_all.shape}"
    )

    return x_train_all, y_train_all, x_val_all, y_val_all


def _optimize_hyperparameters(
    fold_idx: int,
    test_gauge_id: str,
    x_train_all: np.ndarray,
    y_train_all: np.ndarray,
    x_val_all: np.ndarray,
    y_val_all: np.ndarray,
    n_trials: int,
    timeout: int,
    n_jobs: int,
) -> dict | None:
    """Run hyperparameter optimization for a fold.

    Args:
        fold_idx: Fold index
        test_gauge_id: Test gauge ID
        x_train_all: Training features
        y_train_all: Training targets
        x_val_all: Validation features
        y_val_all: Validation targets
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds
        n_jobs: Number of parallel jobs

    Returns:
        Best parameters dictionary or None if optimization failed
    """
    logger.info(f"Starting hyperparameter optimization (fold {fold_idx + 1})...")

    study = optuna.create_study(
        study_name=f"LOBO_fold_{fold_idx + 1}_{test_gauge_id}",
        directions=["maximize", "maximize", "maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=42 + fold_idx),
    )

    def objective(
        trial: optuna.Trial,
        x_tr: np.ndarray = x_train_all,
        y_tr: np.ndarray = y_train_all,
        x_v: np.ndarray = x_val_all,
        y_v: np.ndarray = y_val_all,
        nj: int = n_jobs,
    ) -> tuple[float, float, float, float]:
        return objective_universal(trial, x_tr, y_tr, x_v, y_v, nj)

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False,
        )

        logger.info(
            f"Optimization complete: {len(study.trials)} trials, "
            f"{len(study.best_trials)} Pareto-optimal"
        )

    except Exception as e:
        logger.error(f"Optimization failed for fold {test_gauge_id}: {e}")
        return None

    if not study.best_trials:
        logger.warning(f"No valid trials for fold {test_gauge_id}, skipping")
        return None

    # Select best trial (use first Pareto trial or best KGE)
    best_trial = max(study.best_trials, key=lambda t: t.values[0])
    best_params = dict(best_trial.params)

    logger.info(
        f"Best trial {best_trial.number}: "
        f"KGE={best_trial.values[0]:.3f}, "
        f"LowFlow={best_trial.values[1]:.3f}, "
        f"HighFlow={best_trial.values[2]:.3f}, "
        f"-|PBIAS|={best_trial.values[3]:.3f}"
    )

    # Store study for saving later
    best_params["_study"] = study

    return best_params


def _train_and_evaluate(
    test_gauge_id: str,
    train_gauge_ids: list[str],
    gauge_data_dict: dict,
    best_params: dict,
    n_jobs: int,
    fold_idx: int,
    output_dir: Path | None,
) -> dict[str, float]:
    """Train final model and evaluate on test data.

    Args:
        test_gauge_id: Test gauge ID
        train_gauge_ids: Training gauge IDs
        gauge_data_dict: Dictionary of gauge data
        best_params: Best hyperparameters (includes _study)
        n_jobs: Number of parallel jobs
        fold_idx: Fold index
        output_dir: Output directory

    Returns:
        Test metrics dictionary
    """
    logger.info("Training universal model on full training data (2008-2018)...")

    x_full_train, y_full_train = collect_full_training_data(
        train_gauge_ids, gauge_data_dict
    )
    logger.info(f"Full training data shape: {x_full_train.shape}")

    # Extract study and remove from params
    study = best_params.pop("_study", None)

    # Train final model with best hyperparameters
    final_model = train_universal_model(
        x_train=x_full_train,
        y_train=y_full_train,
        params=best_params,
        n_jobs=n_jobs,
    )

    # Test on held-out basin (2019-2020)
    logger.info(f"Testing on held-out basin {test_gauge_id} (2019-2020)...")

    test_features_df = gauge_data_dict[test_gauge_id]["features"]
    test_data = test_features_df.loc["2019":"2020"]

    if len(test_data) == 0:
        logger.warning(f"No test data for {test_gauge_id}, skipping")
        return {}

    feature_cols = [col for col in test_data.columns if col != "q_mm_day"]
    x_test = test_data[feature_cols].to_numpy()
    y_test = test_data["q_mm_day"].to_numpy()

    y_pred = final_model.predict(x_test)
    y_pred = np.maximum(y_pred, 0.0)

    # Calculate test metrics
    test_metrics = evaluate_model(y_test, y_pred)
    test_metrics["n_train_samples"] = float(len(x_full_train))
    test_metrics["n_test_samples"] = float(len(x_test))
    test_metrics["n_train_basins"] = float(len(train_gauge_ids))
    # Store basin ID separately (not in metrics dict for consistency)
    # test_metrics["test_basin"] = test_gauge_id
    test_metrics["fold_idx"] = float(fold_idx)

    logger.info(
        f"Test metrics for {test_gauge_id} - "
        f"KGE: {test_metrics.get('KGE', np.nan):.3f}, "
        f"NSE: {test_metrics.get('NSE', np.nan):.3f}, "
        f"PBIAS: {test_metrics.get('PBIAS', np.nan):.2f}%"
    )

    # Save fold results
    if output_dir is not None and study is not None:
        fold_dir = output_dir / f"fold_{fold_idx + 1}_{test_gauge_id}"
        save_fold_results(fold_dir, final_model, test_metrics, best_params, study)

    return test_metrics


def _save_summary(
    lobo_results: dict, available_gauges: list[str], output_dir: Path
) -> None:
    """Save summary statistics.

    Args:
        lobo_results: Dictionary of results per gauge
        available_gauges: List of available gauge IDs
        output_dir: Output directory
    """
    logger.info(f"\n{'=' * 80}\nLOBO Cross-Validation Summary\n{'=' * 80}")
    logger.info(f"Completed {len(lobo_results)}/{len(available_gauges)} folds")

    metrics_df = pd.DataFrame(lobo_results).T
    for metric in ["KGE", "NSE", "RMSE", "MAE", "PBIAS"]:
        if metric in metrics_df.columns:
            mean_val = metrics_df[metric].mean()
            std_val = metrics_df[metric].std()
            logger.info(f"{metric}: {mean_val:.3f} ± {std_val:.3f}")

    # Save summary
    summary_path = output_dir / "lobo_summary.csv"
    metrics_df.to_csv(summary_path)
    logger.info(f"Saved summary to {summary_path}")
