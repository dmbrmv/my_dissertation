"""Leave-One-Basin-Out (LOBO) spatial Random Forest for ungauged prediction.

Nested cross-validation strategy:
- Outer loop: LOBO - each basin held out once for testing
- Inner loop: Fixed temporal split (2008-2015 train, 2016-2018 val) for tuning
- Test period: 2019-2020 on the held-out basin

This approach trains a universal model that learns spatial patterns across
catchments and can predict streamflow in ungauged basins.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import optuna  # type: ignore[import-untyped]
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]
import xarray as xr

from src.models.gr4j.pet import pet_oudin
from src.models.rfr.rfr_optuna import create_temporal_features
from src.utils.logger import setup_logger
from src.utils.metrics import (
    kling_gupta_efficiency,
    percent_bias,
)
from src.utils.metrics_enhanced import (
    composite_high_flow_metric,
    composite_low_flow_metric,
)

logger = setup_logger("rfr_lobo", log_file="logs/rfr_lobo.log")


def create_universal_features(
    data: pd.DataFrame,
    gauge_id: str,
    dataset: str,
    latitude: float,
    static_attrs: np.ndarray,
    rolling_windows: list[int] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Create feature matrix for universal spatial model.

    Combines temporal features (rolling windows, PET, cyclic DOY) with
    static watershed attributes.

    Args:
        data: DataFrame with meteorological data and discharge
        gauge_id: Gauge identifier (used for logging)
        dataset: Meteorological dataset name
        latitude: Gauge latitude for PET calculation
        static_attrs: 1D array of static watershed attributes
        rolling_windows: Rolling window sizes in days

    Returns:
        Tuple of (feature DataFrame, number of dynamic features)
    """
    if rolling_windows is None:
        rolling_windows = [2**n for n in range(6)]  # [1, 2, 4, 8, 16, 32]

    # Calculate mean temperature if needed
    if "t_mean_e5l" not in data.columns:
        data["t_mean_e5l"] = (data["t_max_e5l"] + data["t_min_e5l"]) / 2.0

    # Calculate PET
    t_mean_list = data["t_mean_e5l"].tolist()
    day_of_year_list = data.index.dayofyear.tolist()  # type: ignore[attr-defined]
    pet_values = pet_oudin(t_mean_list, day_of_year_list, latitude)

    # Create temporal features
    base_features = [f"prcp_{dataset}", "t_min_e5l", "t_max_e5l"]
    features_df = create_temporal_features(
        data,
        rolling_windows=rolling_windows,
        base_features=base_features,
        pet_values=pet_values,
    )

    # Number of dynamic features (temporal + cyclic DOY + PET)
    feature_cols = [col for col in features_df.columns if col != "q_mm_day"]
    n_dynamic = len(feature_cols)

    # Add static attributes (repeat for each timestep)
    static_matrix = np.tile(static_attrs, (len(features_df), 1))
    static_col_names = [f"static_{i}" for i in range(len(static_attrs))]

    for i, col_name in enumerate(static_col_names):
        features_df[col_name] = static_matrix[:, i]

    logger.debug(
        f"Created features for {gauge_id}: {n_dynamic} dynamic + "
        f"{len(static_attrs)} static = {len(feature_cols) + len(static_attrs)} total"
    )

    return features_df, n_dynamic


def _load_gauge_data(
    gauge_id: str,
    dataset: str,
    gauges_gdf: gpd.GeoDataFrame,
    static_file: Path,
    static_columns: list[str],
    rolling_windows: list[int],
) -> tuple[pd.DataFrame, float, np.ndarray] | None:
    """Load and preprocess data for a single gauge.

    Args:
        gauge_id: Gauge identifier
        dataset: Meteorological dataset name
        gauges_gdf: GeoDataFrame with gauge geometries
        static_file: Path to static attributes CSV
        static_columns: List of static attribute column names
        rolling_windows: Rolling window sizes

    Returns:
        Tuple of (feature DataFrame, latitude, static attributes) or None if error
    """
    try:
        # Check gauge exists
        if gauge_id not in gauges_gdf.index:
            logger.warning(f"Gauge {gauge_id} not in GeoDataFrame, skipping")
            return None

        # Get latitude
        point_geom = gauges_gdf.loc[gauge_id, "geometry"]
        latitude = float(point_geom.y)  # type: ignore[union-attr]

        # Load static attributes
        static_df = pd.read_csv(static_file, dtype={"gauge_id": str})
        static_df.set_index("gauge_id", inplace=True)

        if gauge_id not in static_df.index:
            logger.warning(f"Gauge {gauge_id} not in static attributes, skipping")
            return None

        static_row = static_df.loc[gauge_id][static_columns]
        static_attrs = static_row.values.astype(float)

        # Load meteorological data
        data_path = Path(f"data/nc_all_q/{gauge_id}.nc")
        if not data_path.exists():
            logger.warning(f"Data file not found for gauge {gauge_id}, skipping")
            return None

        with xr.open_dataset(data_path) as ds:
            df = ds.to_dataframe()

        # Check required columns
        required_cols = ["q_mm_day", "t_min_e5l", "t_max_e5l", f"prcp_{dataset}"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(
                f"Missing columns {missing_cols} for {gauge_id} with "
                f"dataset {dataset}, skipping"
            )
            return None

        # Select relevant data
        gauge_data = df.loc["2008":"2020", required_cols].copy()

        if len(gauge_data) == 0:
            logger.warning(f"No data for {gauge_id} in 2008-2020 period, skipping")
            return None

        # Create features
        features_df, _ = create_universal_features(
            gauge_data,
            gauge_id,
            dataset,
            latitude,
            static_attrs,
            rolling_windows,
        )

        return features_df, latitude, static_attrs

    except Exception as e:
        logger.error(f"Error loading data for gauge {gauge_id}: {e}", exc_info=True)
        return None


def _objective_universal(
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
        "min_impurity_decrease", 0.0, 0.01, log=True
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


def _collect_inner_split_data(
    train_gauge_ids: list[str],
    gauge_data_dict: dict,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Collect training/validation data for inner temporal split.

    Args:
        train_gauge_ids: List of training gauge IDs
        gauge_data_dict: Dictionary mapping gauge_id -> data dict

    Returns:
        Tuple of (x_train_list, y_train_list, x_val_list, y_val_list)
    """
    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []

    for train_gauge_id in train_gauge_ids:
        features_df = gauge_data_dict[train_gauge_id]["features"]

        # Inner temporal split: 2008-2015 train, 2016-2018 val
        train_data = features_df.loc["2008":"2015"]
        val_data = features_df.loc["2016":"2018"]

        if len(train_data) == 0 or len(val_data) == 0:
            logger.warning(
                f"Empty train/val data for {train_gauge_id} in inner split, skipping"
            )
            continue

        # Separate features and targets
        feature_cols = [col for col in train_data.columns if col != "q_mm_day"]
        x_train_list.append(train_data[feature_cols].to_numpy())
        y_train_list.append(train_data["q_mm_day"].to_numpy())
        x_val_list.append(val_data[feature_cols].to_numpy())
        y_val_list.append(val_data["q_mm_day"].to_numpy())

    return x_train_list, y_train_list, x_val_list, y_val_list


def _collect_full_training_data(
    train_gauge_ids: list[str],
    gauge_data_dict: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect full training data (2008-2018) from training basins.

    Args:
        train_gauge_ids: List of training gauge IDs
        gauge_data_dict: Dictionary mapping gauge_id -> data dict

    Returns:
        Tuple of (X_full_train, y_full_train)
    """
    x_full_train_list = []
    y_full_train_list = []

    for train_gauge_id in train_gauge_ids:
        features_df = gauge_data_dict[train_gauge_id]["features"]
        full_train_data = features_df.loc["2008":"2018"]

        if len(full_train_data) == 0:
            continue

        feature_cols = [col for col in full_train_data.columns if col != "q_mm_day"]
        x_full_train_list.append(full_train_data[feature_cols].to_numpy())
        y_full_train_list.append(full_train_data["q_mm_day"].to_numpy())

    x_full_train = np.vstack(x_full_train_list)
    y_full_train = np.concatenate(y_full_train_list)

    return x_full_train, y_full_train


def _save_fold_results(
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
    logger.info("Loading data for all gauges...")
    gauge_data_dict = {}

    for gauge_id in gauge_ids:
        result = _load_gauge_data(
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

    available_gauges = list(gauge_data_dict.keys())
    logger.info(f"Successfully loaded {len(available_gauges)}/{len(gauge_ids)} gauges")

    if len(available_gauges) < 2:
        logger.error("Need at least 2 gauges for LOBO cross-validation")
        return {}

    # Create output directory
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    # LOBO outer loop
    lobo_results = {}

    for fold_idx, test_gauge_id in enumerate(available_gauges):
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

        # Collect training data from all training gauges
        x_train_list, y_train_list, x_val_list, y_val_list = _collect_inner_split_data(
            train_gauge_ids, gauge_data_dict
        )

        if not x_train_list or not x_val_list:
            logger.warning(f"No training data for fold {test_gauge_id}, skipping")
            continue

        # Concatenate all training gauges
        x_train_all = np.vstack(x_train_list)  # noqa: N806
        y_train_all = np.concatenate(y_train_list)
        x_val_all = np.vstack(x_val_list)  # noqa: N806
        y_val_all = np.concatenate(y_val_list)

        logger.info(
            f"Inner loop data shapes - Train: {x_train_all.shape}, Val: {x_val_all.shape}"
        )

        # Inner loop: Hyperparameter optimization on pooled training basins
        logger.info(f"Starting hyperparameter optimization (fold {fold_idx + 1})...")

        study = optuna.create_study(
            study_name=f"LOBO_fold_{fold_idx + 1}_{test_gauge_id}",
            directions=["maximize", "maximize", "maximize", "maximize"],
            sampler=optuna.samplers.TPESampler(seed=42 + fold_idx),
        )

        def objective(
            trial: optuna.Trial,
            x_tr: np.ndarray = x_train_all,  # noqa: N803
            y_tr: np.ndarray = y_train_all,
            x_v: np.ndarray = x_val_all,  # noqa: N803
            y_v: np.ndarray = y_val_all,
            nj: int = n_jobs,
        ) -> tuple[float, float, float, float]:
            return _objective_universal(trial, x_tr, y_tr, x_v, y_v, nj)

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
            continue

        if not study.best_trials:
            logger.warning(f"No valid trials for fold {test_gauge_id}, skipping")
            continue

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

        # Train final model on full training data (2008-2018) from training basins
        logger.info("Training universal model on full training data (2008-2018)...")

        x_full_train, y_full_train = _collect_full_training_data(
            train_gauge_ids, gauge_data_dict
        )

        logger.info(f"Full training data shape: {x_full_train.shape}")

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
            continue

        feature_cols = [col for col in test_data.columns if col != "q_mm_day"]
        x_test = test_data[feature_cols].to_numpy()  # noqa: N806
        y_test = test_data["q_mm_day"].to_numpy()

        y_pred = final_model.predict(x_test)
        y_pred = np.maximum(y_pred, 0.0)

        # Calculate test metrics
        from src.utils.metrics import evaluate_model

        test_metrics = evaluate_model(y_test, y_pred)
        test_metrics["n_train_samples"] = float(len(x_full_train))
        test_metrics["n_test_samples"] = float(len(x_test))
        test_metrics["n_train_basins"] = float(len(train_gauge_ids))
        test_metrics["test_basin"] = test_gauge_id
        test_metrics["fold_idx"] = float(fold_idx)

        logger.info(
            f"Test metrics for {test_gauge_id} - "
            f"KGE: {test_metrics.get('KGE', np.nan):.3f}, "
            f"NSE: {test_metrics.get('NSE', np.nan):.3f}, "
            f"PBIAS: {test_metrics.get('PBIAS', np.nan):.2f}%"
        )

        lobo_results[test_gauge_id] = test_metrics

        # Save fold results
        if output_dir is not None:
            fold_dir = output_dir / f"fold_{fold_idx + 1}_{test_gauge_id}"
            _save_fold_results(fold_dir, final_model, test_metrics, best_params, study)

    # Summary statistics
    if lobo_results:
        logger.info(f"\n{'=' * 80}\nLOBO Cross-Validation Summary\n{'=' * 80}")
        logger.info(f"Completed {len(lobo_results)}/{len(available_gauges)} folds")

        metrics_df = pd.DataFrame(lobo_results).T
        for metric in ["KGE", "NSE", "RMSE", "MAE", "PBIAS"]:
            if metric in metrics_df.columns:
                mean_val = metrics_df[metric].mean()
                std_val = metrics_df[metric].std()
                logger.info(f"{metric}: {mean_val:.3f} ± {std_val:.3f}")

        # Save summary
        if output_dir is not None:
            summary_path = output_dir / "lobo_summary.csv"
            metrics_df.to_csv(summary_path)
            logger.info(f"Saved summary to {summary_path}")

    return lobo_results


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
    # Convert Optuna params to sklearn format (same as gauge-specific)
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
