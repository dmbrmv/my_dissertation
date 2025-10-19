"""RFR spatial (LOBO) single-objective KGE optimization - SIMPLIFIED approach.

This script uses Leave-One-Basin-Out cross-validation with single-objective
optimization for spatial generalization testing.

Key improvements over multi-objective:
- Reduced to n_trials=100 (sufficient for RF hyperparameters)
- Clearer calibration target (KGE only)
- No artificial flow regime conflicts
- Minimal logging overhead
"""

import json
import logging
import multiprocessing as mp
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import optuna  # type: ignore[import-untyped]
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]
import xarray as xr

sys.path.append("./")

from src.models.gr4j.pet import pet_oudin
from src.utils.logger import setup_logger
from src.utils.metrics import evaluate_model, kling_gupta_efficiency

Path("logs").mkdir(exist_ok=True)
logger = setup_logger(
    "rfr_spatial_simple", log_file="logs/rfr_spatial_simple.log", level="INFO"
)

# Suppress verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)


def create_temporal_features(
    df: pd.DataFrame, rolling_windows: list[int], latitude: float
) -> pd.DataFrame:
    """Create temporal features for Random Forest."""
    result = df.copy()

    # Calculate PET
    if "day_of_year" not in result.columns:
        result["day_of_year"] = result.index.dayofyear  # type: ignore[attr-defined]

    pet = pet_oudin(result["t_mean"].tolist(), result["day_of_year"].tolist(), latitude)
    result["pet"] = np.asarray(pet, dtype=float)

    # Cyclic temporal encoding
    doy = result["day_of_year"].values
    doy_array = np.asarray(doy, dtype=float)
    result["doy_sin"] = np.sin(2 * np.pi * doy_array / 365.25)
    result["doy_cos"] = np.cos(2 * np.pi * doy_array / 365.25)

    # Rolling window features
    for window in rolling_windows:
        result[f"prcp_roll_{window}"] = (
            result["prcp"].rolling(window=window, min_periods=1).mean()
        )
        result[f"t_mean_roll_{window}"] = (
            result["t_mean"].rolling(window=window, min_periods=1).mean()
        )

    return result


def load_gauge_data(
    gauge_id: str,
    dataset: str,
    gauges_gdf: gpd.GeoDataFrame,
    static_file: Path,
    static_columns: list[str],
    rolling_windows: list[int],
) -> tuple[pd.DataFrame, float, np.ndarray] | None:
    """Load and prepare data for a single gauge."""
    try:
        # Get latitude
        point_geom = gauges_gdf.loc[gauge_id, "geometry"]
        latitude = float(point_geom.y)  # type: ignore[union-attr]

        # Load static attributes
        static_df = pd.read_csv(static_file, dtype={"gauge_id": str})
        static_df.set_index("gauge_id", inplace=True)

        if gauge_id not in static_df.index:
            return None

        static_attrs = static_df.loc[gauge_id][static_columns].values.astype(float)

        # Load time series data
        with xr.open_dataset(f"data/nc_all_q/{gauge_id}.nc") as ds:
            df = ds.to_dataframe()

        df["t_mean"] = (df["t_max_e5l"] + df["t_min_e5l"]) / 2

        # Select relevant columns
        gauge_data = df.loc[
            "2008":"2020",
            [
                "q_mm_day",
                "t_mean",
                f"prcp_{dataset}",
            ],
        ].copy()

        gauge_data.rename(columns={f"prcp_{dataset}": "prcp"}, inplace=True)

        # Create features
        gauge_data = create_temporal_features(gauge_data, rolling_windows, latitude)
        gauge_data.dropna(inplace=True)

        return gauge_data, latitude, static_attrs

    except Exception:
        return None


def run_lobo_fold(
    test_gauge_id: str,
    all_gauge_ids: list[str],
    dataset: str,
    gauges_gdf: gpd.GeoDataFrame,
    static_file: Path,
    static_columns: list[str],
    rolling_windows: list[int],
    n_trials: int,
    timeout: int,
) -> dict[str, float] | None:
    """Run single LOBO fold with single-objective optimization."""
    # Load data for all gauges
    train_data_list = []
    test_data = None
    test_static = None

    for gauge_id in all_gauge_ids:
        result = load_gauge_data(
            gauge_id, dataset, gauges_gdf, static_file, static_columns, rolling_windows
        )

        if result is None:
            continue

        features_df, latitude, static_attrs = result

        if gauge_id == test_gauge_id:
            test_data = features_df
            test_static = static_attrs
        else:
            # Add gauge ID for tracking
            features_df["gauge_id"] = gauge_id
            # Add static attributes as columns
            for i, col in enumerate(static_columns):
                features_df[f"static_{col}"] = static_attrs[i]
            train_data_list.append(features_df)

    if not train_data_list or test_data is None:
        return None

    # Combine training data from multiple gauges
    train_df_combined = pd.concat(train_data_list, axis=0)

    # Split training data into train/val (temporal)
    train_split = train_df_combined.loc["2008":"2015", :].copy()
    val_split = train_df_combined.loc["2016":"2018", :].copy()

    # Test period for held-out gauge
    test_split = test_data.loc["2019":"2020", :].copy()

    if len(train_split) < 100 or len(val_split) < 50 or len(test_split) < 30:
        return None

    # Prepare features
    feature_cols = [
        col
        for col in train_split.columns
        if col not in ["q_mm_day", "day_of_year", "gauge_id"]
    ]

    X_train = train_split[feature_cols].values
    y_train = train_split["q_mm_day"].values

    X_val = val_split[feature_cols].values
    y_val = val_split["q_mm_day"].values

    # Test features (need to add static attrs)
    test_feature_cols = [
        col for col in test_split.columns if col not in ["q_mm_day", "day_of_year"]
    ]
    X_test = test_split[test_feature_cols].values

    # Add static attributes to test
    if test_static is not None:
        n_test = X_test.shape[0]
        static_test = np.tile(test_static, (n_test, 1))
        X_test = np.hstack([X_test, static_test])

    y_test = test_split["q_mm_day"].values

    # Run optimization
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"RF_LOBO_{test_gauge_id}_{dataset}",
    )

    def objective(trial: optuna.Trial) -> float:
        """Single-objective KGE optimization."""
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 5, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        max_features = trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5, 0.7]
        )

        try:
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
                n_jobs=-1,
            )

            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            kge = kling_gupta_efficiency(y_val, y_pred)

            if pd.isna(kge):
                return -999.0

            return kge

        except Exception:
            return -999.0

    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )

    if study.best_trial is None:
        return None

    # Train final model on full training data (2008-2018)
    full_train = train_df_combined.loc["2008":"2018", :].copy()
    X_full = full_train[feature_cols].values
    y_full = full_train["q_mm_day"].values

    best_params = study.best_trial.params

    rf_final = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features=best_params["max_features"],
        random_state=42,
        n_jobs=-1,
    )

    rf_final.fit(X_full, y_full)

    # Predict on test gauge (2019-2020)
    y_pred_test = rf_final.predict(X_test)

    # Calculate metrics
    metrics = evaluate_model(y_test, y_pred_test)

    # Add test gauge ID to metrics for tracking
    metrics["test_gauge_id"] = test_gauge_id
    metrics["dataset"] = dataset

    return metrics


def main() -> None:
    """Run SIMPLIFIED LOBO cross-validation for spatial RF."""
    from src.readers.geom_reader import load_geodata

    # Load gauge data
    _, gauges = load_geodata(folder_depth=".")

    full_gauges = [
        i.stem for i in Path("data/nc_all_q").glob("*.nc") if i.stem in gauges.index
    ]

    # Configuration
    dataset = "e5l"  # ERA5-Land (most reliable)

    save_storage = Path("data/optimization/rfr_spatial_simple/")
    save_storage.mkdir(parents=True, exist_ok=True)

    # Rolling windows
    rolling_windows = [1, 2, 4, 8, 16, 32]

    # Static attributes
    static_file = Path("data/attributes/hydro_atlas_cis_camels.csv")
    static_parameters = [
        "for_pc_sse",
        "crp_pc_sse",
        "inu_pc_ult",
        "ire_pc_sse",
        "lka_pc_use",
        "prm_pc_sse",
        "pst_pc_sse",
        "cly_pc_sav",
        "slt_pc_sav",
        "snd_pc_sav",
        "kar_pc_sse",
        "urb_pc_sse",
        "gwt_cm_sav",
        "lkv_mc_usu",
        "rev_mc_usu",
        "ws_area",
        "ele_mt_sav",
    ]

    if not static_file.exists():
        logger.error(f"Static attributes file not found: {static_file}")
        return

    n_trials = 100
    timeout = 1800  # 30 minutes per fold

    # Log configuration
    logger.info("=" * 80)
    logger.info("SIMPLIFIED LOBO CROSS-VALIDATION (SINGLE-OBJECTIVE)")
    logger.info("=" * 80)
    logger.info("Objective: KGE (Kling-Gupta Efficiency)")
    logger.info(f"Number of basins: {len(full_gauges)}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Rolling windows: {rolling_windows}")
    logger.info(f"Static features: {len(static_parameters)}")
    logger.info(f"Trials per fold: {n_trials} (TPE sampler)")
    logger.info(f"Timeout per fold: {timeout}s ({timeout / 60:.1f} min)")
    logger.info(f"Results: {save_storage}")
    logger.info("=" * 80)

    # Run LOBO folds
    lobo_results = {}
    n_completed = 0

    for test_gauge_id in full_gauges:
        output_file = save_storage / f"{test_gauge_id}_{dataset}_metrics.json"

        if output_file.exists():
            # Load existing results
            with open(output_file) as f:
                metrics = json.load(f)
                lobo_results[test_gauge_id] = metrics
            continue

        metrics = run_lobo_fold(
            test_gauge_id=test_gauge_id,
            all_gauge_ids=full_gauges,
            dataset=dataset,
            gauges_gdf=gauges,
            static_file=static_file,
            static_columns=static_parameters,
            rolling_windows=rolling_windows,
            n_trials=n_trials,
            timeout=timeout,
        )

        if metrics is not None:
            lobo_results[test_gauge_id] = metrics

            # Save results
            with open(output_file, "w") as f:
                json.dump(metrics, f, indent=2)

            n_completed += 1
            logger.info(
                f"✓ [{n_completed}/{len(full_gauges)}] {test_gauge_id}/{dataset}: "
                f"KGE={metrics.get('KGE', np.nan):.3f}, "
                f"NSE={metrics.get('NSE', np.nan):.3f}"
            )
        else:
            logger.error(f"✗ {test_gauge_id}/{dataset}: Failed to run LOBO fold")

    # Save summary
    if lobo_results:
        results_df = pd.DataFrame(lobo_results).T
        summary_file = save_storage / "lobo_summary.csv"
        results_df.to_csv(summary_file)

        logger.info("=" * 80)
        logger.info("✅ LOBO CROSS-VALIDATION COMPLETED!")
        logger.info(f"Completed {len(lobo_results)}/{len(full_gauges)} folds")
        logger.info("=" * 80)
        logger.info("Performance Summary:")
        for metric in ["KGE", "NSE", "RMSE", "MAE", "PBIAS"]:
            if metric in results_df.columns:
                mean_val = results_df[metric].mean()
                median_val = results_df[metric].median()
                logger.info(f"  {metric}: mean={mean_val:.3f}, median={median_val:.3f}")

        logger.info(f"Results saved to: {save_storage}")
        logger.info("=" * 80)
    else:
        logger.error("No LOBO results obtained")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
