"""Feature engineering for regional RFR spatial models.

Handles creation of universal feature matrices combining temporal dynamics
(rolling windows, PET, cyclic encoding) with static watershed attributes.
"""

import numpy as np
import pandas as pd

from src.models.gr4j.pet import pet_oudin
from src.models.rfr.rfr_optuna import create_temporal_features
from src.utils.logger import setup_logger

logger = setup_logger("rfr_spatial_features", log_file="logs/rfr_spatial.log")


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
