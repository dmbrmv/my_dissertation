import pandas as pd

from src.readers.hydro_data_reader import select_uncorrelated_features
from src.utils.logger import setup_logger

loader_logger = setup_logger("hydro_atlas_loader", log_file="logs/data_loader.log")


def load_static_data(data_path: str, valid_gauges: list[str]) -> pd.DataFrame:
    """Load and filter static data for valid gauges."""
    static_data = pd.read_csv(data_path, dtype={"gauge_id": str}, index_col="gauge_id")

    return static_data.loc[valid_gauges, :]


def get_combined_features(
    static_data: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame]:
    """Select and combine static features."""
    old_static_features = [
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
        "sgr_dk_sav",
        "slp_dg_sav",
        "ws_area",
        "ele_mt_sav",
    ]
    uncorrelated_static_features = select_uncorrelated_features(static_data)
    combined_feature = sorted(set(old_static_features + uncorrelated_static_features))
    combined_features_df = static_data[combined_feature].reset_index()
    loader_logger.info(
        f"Selected {len(combined_feature)} uncorrelated features from static_data."
    )
    return combined_feature, combined_features_df


def normalize_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize the static dataframe columns to [0, 1] range and return normalization coefficients.

    Args:
        df (pd.DataFrame): DataFrame with static features. Must include 'gauge_id' column.

    Returns:
        tuple[pd.DataFrame, dict]: Tuple of (normalized DataFrame, normalization coefficients).

    Raises:
        ValueError: If a column has constant value (zero range).
    """
    normalization_coeffs: dict[str, dict[str, float]] = {}
    df_norm = df.copy()
    for col in df.columns:
        if col == "gauge_id":
            continue
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max == col_min:
            raise ValueError(
                f"Column '{col}' has constant value; cannot normalize to [0, 1]."
            )
        df_norm[col] = (df[col] - col_min) / (col_max - col_min)
        normalization_coeffs[col] = {
            "min": float(col_min),
            "max": float(col_max),
        }
    return df_norm, normalization_coeffs
