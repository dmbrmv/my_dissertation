"""Regional spatial Random Forest Regressor with LOBO cross-validation.

Universal (spatial-ungauged) hydrological modeling using Random Forest
with Leave-One-Basin-Out nested cross-validation.
"""

from src.models.rfr_spatial.data_loader import (
    collect_full_training_data,
    collect_inner_split_data,
    load_gauge_data,
)
from src.models.rfr_spatial.features import create_universal_features
from src.models.rfr_spatial.io import save_fold_results
from src.models.rfr_spatial.lobo import run_lobo_optimization
from src.models.rfr_spatial.optimization import (
    objective_universal,
    train_universal_model,
)

__all__ = [
    # Main API
    "run_lobo_optimization",
    # Feature engineering
    "create_universal_features",
    # Data operations
    "load_gauge_data",
    "collect_inner_split_data",
    "collect_full_training_data",
    # Optimization
    "objective_universal",
    "train_universal_model",
    # I/O
    "save_fold_results",
]
