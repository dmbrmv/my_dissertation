"""TFT Predictions - Hydrological Forecasting with Darts."""

__version__ = "0.1.0"
__author__ = "TFT Predictions Team"
__description__ = "Hydrological time series forecasting using Temporal Fusion Transformer models"

from src.config.settings import Settings
from src.evaluation.metrics import nse, kge, rmse

__all__ = [
    "Settings",
    "nse", 
    "kge",
    "rmse",
]
