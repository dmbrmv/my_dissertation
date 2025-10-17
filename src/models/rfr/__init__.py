"""Random Forest Regressor (RFR) model for hydrological forecasting.

This package provides Random Forest-based rainfall-runoff modeling with:
- Multi-objective Optuna optimization
- Temporal feature engineering
- Pareto front analysis
- Parallel processing capabilities
"""

from src.models.rfr.pareto import (
    analyze_pareto_front,
    compare_pareto_fronts,
    load_best_model,
    save_optimization_results,
    select_best_trial_weighted,
)
from src.models.rfr.rfr_optuna import (
    create_temporal_features,
    multi_objective_composite,
    run_optimization,
    train_final_model,
)

__all__ = [
    # Optimization
    "run_optimization",
    "multi_objective_composite",
    "train_final_model",
    "create_temporal_features",
    # Pareto analysis
    "analyze_pareto_front",
    "select_best_trial_weighted",
    "save_optimization_results",
    "load_best_model",
    "compare_pareto_fronts",
]
