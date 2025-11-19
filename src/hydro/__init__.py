"""Hydrological analysis package for CAMELS-RU dataset.

This package provides comprehensive hydrological metrics and analysis tools
for discharge time series data, organized into thematic modules.
"""

from .base_flow import BaseFlowSeparation, calculate_bfi
from .flow_duration import FlowDurationCurve, calculate_fdc_metrics
from .flow_extremes import FlowExtremes, calculate_extreme_metrics
from .flow_indices import HydrologicalIndices
from .flow_timing import FlowTiming, calculate_timing_metrics
from .flow_variability import FlowVariability, calculate_variability_metrics
from .period_based_metrics import (
    aggregate_period_metrics,
    calculate_comprehensive_metrics,
    calculate_period_metrics,
    calculate_runoff_ratio,
    split_by_period,
)

__all__ = [
    "BaseFlowSeparation",
    "calculate_bfi",
    "FlowDurationCurve",
    "calculate_fdc_metrics",
    "FlowTiming",
    "calculate_timing_metrics",
    "FlowExtremes",
    "calculate_extreme_metrics",
    "FlowVariability",
    "calculate_variability_metrics",
    "HydrologicalIndices",
    "calculate_comprehensive_metrics",
    "split_by_period",
    "calculate_period_metrics",
    "aggregate_period_metrics",
    "calculate_runoff_ratio",
]
