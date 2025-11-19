"""Time series statistics package for CAMELS-RU dataset.

This package provides unified statistical analysis tools for time series data
including trend analysis, homogeneity testing, and change point detection.
"""

# Import only the main functions to avoid circular imports
try:
    from .trends import analyze_trends
except ImportError:
    analyze_trends = None

try:
    from .homogeneity import test_homogeneity
except ImportError:
    test_homogeneity = None

try:
    from .change_points import detect_change_points
except ImportError:
    detect_change_points = None

__all__ = [
    "analyze_trends",
    "test_homogeneity",
    "detect_change_points",
]
