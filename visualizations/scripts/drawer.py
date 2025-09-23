"""Legacy aggregation module for visualisation helpers.

This module maintains the original public API by re-exporting functions from
purpose-specific modules:

* ``boxplots`` – boxplot utilities
* ``hex_maps`` – hexagon-based visualisations
* ``hex_utils`` – reusable geometry and aggregation helpers
* ``russia_maps`` – country-scale plotting routines
* ``metrics`` – tabular metric helpers
"""

from __future__ import annotations

from ._style import configure_fonts
from .boxplots import create_nse_boxplots
from .hex_maps import hexes_plot, hexes_plots_n
from .hex_utils import (
    EQ_AREA_CRS,
    aggregate_nse_to_hex,
    build_hex_grid,
    suggest_hex_radius,
    summarize_hex_coverage,
    to_equal_area,
)
from .metrics import metric_viewer
from .russia_maps import russia_plots, russia_plots_n

# Ensure fonts are configured when importing through the legacy module.
configure_fonts()


__all__ = [
    "EQ_AREA_CRS",
    "aggregate_nse_to_hex",
    "build_hex_grid",
    "create_nse_boxplots",
    "hexes_plot",
    "hexes_plots_n",
    "metric_viewer",
    "russia_plots",
    "russia_plots_n",
    "summarize_hex_coverage",
    "suggest_hex_radius",
    "to_equal_area",
]
