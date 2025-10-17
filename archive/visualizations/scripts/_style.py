"""Shared styling configuration for visualization scripts."""

from __future__ import annotations

import matplotlib.pyplot as plt


def configure_fonts() -> None:
    """Apply consistent font settings across all plots."""
    plt.rcParams["font.family"] = "DeJavu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]


# Apply immediately on import so downstream modules inherit the fonts.
configure_fonts()


__all__ = ["configure_fonts"]
