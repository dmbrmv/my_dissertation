"""High-level helpers for boxplot visualisations."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from . import _style  # noqa: F401  # ensures font configuration is applied


def create_nse_boxplots(
    df: pd.DataFrame,
    nse_columns: Sequence[str],
    figsize: tuple[float, float] = (14, 8),
    color: str = "lightblue",
    save_path: str | None = None,
    max_cols: int = 3,
):
    """Create boxplots for NSE columns in a dataframe."""
    n_cols = len(nse_columns)
    if n_cols == 0:
        raise ValueError("nse_columns must contain at least one column name.")

    ncols = min(n_cols, max_cols)
    nrows = (n_cols + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if n_cols == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    russian_labels = ["а)", "б)", "в)", "г)", "д)", "е)", "ж)", "з)", "и)", "к)"]

    for i, col in enumerate(nse_columns):
        ax = axes[i]

        median_val = df[col].median()
        total_gauges = len(df[col].dropna())
        satisfactory = (df[col] >= 0.5).sum()
        satisfactory_pct = satisfactory / total_gauges * 100 if total_gauges > 0 else 0

        df[col].plot.box(
            ax=ax,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker="o", markerfacecolor="red", markersize=4, alpha=0.5),
        )

        title = f"Медиана: {median_val:.2f}, NSE≥0.5: {satisfactory} ({satisfactory_pct:.1f}%)"
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_ylabel("NSE", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(True, alpha=0.3, linestyle="--")

        if i < len(russian_labels):
            ax.text(
                0.95,
                0.95,
                russian_labels[i],
                transform=ax.transAxes,
                fontsize=16,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.8, linewidth=2)

    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format="png", dpi=300, bbox_inches="tight")

    return fig


__all__ = ["create_nse_boxplots"]
