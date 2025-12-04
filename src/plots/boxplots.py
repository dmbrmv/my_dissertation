import matplotlib.pyplot as plt


def create_nse_boxplots(
    df,
    nse_columns,
    figsize=(14, 8),
    color="lightblue",
    save_path=None,
    max_cols=3,
):
    """Create boxplots for NSE columns in a dataframe.

    Args:
        df: DataFrame containing NSE columns
        nse_columns: List of column names containing NSE values
        figsize: Tuple of figure size (width, height)
        color: Color for the boxplots
        save_path: Optional path to save the figure as PNG (e.g., 'output/nse_boxplots.png')
        max_cols: Maximum number of columns in the subplot grid (default: 3)

    Returns:
        matplotlib.figure.Figure: The created figure

    """
    n_cols = len(nse_columns)
    ncols = min(n_cols, max_cols)  # Maximum 3 columns
    nrows = (n_cols + ncols - 1) // ncols  # Calculate required rows

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Ensure axes is always a list for consistent indexing
    if n_cols == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Russian alphabet letters for labeling
    russian_labels = [
        "а)",
        "б)",
        "в)",
        "г)",
        "д)",
        "е)",
        "ж)",
        "з)",
        "и)",
        "к)",
    ]

    for i, col in enumerate(nse_columns):
        ax = axes[i]

        # Calculate statistics
        median_val = df[col].median()
        total_gauges = len(df[col].dropna())
        satisfactory = (df[col] >= 0.5).sum()
        satisfactory_pct = satisfactory / total_gauges * 100 if total_gauges > 0 else 0

        # Create boxplot
        df[col].plot.box(
            ax=ax,
            patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.7},
            medianprops={"color": "red", "linewidth": 2},
            whiskerprops={"linewidth": 1.5},
            capprops={"linewidth": 1.5},
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 4, "alpha": 0.5},
        )

        # Customize the plot
        title = f"Медиана: {median_val:.2f}, NSE≥0.5: {satisfactory} ({satisfactory_pct:.1f}%)"
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_ylabel("NSE", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Add Russian alphabet label in top right corner
        if i < len(russian_labels):
            ax.text(
                0.95,
                0.95,
                russian_labels[i],
                transform=ax.transAxes,
                fontsize=16,
                ha="right",
                va="top",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
            )

        # Limit NSE axis to 0-1.0 for better visibility
        ax.set_ylim(-1.0, 1.0)

        # Add horizontal line at NSE = 0.5
        ax.axhline(
            y=0.5,
            color="orange",
            linestyle="--",
            alpha=0.8,
            linewidth=2,
        )

    # Hide unused subplots
    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    return fig
