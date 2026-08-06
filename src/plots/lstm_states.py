"""Plotting helpers for LSTM hidden/cell state analysis (Chapter 3).

Plot titles, axis labels, and legends use Russian strings (RU_LABELS).
"""

from __future__ import annotations

from collections.abc import Callable

import geopandas as gpd
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analytics.static_analysis import get_cluster_colors
from src.plots.hex_maps import hex_model_distribution_plot, hexes_plots_n, hybrid_cluster_metrics_maps_n

RU_LABELS = {
    "h_n": "скрытый фильтр (h_n)",
    "c_n": "фильтр состояния (c_n)",
    "evaporation": "испарение (GLEAM)",
    "swe": "влагозапас снега (ERA5-Land)",
    "snow_depth": "высота снежного покрова (ERA5-Land)",
    "subsurface": "грунтовый сток (ERA5-Land)",
    "inactive": "неактивная ячейка",
    "evaporation_pos": "испарение (+)",
    "evaporation_neg": "испарение (−)",
    "swe_pos": "влагозапас снега (+)",
    "swe_neg": "влагозапас снега (−)",
    "snow_depth_pos": "высота снежного покрова (+)",
    "snow_depth_neg": "высота снежного покрова (−)",
    "subsurface_pos": "грунтовый сток (+)",
    "subsurface_neg": "грунтовый сток (−)",
}

GROUP_RU_SHORT = {
    "evaporation_pos": "испар. (+)",
    "evaporation_neg": "испар. (−)",
    "swe_pos": "СВЭ (+)",
    "swe_neg": "СВЭ (−)",
    "snow_depth_pos": "снег (+)",
    "snow_depth_neg": "снег (−)",
    "subsurface_pos": "грунт. (+)",
    "subsurface_neg": "грунт. (−)",
    "inactive": "неактив.",
}

FIGURE_DPI = 200
CORR_HEX_LIMITS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SIGNED_CORR_HEX_LIMITS = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
POS_SHARE_LIMITS = [0.0, 0.25, 0.5, 0.75, 1.0]
POS_SHARE_CB_LABELS = ["(−)", "0.25–0.5", "0.5–0.75", "(+)"]
AMBIGUOUS_HEX_LABEL = "Неоднозначно"

PROCESS_GROUP_ORDER = [
    "evaporation_pos",
    "evaporation_neg",
    "swe_pos",
    "swe_neg",
    "snow_depth_pos",
    "snow_depth_neg",
    "subsurface_pos",
    "subsurface_neg",
    "inactive",
]

# Primary/secondary slot indices in get_cluster_colors(30) for each process pair.
_PROCESS_PALETTE_SLOTS: dict[str, tuple[int, int]] = {
    "evaporation": (2, 2),  # green / light green
    "swe": (0, 0),  # blue / light blue
    "snow_depth": (4, 4),  # purple / light purple
    "subsurface": (1, 1),  # orange / light orange
}
BASE_PROCESSES = tuple(_PROCESS_PALETTE_SLOTS.keys())


def build_process_colors() -> dict[str, str]:
    """Build signed process-group colors from the canonical cluster palette."""
    palette = get_cluster_colors(30)
    primary = palette[:15]
    secondary = palette[15:30]

    by_group: dict[str, str] = {"inactive": primary[7]}
    for param, (pos_idx, neg_idx) in _PROCESS_PALETTE_SLOTS.items():
        by_group[f"{param}_pos"] = primary[pos_idx]
        by_group[f"{param}_neg"] = secondary[neg_idx]

    return {group: by_group[group] for group in PROCESS_GROUP_ORDER}


def process_ru_name(param_key: str) -> str:
    return RU_LABELS.get(param_key, param_key)


def group_ru_label(group: str) -> str:
    return GROUP_RU_SHORT.get(group, group.replace("_", " "))


def build_gauge_process_correlation_gdf(
    h_corr: pd.DataFrame,
    c_corr: pd.DataFrame,
    watersheds: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Build watershed GeoDataFrame with per-gauge max |r| and signed r for h_n and c_n.

    Requires polygon watershed geometries (``ws``), not gauge point locations.
    """
    common = h_corr.index.intersection(c_corr.index).intersection(watersheds.index)
    records: list[dict] = []

    for gauge_id in common:
        h_row = h_corr.loc[gauge_id]
        c_row = c_corr.loc[gauge_id]
        if h_row.isna().all() or c_row.isna().all():
            continue

        best_h = h_row.abs().idxmax()
        best_c = c_row.abs().idxmax()
        records.append(
            {
                "gauge_id": gauge_id,
                "r_h_n": float(abs(h_row[best_h])),
                "r_c_n": float(abs(c_row[best_c])),
                "best_h_cell": int(best_h),
                "best_c_cell": int(best_c),
                "signed_r_h_n": float(h_row[best_h]),
                "signed_r_c_n": float(c_row[best_c]),
            }
        )

    df = pd.DataFrame(records).set_index("gauge_id")
    gdf = watersheds.loc[df.index, ["geometry"]].join(df)
    return gdf


def plot_process_correlation_hex_maps(
    corr_gdf: gpd.GeoDataFrame,
    basemap_data: gpd.GeoDataFrame,
    param_key: str,
    *,
    figsize: tuple[float, float] = (15, 10),
) -> tuple[plt.Figure, dict, float, dict]:
    """Hex maps of |r| and signed r for h_n and c_n (2×2 grid)."""
    param_ru = process_ru_name(param_key)
    median_abs_h = corr_gdf["r_h_n"].median()
    median_abs_c = corr_gdf["r_c_n"].median()
    median_signed_h = corr_gdf["signed_r_h_n"].median()
    median_signed_c = corr_gdf["signed_r_c_n"].median()

    metric_cols = ["r_h_n", "r_c_n", "signed_r_h_n", "signed_r_c_n"]
    plot_gdf = corr_gdf[["geometry", *metric_cols]]

    return hexes_plots_n(
        watersheds=plot_gdf,
        basemap_data=basemap_data,
        metric_cols=metric_cols,
        label_list=["а)", "б)", "в)", "г)"],
        nrows=2,
        ncols=2,
        target_ws_per_hex=6.0,
        area_weighted=False,
        min_overlap_share=0.15,
        quantile=0.5,
        list_of_limits_per_metric=[
            CORR_HEX_LIMITS,
            CORR_HEX_LIMITS,
            SIGNED_CORR_HEX_LIMITS,
            SIGNED_CORR_HEX_LIMITS,
        ],
        cmap_names=["RdYlGn", "RdYlGn", "seismic", "seismic"],
        figsize=figsize,
        basemap_alpha=0.85,
        with_histogram=True,
        title_text=[
            f"|r| для скрытого фильтра и {param_ru},\nмедиана = {median_abs_h:.2f}",
            f"|r| для фильтра состояния и {param_ru},\nмедиана = {median_abs_c:.2f}",
            f"Знаковый r (h_n) и {param_ru},\nмедиана = {median_signed_h:.2f}",
            f"Знаковый r (c_n) и {param_ru},\nмедиана = {median_signed_c:.2f}",
        ],
    )


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))


def plot_cell_assignment_grid(
    cell_assignment_df: pd.DataFrame,
    process_colors: dict[str, str],
    hidden_size: int,
    title: str,
    *,
    figsize: tuple[float, float] = (18, 16),
) -> plt.Figure:
    """Plot 16×16 grid of LSTM cell process assignments."""
    fig, ax = plt.subplots(figsize=figsize)
    grid_size = int(np.sqrt(hidden_size))
    cell_grid = np.zeros((grid_size, grid_size, 3))

    for _, row in cell_assignment_df.iterrows():
        cell_id = int(row["cell_id"])
        group = row["primary_group"]
        color = process_colors.get(group, process_colors["inactive"])
        i, j = cell_id // grid_size, cell_id % grid_size
        cell_grid[i, j] = _hex_to_rgb(color)

    ax.imshow(cell_grid, aspect="equal")
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.5)
        ax.axvline(i - 0.5, color="white", linewidth=0.5)

    ax.set_xticks(range(0, grid_size, 2))
    ax.set_yticks(range(0, grid_size, 2))
    ax.set_xlabel("Столбец ячейки", fontsize=12)
    ax.set_ylabel("Строка ячейки", fontsize=12)
    ax.set_title(title, fontsize=14)

    legend_elements = [
        Patch(facecolor=color, label=group_ru_label(group))
        for group, color in process_colors.items()
        if group in cell_assignment_df["primary_group"].values
    ]
    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
    )
    plt.tight_layout()
    return fig


def plot_hn_cn_comparison_grid(
    h_cell_assignment: pd.DataFrame,
    c_cell_assignment: pd.DataFrame,
    process_colors: dict[str, str],
    hidden_size: int,
    *,
    figsize: tuple[float, float] = (28, 12),
) -> plt.Figure:
    """Side-by-side h_n vs c_n cell assignment grids."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    grid_size = int(np.sqrt(hidden_size))

    for ax, cell_assign, title in zip(
        axes,
        [h_cell_assignment, c_cell_assignment],
        [
            f"{RU_LABELS['h_n']}\nкратковременная память",
            f"{RU_LABELS['c_n']}\nдолговременная память",
        ],
        strict=True,
    ):
        cell_grid = np.zeros((grid_size, grid_size, 3))
        for _, row in cell_assign.iterrows():
            cell_id = int(row["cell_id"])
            group = row["primary_group"]
            color = process_colors.get(group, process_colors["inactive"])
            i, j = cell_id // grid_size, cell_id % grid_size
            cell_grid[i, j] = _hex_to_rgb(color)

        ax.imshow(cell_grid, aspect="equal")
        for i in range(grid_size + 1):
            ax.axhline(i - 0.5, color="white", linewidth=0.5)
            ax.axvline(i - 0.5, color="white", linewidth=0.5)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Столбец ячейки")
        ax.set_ylabel("Строка ячейки")

    legend_elements = [
        Patch(facecolor=color, label=group_ru_label(group)) for group, color in process_colors.items()
    ]
    fig.legend(
        handles=legend_elements,
        loc="center right",
        bbox_to_anchor=(1.06, 0.5),
        fontsize=10,
    )
    plt.tight_layout()
    return fig


def plot_dominant_process_hex_map(
    ws_with_process: gpd.GeoDataFrame,
    basemap_data: gpd.GeoDataFrame,
    process_colors: dict[str, str],
    *,
    figsize: tuple[float, float] = (15, 8),
) -> tuple[plt.Figure, plt.Axes, gpd.GeoDataFrame, pd.Series]:
    """Hex map of modal dominant LSTM process per watershed region."""
    plot_gdf = ws_with_process.copy()
    present_groups = [
        g
        for g in PROCESS_GROUP_ORDER
        if g != "inactive" and g in plot_gdf["dominant_group"].dropna().unique()
    ]
    if not present_groups:
        raise ValueError("No dominant process groups available for hex mapping.")

    ru_labels = [group_ru_label(g) for g in present_groups]
    plot_gdf["dominant_group_ru"] = plot_gdf["dominant_group"].map(group_ru_label)
    model_dict = {ru: float(i) for i, ru in enumerate(ru_labels)}
    # hex_model_distribution_plot appends AMBIGUOUS_HEX_LABEL; reserve its color explicitly
    # so get_unified_colors does not cycle back to the first process color (evaporation green).
    model_dict[AMBIGUOUS_HEX_LABEL] = float(len(ru_labels))
    color_list = [process_colors[g] for g in present_groups]
    color_list.append(process_colors.get("inactive", "#95a5a6"))

    return hex_model_distribution_plot(
        watersheds=plot_gdf,
        basemap_data=basemap_data,
        model_col="dominant_group_ru",
        model_dict=model_dict,
        color_list=color_list,
        target_ws_per_hex=6.0,
        min_overlap_share=0.15,
        figsize=figsize,
        with_histogram=True,
        legend_cols=2,
        title=(
            "Доминирующий метеопроцесс в скрытом фильтре (h_n), гексагональная агрегация\n"
            "(нормализация Z-score; знак процесса: (+)/ (−))"
        ),
    )


def plot_cell_meteo_exemplars(
    param_key: str,
    h_corr: pd.DataFrame,
    h_states_all: np.ndarray,
    state_gauge_ids: list[str],
    load_meteo_fn: Callable[[str, str], np.ndarray],
    *,
    n_samples: int = 6,
    pool_size: int = 420,
    random_state: int = 42,
    figsize: tuple[float, float] = (25, 15),
) -> plt.Figure:
    """Dual-panel time series: best h_n cell vs meteo for exemplar gauges."""
    param_ru = process_ru_name(param_key)
    gauge_r = h_corr.abs().max(axis=1).dropna()
    sample_ids = (
        gauge_r.sort_values(ascending=False)
        .head(pool_size)
        .sample(n=min(n_samples, pool_size, len(gauge_r)), random_state=random_state)
        .index.tolist()
    )

    date_range = pd.date_range("2019-01-01", "2020-12-31", freq="D")
    cell_axs = [0, 1, 2, 6, 7, 8]
    meteo_axs = [3, 4, 5, 9, 10, 11]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 3)
    axs = [fig.add_subplot(gs[i, j]) for i in range(4) for j in range(3)]

    for gauge_id, cell_i, meteo_j in zip(sample_ids, cell_axs, meteo_axs, strict=False):
        gauge_idx = state_gauge_ids.index(str(gauge_id))
        best_cell = int(h_corr.loc[gauge_id].abs().idxmax())
        cell_data = h_states_all[gauge_idx, :, best_cell]
        meteo_data = load_meteo_fn(str(gauge_id), param_key)
        min_len = min(len(cell_data), len(meteo_data), len(date_range))
        dr = date_range[:min_len]
        cell_data = cell_data[:min_len]
        meteo_data = meteo_data[:min_len]
        r_val = float(h_corr.loc[gauge_id, best_cell])

        axs[cell_i].plot(dr, cell_data, color="#c0392b", linewidth=0.8)
        axs[cell_i].set_ylabel(f"Ячейка {best_cell}", fontsize=10)
        axs[cell_i].set_title(
            f"Корреляция ячейки скрытого фильтра и {param_ru.split(' (')[0]} "
            f"для поста {gauge_id}, r={r_val:.2f}",
            fontsize=11,
        )
        axs[cell_i].set_xlim(pd.Timestamp("2019-01-01"), pd.Timestamp("2021-01-01"))
        axs[cell_i].grid(False)

        axs[meteo_j].plot(dr, meteo_data, color="#2980b9", linewidth=0.8)
        axs[meteo_j].set_ylabel(param_ru, fontsize=10)
        axs[meteo_j].set_title(
            f"Суммарный {param_ru.split(' (')[0]} на водосборе поста {gauge_id}",
            fontsize=11,
        )
        axs[meteo_j].set_xlim(pd.Timestamp("2019-01-01"), pd.Timestamp("2021-01-01"))
        axs[meteo_j].grid(False)

    plt.tight_layout()
    return fig


def plot_hypothesis_bars(hypothesis_df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart comparing h_n vs c_n mean |r| per process."""
    palette = get_cluster_colors(15)
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(hypothesis_df))
    width = 0.35

    process_labels = [process_ru_name(p) for p in hypothesis_df["Process"]]
    ax.bar(
        x - width / 2,
        hypothesis_df["h_n Mean |r|"],
        width,
        label=RU_LABELS["h_n"],
        color=palette[3],
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        hypothesis_df["c_n Mean |r|"],
        width,
        label=RU_LABELS["c_n"],
        color=palette[0],
        alpha=0.85,
    )

    for i, row in hypothesis_df.iterrows():
        if row["Matches Hypothesis"]:
            ax.axvspan(i - 0.5, i + 0.5, color=palette[2], alpha=0.08)

    ax.set_xticks(x)
    ax.set_xticklabels(process_labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Среднее |r| по ячейкам", fontsize=12)
    ax.set_title(
        "Проверка гипотезы: специализация h_n и c_n по временным масштабам процессов",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def _modal_group(series: pd.Series) -> str:
    modes = series.mode()
    if modes.empty:
        return "inactive"
    return str(modes.iloc[0])


def build_gauge_signed_max_r(
    correlation_matrices: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Per gauge: signed Pearson r at the LSTM cell with max |r| for each meteo param."""
    if not correlation_matrices:
        raise ValueError("correlation_matrices must contain at least one param DataFrame.")

    param_names = list(correlation_matrices.keys())
    common_index = correlation_matrices[param_names[0]].index
    for corr_df in correlation_matrices.values():
        common_index = common_index.intersection(corr_df.index)

    records: list[dict] = []
    for gauge_id in common_index:
        row: dict = {"gauge_id": gauge_id}
        for param in param_names:
            corr_row = correlation_matrices[param].loc[gauge_id]
            if corr_row.isna().all():
                row[f"{param}_signed_max_r"] = np.nan
                continue
            best_cell = corr_row.abs().idxmax()
            row[f"{param}_signed_max_r"] = float(corr_row[best_cell])
        records.append(row)

    return pd.DataFrame(records).set_index("gauge_id")


def build_hybrid_cluster_process_summary(
    h_gauge_dominant: pd.DataFrame,
    hybrid_clusters: pd.Series,
    *,
    gauge_signed_max_r: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Summarize dominant LSTM process assignments within hybrid clusters."""
    cluster_process = h_gauge_dominant.copy()
    cluster_process["hybrid_cluster"] = hybrid_clusters.reindex(cluster_process.index)
    cluster_process = cluster_process.dropna(subset=["hybrid_cluster"])

    crosstab = pd.crosstab(
        cluster_process["hybrid_cluster"],
        cluster_process["dominant_group"],
    )
    for group in PROCESS_GROUP_ORDER:
        if group not in crosstab.columns:
            crosstab[group] = 0
    crosstab = crosstab[[g for g in PROCESS_GROUP_ORDER if g in crosstab.columns]]

    modal = cluster_process.groupby("hybrid_cluster")["dominant_group"].agg(_modal_group)
    modal_ru = modal.map(group_ru_label)

    summary_rows: list[dict] = []
    for cluster in crosstab.index:
        row: dict = {
            "hybrid_cluster": cluster,
            "modal_dominant_group": modal.loc[cluster],
            "modal_dominant_group_ru": modal_ru.loc[cluster],
            "n_gauges": int(crosstab.loc[cluster].sum()),
        }
        for param in BASE_PROCESSES:
            pos_col = f"{param}_pos"
            neg_col = f"{param}_neg"
            pos = int(crosstab.loc[cluster].get(pos_col, 0))
            neg = int(crosstab.loc[cluster].get(neg_col, 0))
            signed_total = pos + neg
            row[f"{param}_pos_share"] = pos / signed_total if signed_total > 0 else np.nan
            row[f"{param}_pos_count"] = pos
            row[f"{param}_neg_count"] = neg
            if gauge_signed_max_r is not None:
                signed_col = f"{param}_signed_max_r"
                cluster_gauges = cluster_process.index[cluster_process["hybrid_cluster"] == cluster]
                shared = cluster_gauges.intersection(gauge_signed_max_r.index)
                vals = gauge_signed_max_r.loc[shared, signed_col].dropna()
                row[f"{param}_median_signed_r"] = float(vals.median()) if not vals.empty else np.nan
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("hybrid_cluster")
    summary_df = summary_df.sort_values("n_gauges", ascending=False)

    crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100

    return {
        "cluster_process": cluster_process,
        "crosstab": crosstab,
        "crosstab_pct": crosstab_pct,
        "summary_df": summary_df,
        "modal_dominant_group": modal,
        "modal_dominant_group_ru": modal_ru,
    }


def prepare_hybrid_cluster_watershed_gdf(
    ws: gpd.GeoDataFrame,
    hybrid_clusters: pd.Series,
    summary_df: pd.DataFrame,
    *,
    hybrid_col: str = "hybrid_class",
) -> gpd.GeoDataFrame:
    """Attach hybrid class and cluster-level summary metrics to watershed polygons."""
    ws_hybrid = ws[["geometry"]].copy()
    ws_hybrid[hybrid_col] = ws_hybrid.index.map(hybrid_clusters)
    ws_hybrid = ws_hybrid.dropna(subset=[hybrid_col])

    for col in summary_df.columns:
        ws_hybrid[col] = ws_hybrid[hybrid_col].map(summary_df[col])

    return ws_hybrid


def dissolve_hybrid_cluster_regions(
    ws_hybrid: gpd.GeoDataFrame,
    hybrid_col: str = "hybrid_class",
    value_cols: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Merge watershed polygons into one region per hybrid cluster."""
    if hybrid_col not in ws_hybrid.columns:
        raise KeyError(f"Column '{hybrid_col}' not found in watershed GeoDataFrame.")

    if value_cols is None:
        value_cols = [col for col in ws_hybrid.columns if col not in {"geometry", hybrid_col}]

    agg_dict = dict.fromkeys(value_cols, "first")
    dissolved = ws_hybrid.dissolve(by=hybrid_col, aggfunc=agg_dict)
    return dissolved.reset_index()


def _category_label(value: str) -> str:
    if value in GROUP_RU_SHORT or value in PROCESS_GROUP_ORDER:
        return group_ru_label(value)
    return str(value)


def _plot_dissolved_cluster_panel(
    ax,
    dissolved: gpd.GeoDataFrame,
    category_col: str,
    category_colors: dict[str, str],
    *,
    basemap_proj: gpd.GeoDataFrame | None = None,
    basemap_crs: str | None = None,
    basemap_alpha: float = 0.85,
    panel_label: str | None = None,
    title: str | None = None,
    with_histogram: bool = False,
    histogram_rect: tuple[float, float, float, float] = (0.05, 0.05, 0.30, 0.24),
    legend_cols: int = 4,
    legend_order: list | None = None,
) -> list:
    """Render dissolved cluster regions with bottom legend; return legend handles."""
    from matplotlib.lines import Line2D

    from src.plots.styling_utils import render_basemap

    if basemap_proj is not None and basemap_crs is not None:
        render_basemap(ax, basemap_proj, basemap_crs, alpha=basemap_alpha)

    plot_gdf = dissolved.copy()
    plot_gdf["_area"] = plot_gdf.geometry.area
    plot_gdf = plot_gdf.sort_values("_area", ascending=False)

    present = sorted(plot_gdf[category_col].dropna().unique(), key=str)
    if legend_order is not None:
        label_order = [
            cat for cat in legend_order if cat in category_colors or str(cat) in category_colors
        ]
    else:
        label_order = [cat for cat in present if cat in category_colors or str(cat) in category_colors]

    for cat in label_order:
        subset = plot_gdf[plot_gdf[category_col] == cat]
        if subset.empty:
            continue
        color = category_colors.get(cat, category_colors.get(str(cat), "#bdc3c7"))
        subset.plot(ax=ax, color=color, edgecolor="black", linewidth=0.4, legend=False)

    if with_histogram and label_order:
        counts = plot_gdf[category_col].value_counts().reindex(label_order, fill_value=0)
        ax_hist = ax.inset_axes(histogram_rect)
        bar_colors = [
            category_colors.get(c, category_colors.get(str(c), "#bdc3c7")) for c in label_order
        ]
        bars = ax_hist.bar(
            np.arange(len(label_order)),
            counts.values,
            width=0.9,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.8,
        )
        ax_hist.bar_label(bars, fmt="%d")
        ax_hist.set_xticks(np.arange(len(label_order)))
        ax_hist.set_xticklabels(
            [_category_label(str(c)) for c in label_order], rotation=30, ha="right", fontsize=7
        )
        ax_hist.set_facecolor("white")
        ax_hist.grid(False)
        ax_hist.set(frame_on=False)

    legend_handles = [
        Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            markerfacecolor=category_colors.get(cat, category_colors.get(str(cat), "#bdc3c7")),
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=9,
            label=_category_label(str(cat)),
        )
        for cat in label_order
    ]

    if panel_label:
        ax.text(0, 1, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=14)
    if title:
        ax.set_title(title, fontsize=12)

    return legend_handles


def plot_hybrid_cluster_modal_process_map(
    ws_dissolved: gpd.GeoDataFrame,
    basemap_data: gpd.GeoDataFrame,
    process_colors: dict[str, str],
    *,
    hybrid_col: str = "hybrid_class",
    figsize: tuple[float, float] = (12, 8),
    rus_extent: tuple[float, float, float, float] = (50, 140, 32, 90),
    basemap_alpha: float = 0.85,
) -> plt.Figure:
    """Map modal dominant LSTM process per hybrid cluster (dissolved regions)."""
    from src.plots.styling_utils import get_russia_projection

    aea_crs = get_russia_projection()
    aea_proj4 = aea_crs.proj4_init

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": aea_crs})
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_extent(rus_extent)  # type: ignore

    plot_gdf = ws_dissolved.to_crs(aea_proj4)
    basemap_proj = basemap_data.to_crs(aea_proj4)

    signed_groups = [g for g in PROCESS_GROUP_ORDER if g != "inactive"]
    modal_colors = {group_ru_label(g): process_colors[g] for g in signed_groups}
    full_legend_order = [group_ru_label(g) for g in signed_groups]

    legend_handles = _plot_dissolved_cluster_panel(
        ax,
        plot_gdf,
        "modal_dominant_group_ru",
        modal_colors,
        basemap_proj=basemap_proj,
        basemap_crs=aea_proj4,
        basemap_alpha=basemap_alpha,
        title="Модальный доминирующий метеопроцесс (h_n) по гибридным кластерам",
        with_histogram=True,
        legend_cols=4,
        legend_order=full_legend_order,
    )

    n_legend_cols = min(4, max(1, len(legend_handles)))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=n_legend_cols,
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.16)
    return fig


def plot_hybrid_cluster_sign_tendency_maps_n(
    ws_dissolved: gpd.GeoDataFrame,
    basemap_data: gpd.GeoDataFrame,
    *,
    hybrid_col: str = "hybrid_class",
    figsize: tuple[float, float] = (15, 10),
    basemap_alpha: float = 0.85,
) -> tuple[plt.Figure, pd.DataFrame]:
    """2×2 maps of cluster-median signed max |r| per process family (h_n)."""
    metric_cols = [f"{param}_median_signed_r" for param in BASE_PROCESSES]
    title_text = [
        (
            f"Медианный знаковый r (h_n) и {process_ru_name(param)},\n"
            f"медиана по кластерам = {ws_dissolved[f'{param}_median_signed_r'].median():.2f}"
        )
        for param in BASE_PROCESSES
    ]

    plot_cols = ["geometry", hybrid_col, *metric_cols]
    missing = [c for c in plot_cols if c not in ws_dissolved.columns]
    if missing:
        raise KeyError(f"ws_dissolved is missing columns: {missing}")

    return hybrid_cluster_metrics_maps_n(
        watersheds=ws_dissolved[plot_cols],
        basemap_data=basemap_data,
        cluster_col=hybrid_col,
        metric_cols=metric_cols,
        label_list=["а)", "б)", "в)", "г)"],
        nrows=2,
        ncols=2,
        agg="median",
        list_of_limits=SIGNED_CORR_HEX_LIMITS,
        cmap_name="seismic",
        figsize=figsize,
        basemap_alpha=basemap_alpha,
        with_histogram=True,
        title_text=title_text,
    )


def plot_hybrid_cluster_process_stacked_bars(
    crosstab_pct: pd.DataFrame,
    process_colors: dict[str, str],
    *,
    figsize: tuple[float, float] = (14, 10),
) -> plt.Figure:
    """Stacked bar chart of dominant process shares within each hybrid cluster."""
    plot_df = crosstab_pct.copy()
    if "All" in plot_df.index:
        plot_df = plot_df.drop("All")
    if "All" in plot_df.columns:
        plot_df = plot_df.drop("All", axis=1)

    group_cols = [g for g in PROCESS_GROUP_ORDER if g in plot_df.columns]
    plot_df = plot_df[group_cols]
    plot_df = plot_df.loc[plot_df.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(plot_df))
    bottom = np.zeros(len(plot_df))

    for group in group_cols:
        values = plot_df[group].values
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=group_ru_label(group),
            color=process_colors.get(group, "#bdc3c7"),
            edgecolor="black",
            linewidth=0.3,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index.astype(str), rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Доля постов в кластере, %", fontsize=12)
    ax.set_xlabel("Гибридный кластер", fontsize=12)
    ax.set_title(
        "Доминирующий метеопроцесс в скрытом фильтре (h_n) по гибридным кластерам\n"
        "Доля постов внутри каждого кластера, %",
        fontsize=13,
    )
    ax.set_ylim(0, 100)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        ncol=1,
    )
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


def plot_cluster_process_heatmap(
    crosstab_pct: pd.DataFrame,
    process_colors: dict[str, str],
    *,
    figsize: tuple[float, float] = (16, 12),
) -> plt.Figure:
    """Heatmap of dominant process proportions within hybrid clusters."""
    col_labels = [group_ru_label(c) for c in crosstab_pct.columns]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        crosstab_pct,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        ax=ax,
        xticklabels=col_labels,
        cbar_kws={"label": "Доля в кластере, %"},
    )
    ax.set_xlabel("Доминирующий метеопроцесс", fontsize=12)
    ax.set_ylabel("Гибридный кластер", fontsize=12)
    ax.set_title(
        "Доминирующий метеопроцесс в скрытом фильтре (h_n) по гибридным кластерам\n"
        "Доля постов внутри каждого кластера, %",
        fontsize=14,
    )
    plt.tight_layout()
    return fig


__all__ = [
    "FIGURE_DPI",
    "CORR_HEX_LIMITS",
    "SIGNED_CORR_HEX_LIMITS",
    "AMBIGUOUS_HEX_LABEL",
    "PROCESS_GROUP_ORDER",
    "build_process_colors",
    "RU_LABELS",
    "GROUP_RU_SHORT",
    "process_ru_name",
    "group_ru_label",
    "build_gauge_process_correlation_gdf",
    "build_gauge_signed_max_r",
    "build_hybrid_cluster_process_summary",
    "prepare_hybrid_cluster_watershed_gdf",
    "dissolve_hybrid_cluster_regions",
    "POS_SHARE_LIMITS",
    "POS_SHARE_CB_LABELS",
    "plot_process_correlation_hex_maps",
    "plot_cell_assignment_grid",
    "plot_hn_cn_comparison_grid",
    "plot_dominant_process_hex_map",
    "plot_cell_meteo_exemplars",
    "plot_hypothesis_bars",
    "plot_hybrid_cluster_modal_process_map",
    "plot_hybrid_cluster_sign_tendency_maps_n",
    "plot_hybrid_cluster_process_stacked_bars",
    "plot_cluster_process_heatmap",
]
