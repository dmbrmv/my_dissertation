import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys
    from pathlib import Path
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import xarray as xr

    # Import custom modules
    from src.analytics.clustering import (
        normalize_seasonal_patterns,
        perform_hierarchical_clustering,
        prepare_clustering_features,
        smooth_discharge_patterns,
    )
    from src.analytics.regime_analysis import (
        build_regime_dataframe,
        group_patterns_by_cluster,
    )
    from src.analytics.static_analysis import (
        get_cluster_colors,
        get_cluster_markers,
        get_marker_size_corrections,
    )
    from src.hydro.base_flow import BaseFlowSeparation
    from src.plots.cluster_plots import (
        plot_dendrogram,
        plot_hydrograph_clusters,
    )
    from src.plots.maps import russia_plots
    from src.readers.geom_reader import load_geodata
    from src.utils.logger import setup_logger

    plt.rcParams["font.family"] = "DeJavu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    folder_depth = ".."
    log = setup_logger("chapter_one", log_file=f"{folder_depth}/logs/chapter_one.log")

    DATA_DIR = Path("data")
    RES_DIR = Path("res/chapter_one")
    FIGURES_DIR = RES_DIR / "figures"
    TABLES_DIR = RES_DIR / "tables"
    DATA_OUT_DIR = RES_DIR / "data"

    for d in [FIGURES_DIR, TABLES_DIR, DATA_OUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    return (
        FIGURES_DIR,
        TABLES_DIR,
        build_regime_dataframe,
        get_cluster_colors,
        get_cluster_markers,
        get_marker_size_corrections,
        gpd,
        group_patterns_by_cluster,
        load_geodata,
        log,
        normalize_seasonal_patterns,
        pd,
        perform_hierarchical_clustering,
        plot_dendrogram,
        plot_hydrograph_clusters,
        plt,
        prepare_clustering_features,
        russia_plots,
        smooth_discharge_patterns,
        xr,
    )


@app.cell
def _(gpd, load_geodata):
    # Load watershed geometries and gauge locations
    ws, gauges = load_geodata(folder_depth=".")
    basemap_data = gpd.read_file("data/geometry/basemap_2023.gpkg")
    common_index = gauges.index.to_list()
    return basemap_data, gauges, ws


@app.cell
def _(
    TABLES_DIR,
    gauges,
    log,
    normalize_seasonal_patterns,
    pd,
    prepare_clustering_features,
    smooth_discharge_patterns,
    ws,
    xr,
):
    # Load discharge time series
    discharge_data = {}
    for gauge_id in ws.index:
        try:
            with xr.open_dataset(f"data/nc_all_q/{gauge_id}.nc") as ds:
                df = ds.to_dataframe()
            discharge_data[gauge_id] = df[["q_mm_day"]].squeeze()
        except Exception as e:
            log.error(f"Error loading {gauge_id}: {e}")
            continue

    # Calculate seasonal cycles
    q_df = {}
    for gauge_id, ts in discharge_data.items():
        seasonal_cycle = ts.groupby([ts.index.month, ts.index.day]).median().values
        q_df[gauge_id] = seasonal_cycle

    q_df = pd.DataFrame.from_dict(q_df, orient="columns")
    q_df = q_df.loc[:, q_df.max() < 50]  # Filter outliers

    # Smooth and normalize
    q_df_smoothed = smooth_discharge_patterns(q_df)
    q_df_normalized = normalize_seasonal_patterns(q_df_smoothed)

    # Prepare clustering features
    q_df_clust = prepare_clustering_features(q_df_normalized, gauges, include_coords=True)
    q_df_clust.index.name = "gauge_id"
    q_df_clust.to_csv(f"{TABLES_DIR}/hydro_scaled_features.csv")

    log.info(f"\nPrepared {len(q_df_clust)} gauges for hydrological clustering")
    return q_df_clust, q_df_normalized


@app.cell
def _(
    FIGURES_DIR,
    perform_hierarchical_clustering,
    plot_dendrogram,
    plt,
    q_df_clust,
):
    n_hydro_clusters = 10
    # Perform hydrological clustering
    hydro_labels, Z_hydro = perform_hierarchical_clustering(
        q_df_clust, n_clusters=n_hydro_clusters
    )

    # Plot dendrogram
    fig_hydro_dendro = plot_dendrogram(
        Z_hydro,
        n_clusters=n_hydro_clusters,
        output_path=f"{FIGURES_DIR}/hydro_dendrogram.png",
        title="Дендрограмма гидрологической кластеризации (метод Уорда)",
    )
    plt.show()
    return hydro_labels, n_hydro_clusters


@app.cell
def _(
    FIGURES_DIR,
    basemap_data,
    build_regime_dataframe,
    gauges,
    get_cluster_colors,
    get_cluster_markers,
    get_marker_size_corrections,
    group_patterns_by_cluster,
    hydro_labels,
    log,
    n_hydro_clusters,
    pd,
    plot_hydrograph_clusters,
    plt,
    q_df_clust,
    q_df_normalized,
    russia_plots,
):
    # Get markers and colors for clusters
    markers_hc = get_cluster_markers(n_hydro_clusters)
    colors_hc = get_cluster_colors(n_hydro_clusters)
    marker_corrections = get_marker_size_corrections()
    # Extract regime characteristics
    gauge_hydro_mapping = pd.Series(
        hydro_labels, index=q_df_clust.index, name="hydro_cluster"
    )
    cluster_patterns = group_patterns_by_cluster(q_df_normalized, gauge_hydro_mapping)

    regime_df = build_regime_dataframe(cluster_patterns, language="ru")
    log.info("\nHydrological Regime Characteristics:")
    log.info(regime_df[["cluster_name", "peak_doy", "regime_type", "cv"]])
    # Plot hydrographs
    fig_hydro = plot_hydrograph_clusters(
        cluster_patterns,
        output_path=f"{FIGURES_DIR}/hydrograph_clusters.png",
    )
    plt.show()
    # Prepare cluster names for geographic visualization
    gauge_clustered = gauges.copy()
    # Use display names with "C{id}:" prefix so sorting works correctly
    gauge_clustered["Cluster_Name"] = [f"Г{cl}" for cl in hydro_labels]

    # Plot map with proper cluster names
    fig_clusters_map = russia_plots(
        gdf_to_plot=gauge_clustered,
        basemap_data=basemap_data,
        distinction_col="Cluster_Name",
        markers_list=markers_hc,
        color_list=colors_hc,
        marker_size_corrections=marker_corrections,
        figsize=(16, 9),
        just_points=True,
        legend_cols=4,
        base_marker_size=25,
        base_linewidth=0.5,
    )

    fig_clusters_map.suptitle(
        f"Hydrological Catchment Clusters - {n_hydro_clusters} Clusters (Ward Method)",
        fontsize=16,
        y=0.98,
    )
    fig_clusters_map.savefig(
        f"{FIGURES_DIR}/hydro_clusters_map.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
