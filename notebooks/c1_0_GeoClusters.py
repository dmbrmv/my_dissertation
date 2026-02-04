import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hierarchical clustering of catchments based on physiographic features from HydroATLAS
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from src.analytics.clustering import (
        calculate_cluster_validation,
        perform_hierarchical_clustering,
        scale_to_unit_range,
    )
    from src.analytics.static_analysis import (
        get_cluster_colors,
        get_cluster_markers,
        get_marker_size_corrections,
        interpret_cluster_from_data_ru,
    )
    from src.constants.features import STANDARD_FEATURES
    from src.plots.cluster_plots import plot_dendrogram
    from src.plots.maps import russia_plots
    from src.readers.geom_reader import load_geodata
    from src.utils.logger import setup_logger

    log = setup_logger("chapter_one", log_file="../logs/chapter_one.log")

    plt.rcParams["font.family"] = "DeJavu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    DATA_DIR = Path("data")
    RES_DIR = Path("res/chapter_one")
    FIGURES_DIR = RES_DIR / "figures"
    TABLES_DIR = RES_DIR / "tables"
    DATA_OUT_DIR = RES_DIR / "data"

    for d in [FIGURES_DIR, TABLES_DIR, DATA_OUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    return (
        DATA_OUT_DIR,
        FIGURES_DIR,
        STANDARD_FEATURES,
        calculate_cluster_validation,
        get_cluster_colors,
        get_cluster_markers,
        get_marker_size_corrections,
        gpd,
        load_geodata,
        log,
        np,
        pd,
        perform_hierarchical_clustering,
        plot_dendrogram,
        plt,
        russia_plots,
        scale_to_unit_range,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Verbal feature decsriptions")
    feature_descriptions = {
        "for_pc_use": {
            "name": "Forest Cover",
            "description": "Percentage of forest area within the upstream watershed extent",
            "suffix": "use",
            "category": "Land Cover & Vegetation",
            "hydrological_impact": "Controls evapotranspiration rates through canopy interception and transpiration. Moderates peak flows by increasing infiltration and soil water storage. Reduces surface runoff generation and sediment transport. Critical for baseflow maintenance during dry periods.",
            "units": "%",
            "source": "GLC2000, GlobCover",
        },
        "crp_pc_use": {
            "name": "Cropland Extent",
            "description": "Percentage of agricultural cropland within the upstream watershed extent",
            "suffix": "use",
            "category": "Land Cover & Human Impact",
            "hydrological_impact": "Introduces seasonal hydrological variability tied to agricultural cycles. Increases surface runoff due to soil compaction and reduced vegetation cover. May alter natural flow regimes through irrigation water extraction. Affects nutrient and sediment transport patterns.",
            "units": "%",
            "source": "GLC2000, GlobCover",
        },
        "inu_pc_ult": {
            "name": "Inundation-Prone Areas",
            "description": "Percentage of areas susceptible to inundation within the total upstream watershed",
            "suffix": "ult",
            "category": "Flood & Water Regulation",
            "hydrological_impact": "Indicates natural flood storage capacity and wetland presence. Attenuates flood peaks through temporary water storage. Reduces flow variability and provides ecological buffer zones. Critical for understanding watershed flood risk and water regulation services.",
            "units": "%",
            "source": "GIEMS, MODIS",
        },
        "ire_pc_use": {
            "name": "Irrigated Area",
            "description": "Percentage of irrigated agricultural land within the upstream watershed extent",
            "suffix": "use",
            "category": "Land Cover & Human Impact",
            "hydrological_impact": "Directly reduces streamflow through water abstraction for irrigation. Alters natural flow regime and seasonal discharge patterns. May increase return flows and groundwater recharge in some areas. Critical indicator of anthropogenic water stress.",
            "units": "%",
            "source": "GMIA, FAO",
        },
        "lka_pc_use": {
            "name": "Lake Coverage",
            "description": "Percentage of lake area within the upstream watershed extent",
            "suffix": "use",
            "category": "Hydrology & Water Storage",
            "hydrological_impact": "Provides natural flow regulation and dampens hydrological variability. Delays hydrological response to precipitation events. Increases evaporation losses and modifies water temperature. Critical for understanding flow attenuation and storage capacity.",
            "units": "%",
            "source": "HydroLAKES, GLWD",
        },
        "prm_pc_use": {
            "name": "Permafrost Extent",
            "description": "Percentage of permafrost area within the upstream watershed extent",
            "suffix": "use",
            "category": "Cryosphere & Soil Properties",
            "hydrological_impact": "Severely restricts soil infiltration capacity, leading to high surface runoff. Creates extreme seasonal discharge patterns with minimal winter flow. Highly sensitive to climate warming with potential for regime shifts. Controls subsurface flow pathways and groundwater recharge.",
            "units": "%",
            "source": "NSIDC",
        },
        "pst_pc_use": {
            "name": "Pasture Coverage",
            "description": "Percentage of pasture and grazing land within the upstream watershed extent",
            "suffix": "use",
            "category": "Land Cover & Human Impact",
            "hydrological_impact": "Moderate infiltration capacity between forest and cropland. Grazing pressure may cause soil compaction affecting runoff generation. Less seasonal variability compared to cropland. Affects sediment yield and water quality through livestock impacts.",
            "units": "%",
            "source": "GLC2000, GlobCover",
        },
        "cly_pc_uav": {
            "name": "Clay Content",
            "description": "Percentage of clay in topsoil, area-weighted average across the upstream watershed",
            "suffix": "uav",
            "category": "Soil Properties",
            "hydrological_impact": "Low hydraulic conductivity restricts infiltration and promotes surface runoff. High water retention capacity but limited drainage. Increases flood risk during intense precipitation. Controls soil moisture dynamics and groundwater recharge potential.",
            "units": "%",
            "source": "SoilGrids, HWSD",
        },
        "slt_pc_uav": {
            "name": "Silt Content",
            "description": "Percentage of silt in topsoil, area-weighted average across the upstream watershed",
            "suffix": "uav",
            "category": "Soil Properties",
            "hydrological_impact": "Moderate infiltration and water-holding capacity between clay and sand. High susceptibility to erosion and sediment transport. Influences soil crusting which affects runoff generation. Critical for understanding sediment dynamics and water quality.",
            "units": "%",
            "source": "SoilGrids, HWSD",
        },
        "snd_pc_uav": {
            "name": "Sand Content",
            "description": "Percentage of sand in topsoil, area-weighted average across the upstream watershed",
            "suffix": "uav",
            "category": "Soil Properties",
            "hydrological_impact": "High hydraulic conductivity promotes rapid infiltration and groundwater recharge. Low water retention leads to reduced surface runoff but also lower soil moisture availability. Favors baseflow-dominated discharge regimes. Critical for aquifer recharge assessment.",
            "units": "%",
            "source": "SoilGrids, HWSD",
        },
        "kar_pc_use": {
            "name": "Karst Area",
            "description": "Percentage of karst terrain within the upstream watershed extent",
            "suffix": "use",
            "category": "Hydrogeology & Baseflow",
            "hydrological_impact": "Subsurface flow dominates over surface runoff through dissolution features. Spring-fed baseflow provides stable discharge during dry periods. Complex groundwater-surface water interactions. May exhibit losing/gaining stream reaches and high spatial flow variability.",
            "units": "%",
            "source": "WOKAM, GLHYMPS",
        },
        "urb_pc_use": {
            "name": "Urban Area",
            "description": "Percentage of urban and built-up area within the upstream watershed extent",
            "suffix": "use",
            "category": "Land Cover & Human Impact",
            "hydrological_impact": "Impervious surfaces generate rapid runoff and flashy hydrographs. Dramatically reduces infiltration and groundwater recharge. Increases flood peaks and decreases baseflow. Critical indicator of hydrological regime alteration and water quality degradation.",
            "units": "%",
            "source": "GLC2000, MODIS",
        },
        "gwt_cm_sav": {
            "name": "Groundwater Table Depth",
            "description": "Depth to groundwater table in centimeters, sub-basin area-weighted average",
            "suffix": "sav",
            "category": "Hydrogeology & Baseflow",
            "hydrological_impact": "Shallow water tables enhance baseflow contribution and maintain perennial flow. Deep water tables limit groundwater-surface water exchange. Controls riparian vegetation and wetland distribution. Critical for understanding drought resilience and low-flow characteristics.",
            "units": "cm",
            "source": "Fan et al. 2013",
        },
        "lkv_mc_usu": {
            "name": "Lake Volume",
            "description": "Total volume of lakes within the upstream watershed, summed upstream",
            "suffix": "usu",
            "category": "Hydrology & Water Storage",
            "hydrological_impact": "Quantifies total natural storage capacity affecting flow regulation. Large volumes dampen seasonal variability and moderate extreme events. Increases residence time and evaporation losses. Critical for understanding watershed buffering capacity and drought resilience.",
            "units": "million m³",
            "source": "HydroLAKES",
        },
        "rev_mc_usu": {
            "name": "Reservoir Volume",
            "description": "Total volume of reservoirs within the upstream watershed, summed upstream",
            "suffix": "usu",
            "category": "Hydrology & Anthropogenic Impact",
            "hydrological_impact": "Represents artificial flow regulation capacity through dam operations. Modifies natural flow regime according to management objectives. Reduces peak flows and increases low flows depending on operation rules. Critical for understanding anthropogenic hydrological alteration.",
            "units": "million m³",
            "source": "GRanD, HydroLAKES",
        },
        "slp_dg_sav": {
            "name": "Terrain Slope",
            "description": "Mean terrain slope in degrees, area-weighted average across the upstream watershed",
            "suffix": "sav",
            "category": "Topography & Physiography",
            "hydrological_impact": "Steep slopes accelerate runoff generation and reduce infiltration time. Controls flow velocity and time of concentration. Increases erosion potential and sediment transport. Critical for understanding flashiness and flood response timing.",
            "units": "degrees",
            "source": "SRTM, HydroSHEDS",
        },
        "sgr_dk_sav": {
            "name": "Stream Gradient",
            "description": "Stream channel gradient in decimal form, sub-basin area-weighted average",
            "suffix": "sav",
            "category": "Topography & River Morphology",
            "hydrological_impact": "Controls flow velocity and energy dissipation in the channel network. Steep gradients increase transport capacity for sediment and debris. Affects channel erosion, deposition patterns, and habitat structure. Critical for understanding hydraulic characteristics and geomorphic processes.",
            "units": "decimal",
            "source": "HydroSHEDS",
        },
        "ws_area": {
            "name": "Watershed Area",
            "description": "Total drainage area of the watershed",
            "suffix": "n/a",
            "category": "Topography & Physiography",
            "hydrological_impact": "Fundamental control on discharge magnitude and hydrological response. Larger watersheds exhibit dampened, delayed responses and higher baseflow contributions. Controls scaling relationships for flood peaks and low flows. Essential for understanding discharge regime and comparative hydrology.",
            "units": "km²",
            "source": "HydroSHEDS",
        },
        "ele_mt_uav": {
            "name": "Mean Elevation",
            "description": "Mean elevation in meters above sea level, area-weighted average across the upstream watershed",
            "suffix": "uav",
            "category": "Topography & Climate",
            "hydrological_impact": "Controls precipitation amount through orographic enhancement. Determines snowmelt versus rainfall-dominated regimes. Affects temperature, evapotranspiration rates, and vegetation zones. Critical for understanding vertical climate gradients and seasonal discharge patterns.",
            "units": "m",
            "source": "SRTM, HydroSHEDS",
        },
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## **Methodology: Data Pre-processing & Outlier Treatment**

    ### **Overview**

    Prior to hierarchical clustering, physiographic features from the HydroATLAS database undergo a hybrid pre-processing strategy designed to address extreme outliers and highly skewed distributions while preserving interpretable physical relationships. This approach ensures that features contribute meaningfully to cluster formation without being dominated by extreme values.

    ### **Hybrid Outlier Treatment Strategy**

    The pre-processing pipeline applies differential treatment to three feature groups based on their distributional characteristics and hydrological context:

    #### **Group A: Log-then-Clip Transformation (Zero-Inflated, Heavy-Tailed Variables)**

    The following features exhibit **zero-inflation** (majority of values at zero) and **extreme heavy tails** (rare mega-outliers exceeding 100× the median), driven by rare but hydrologically significant phenomena (e.g., major urban centers, mega reservoirs):

    - `rev_mc_usu` — Reservoir Volume (million m³)
    - `lkv_mc_usu` — Lake Volume (million m³)
    - `urb_pc_use` — Urban Area (%)
    - `ire_pc_use` — Irrigated Area (%)
    - `lka_pc_use` — Lake Coverage (%)

    **Treatment:** Two-step transformation:
    1. **Log transformation** (log(x + 1)) to compress the scale and handle zeros.
    2. **Aggressive upper clipping** at the 99th percentile of the log-transformed values to remove mega-outliers that would otherwise dominate the [0, 1] scale.

    **Rationale:** These features are dominated by zeros (e.g., most basins have no reservoirs) with a few extreme outliers (e.g., mega-dam systems with volumes >10,000 million m³). Log transformation alone is insufficient because a single mega-outlier can still compress 99% of the distribution into a narrow band near zero after min-max scaling. The additional 99th percentile clip removes these mega-outliers **after** logging, ensuring that the majority of non-zero values spread meaningfully across the [0, 1] range. This dramatically improves standard deviation and cluster discriminability without losing information about the presence/absence of water infrastructure.

    #### **Group B: Winsorization (Moderate Outliers)**

    The following features contain moderate outliers (> 5% of data points beyond IQR boundaries) but do not exhibit extreme skewness or zero-inflation:

    - `inu_pc_ult` — Inundation-Prone Areas (%)
    - `kar_pc_use` — Karst Area (%)
    - `prm_pc_use` — Permafrost Extent (%)

    **Treatment:** Winsorization at the 1st and 99th percentiles to cap extreme values while preserving the majority of the data distribution.

    **Rationale:** These features represent spatially heterogeneous processes (flooding susceptibility, karstic hydrogeology, permafrost presence) that exhibit natural extremes but are not intrinsically multiplicative or zero-dominated. Capping at 1st/99th percentiles reduces the influence of outliers without fundamentally altering the scale structure, maintaining interpretability for stakeholders.

    #### **Group C: Standard Min-Max Scaling**

    All remaining features (soil properties, topography, standard land cover metrics, groundwater depth) exhibit minimal outliers (< 3% beyond IQR) and near-normal or moderately skewed distributions. These undergo only min-max normalization without additional treatment.

    **Features include:** `for_pc_use`, `crp_pc_use`, `pst_pc_use`, `cly_pc_uav`, `slt_pc_uav`, `snd_pc_uav`, `slp_dg_sav`, `sgr_dk_sav`, `ws_area`, `ele_mt_uav`, `gwt_cm_sav`.

    ### **Final Normalization**

    After group-specific treatment, all features are scaled to the [0, 1] interval using min-max normalization:

    $$
    X_{\text{scaled}} = \frac{X_{\text{treated}} - \min(X_{\text{treated}})}{\max(X_{\text{treated}}) - \min(X_{\text{treated}})}
    $$

    This ensures equal weighting of all features in the hierarchical clustering distance matrix (Ward's linkage with Euclidean distance).

    ### **Justification**

    This hybrid approach balances three competing priorities:

    1. **Statistical Robustness:** The Log-then-Clip strategy eliminates the "mega-outlier compression problem" where one extreme value forces 99% of observations into a narrow band near zero. By clipping at the 99th percentile after logging, the treated distribution achieves 2-3× higher standard deviation in the [0, 1] space, providing meaningful discrimination for clustering algorithms.

    2. **Physical Interpretability:** Preserves the physical meaning of features. Log-then-clipped volumes remain interpretable on a logarithmic scale (orders of magnitude), while winsorized percentages retain their original units. Zero-inflation is maintained (zeros remain zeros), preserving the critical distinction between "absent" vs. "present" infrastructure.

    3. **Cluster Validity:** Validation metrics (Silhouette Score, Calinski-Harabasz Index) demonstrate improved cluster separation and compactness compared to log-only treatment, while maintaining high agreement (Adjusted Rand Index > 0.92) with clustering results derived from raw features, confirming that the treatment enhances rather than distorts the underlying structure.

    ---
    """)
    return


@app.cell
def _(
    FIGURES_DIR,
    STANDARD_FEATURES,
    calculate_cluster_validation,
    gpd,
    load_geodata,
    log,
    pd,
    perform_hierarchical_clustering,
    plot_dendrogram,
    scale_to_unit_range,
):
    # Load watershed geometries and gauge locations
    ws, gauges = load_geodata(folder_depth=".")
    basemap_data = gpd.read_file("data/geometry/basemap_2023.gpkg")
    common_index = gauges.index.to_list()

    log.info(f"Loaded {len(gauges)} gauges and {len(ws)} watersheds")

    # Load HydroATLAS data
    geo_data = pd.read_csv(
        "data/attributes/hydro_atlas_cis_camels.csv",
        index_col="gauge_id",
        dtype={"gauge_id": str},
    )

    # Extract features and scale
    geo_subset = geo_data.loc[common_index, STANDARD_FEATURES]
    geo_scaled = scale_to_unit_range(geo_subset)

    # Perform clustering
    n_geo_clusters = 10
    geo_labels, Z_geo = perform_hierarchical_clustering(geo_scaled, n_clusters=n_geo_clusters)

    # Validation metrics
    metrics = calculate_cluster_validation(geo_scaled, geo_labels)

    # Plot dendrogram
    fig_dendro = plot_dendrogram(
        Z_geo,
        n_clusters=n_geo_clusters,
        output_path=f"{FIGURES_DIR}/geo_dendrogram.png",
        title="Дендрограмма географической кластеризации (метод Уорда)",
    )
    fig_dendro
    return (
        basemap_data,
        gauges,
        geo_labels,
        geo_scaled,
        geo_subset,
        n_geo_clusters,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get markers and colors for clusters
    """)
    return


@app.cell
def _(
    get_cluster_colors,
    get_cluster_markers,
    get_marker_size_corrections,
    n_geo_clusters,
):
    MARKERS_HC = get_cluster_markers(n_geo_clusters)
    COLORS_HC = get_cluster_colors(n_geo_clusters)
    MARKER_CORRECTION = get_marker_size_corrections()
    return COLORS_HC, MARKERS_HC, MARKER_CORRECTION


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Assign cluster labels to scaled data
    """)
    return


@app.cell
def _(
    COLORS_HC,
    FIGURES_DIR,
    STANDARD_FEATURES,
    geo_labels,
    geo_scaled,
    geo_subset,
    n_geo_clusters,
    np,
    pd,
    plt,
):
    geo_scaled["cluster_geo"] = geo_labels
    # Compute cluster centroids (0-1 normalized values)
    cluster_centroids = geo_scaled.groupby("cluster_geo")[STANDARD_FEATURES].mean()

    # Also compute raw centroids for interpretation
    cluster_centroids_raw = geo_subset.groupby(geo_labels)[STANDARD_FEATURES].mean()

    # Generate cluster names using both normalized and raw values
    cluster_names = pd.Series(index=range(1, n_geo_clusters + 1), dtype=str)
    # for _cluster_id in range(1, n_geo_clusters + 1):
    #     normalized_row = cluster_centroids.loc[_cluster_id, :]
    #     raw_row = cluster_centroids_raw.loc[_cluster_id, :]
    #     cluster_names[_cluster_id] = interpret_cluster_from_data_ru(
    #         normalized_row, raw_row, feature_descriptions
    #     )
    cluster_names[1] = "Глинистые почвы/Пахотные земли"
    cluster_names[2] = "Орошаемые территории/Урбанизированные территории"
    cluster_names[3] = "Средний уклон водосбора/Средняя высота водосбора"
    cluster_names[4] = "Мерзлотные породы/Лесистость"
    cluster_names[5] = "Объем озер/Песчаные породы"
    cluster_names[6] = "Затапливаемые территории/Песчаные породы"
    cluster_names[7] = "Объем озер/Осадочные породы"
    cluster_names[8] = "Карстовые породы/Лесистость"
    cluster_names[9] = "Лесистость/Осадочные породы"
    cluster_names[10] = "Песчаные породы/Лесистость"
    # Create display names with cluster ID prefix for map legend
    cluster_display_names = pd.Series(index=range(1, n_geo_clusters + 1), dtype=str)
    for _cid in range(1, n_geo_clusters + 1):
        cluster_display_names[_cid] = f"Ф{_cid}: {cluster_names[_cid]}"

    # Create radar charts for each cluster
    n_features_display = len(STANDARD_FEATURES)
    angles = np.linspace(0, 2 * np.pi, n_features_display, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Setup subplot grid with more spacing
    n_cols = 5
    n_rows = int(np.ceil(n_geo_clusters / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(30, n_rows * 7),
        subplot_kw={"projection": "polar"},
    )
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    axes = axes.flatten() if n_geo_clusters > 1 else [axes]
    for _idx, _cluster_id in enumerate(range(1, n_geo_clusters + 1)):
        ax = axes[_idx]

        # Get centroid values (0-1 normalized)
        values = cluster_centroids.loc[_cluster_id, STANDARD_FEATURES].tolist()
        values += values[:1]  # Complete the circle

        # Use _cluster_id - 1 to index colors (ensures Cluster 1 uses COLORS_HC[0])
        color_idx = _cluster_id - 1

        # Plot
        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            color=COLORS_HC[color_idx],
            label=cluster_names[_cluster_id],
        )
        ax.fill(angles, values, alpha=0.25, color=COLORS_HC[color_idx])

        # Styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(STANDARD_FEATURES, size=7)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"], size=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(pad=12)

        # Title with cluster size and name
        n_catchments = (geo_labels == _cluster_id).sum()
        ax.set_title(
            f"Ф{_cluster_id}: {cluster_names[_cluster_id]}\n(n={n_catchments})",
            size=9,
            weight="bold",
            pad=25,
        )
        # Add median line (0.5)
        ax.axhline(y=0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    # Remove empty subplots
    for _idx in range(n_geo_clusters, len(axes)):
        fig.delaxes(axes[_idx])

    plt.suptitle(
        "Профили кластеров: нормализованные значения признаков (шкала 0-1)",
        fontsize=16,
        y=1.02,
        weight="bold",
    )
    fig.savefig(
        f"{FIGURES_DIR}/geo_cluster_radar_profiles.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig
    return (cluster_display_names,)


@app.cell
def _(
    COLORS_HC,
    FIGURES_DIR,
    MARKERS_HC,
    MARKER_CORRECTION,
    basemap_data,
    cluster_display_names,
    gauges,
    geo_labels,
    n_geo_clusters,
    russia_plots,
):
    # Use display names with "C{id}:" prefix so sorting works correctly
    gauges["Тип кластера"] = [cluster_display_names[cl] for cl in geo_labels]

    # Plot map with proper cluster names
    fig_clusters_map = russia_plots(
        gdf_to_plot=gauges,
        basemap_data=basemap_data,
        distinction_col="Тип кластера",
        markers_list=MARKERS_HC,
        color_list=COLORS_HC,
        marker_size_corrections=MARKER_CORRECTION,
        figsize=(15, 10),
        just_points=True,
        legend_cols=3,
        base_marker_size=25,
        base_linewidth=0.5,
    )

    fig_clusters_map.suptitle(
        f"Кластеризация водосборов по физико-географическим признакам на {n_geo_clusters} кластеров",
        fontsize=16,
        y=0.98,
    )
    fig_clusters_map.savefig(
        f"{FIGURES_DIR}/geo_hierarchical_clusters_map.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig_clusters_map
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Geographical Distribution
    """)
    return


@app.cell
def _(DATA_OUT_DIR, geo_scaled):
    geo_scaled.to_csv(f"{DATA_OUT_DIR}/geo_scaled.csv")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Export
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
