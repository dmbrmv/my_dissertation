import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _():
    from collections import OrderedDict
    from pathlib import Path
    import re
    import sys
    import warnings

    import cartopy.crs as ccrs
    import geopandas as gpd
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    import numpy as np
    import pandas as pd
    from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

    # Create output directories
    from scipy.ndimage import median_filter
    from scipy.signal import savgol_filter
    import seaborn as sns
    from sklearn.decomposition import PCA

    # Compute silhouette scores
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_samples,
        silhouette_score,
    )
    import xarray as xr

    sys.path.append("../")

    from src.analytics.static_analysis import (
        get_cluster_colors,
        get_cluster_markers,
        get_marker_size_corrections,
        get_size_categories,
        interpret_cluster_from_data,
    )
    from src.analytics.clustering import (
        calculate_cluster_validation,
        compute_cluster_centroids,
        normalize_seasonal_patterns,
        perform_hierarchical_clustering,
        prepare_clustering_features,
        scale_to_unit_range,
        smooth_discharge_patterns,
    )
    from src.plots.maps import russia_plots
    from src.plots.cluster_plots import plot_hybrid_spatial_clusters
    from src.readers.geom_reader import load_geodata
    from src.utils.logger import setup_logger

    plt.rcParams["font.family"] = "DeJavu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    log = setup_logger("chapter_one", log_file="../logs/chapter_one.log")

    ws, gauges = load_geodata(folder_depth="../")
    basemap_data = gpd.read_file("../data/geometry/basemap_2023.gpkg")
    common_index = gauges.index.to_list()
    return (
        FormatStrFormatter,
        OrderedDict,
        Path,
        basemap_data,
        calinski_harabasz_score,
        common_index,
        davies_bouldin_score,
        dendrogram,
        fcluster,
        gauges,
        get_cluster_colors,
        get_cluster_markers,
        get_marker_size_corrections,
        get_size_categories,
        interpret_cluster_from_data,
        linkage,
        log,
        median_filter,
        np,
        pd,
        plot_hybrid_spatial_clusters,
        plt,
        re,
        russia_plots,
        savgol_filter,
        silhouette_samples,
        silhouette_score,
        ws,
        xr,
    )


@app.cell
def _(Path, log):
    def_dir = Path("../res/chapter_one")
    geo_img_dir = def_dir / "geo_analysis" / "images"
    geo_img_dir.mkdir(parents=True, exist_ok=True)
    geo_table_dir = def_dir / "geo_analysis" / "tables"
    geo_table_dir.mkdir(parents=True, exist_ok=True)

    hydro_img_dir = def_dir / "hydro_analysis" / "images"
    hydro_table_dir = def_dir / "hydro_analysis" / "tables"

    hydro_img_dir.mkdir(parents=True, exist_ok=True)
    hydro_table_dir.mkdir(parents=True, exist_ok=True)

    print("Created output directories:")
    log.info("Created output directories:")

    print(f"  Images: {hydro_img_dir}, {geo_img_dir}")
    print(f"  Tables: {hydro_table_dir}, {geo_table_dir}")
    return geo_img_dir, geo_table_dir, hydro_img_dir, hydro_table_dir


@app.cell
def _(geo_img_dir, get_size_categories, log, pd, plt, ws):
    # Compute statistics
    size_counts = ws["size"].value_counts(sort=False).sort_index()
    total_gauges = len(ws)

    # Create summary DataFrame
    size_summary = pd.DataFrame(
        {
            "Size Category": size_counts.index,
            "Count": size_counts.values,
            "Percentage": (size_counts.values / total_gauges * 100).round(2),
        }
    )

    # Calculate area statistics per category
    area_stats = []
    for cat in get_size_categories():
        cat_data = ws[ws["size"] == cat]["area_km2"]
        area_stats.append(
            {
                "Category": cat,
                "Min (km²)": cat_data.min(),
                "Mean (km²)": cat_data.mean(),
                "Median (km²)": cat_data.median(),
                "Max (km²)": cat_data.max(),
            }
        )
    area_stats_df = pd.DataFrame(area_stats)


    # Plot distribution
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(
        range(len(size_counts)),
        size_counts.values,
        color="steelblue",
        edgecolor="black",
        linewidth=1.2,
    )
    ax.set_xticks(range(len(size_counts)))
    ax.set_xticklabels(
        [cat.split(") ")[1] for cat in size_counts.index], rotation=0, ha="center"
    )
    ax.set_xlabel("Диапазон размеров", fontsize=14, fontweight="bold")
    ax.set_ylabel("Количество постов", fontsize=14, fontweight="bold")
    ax.set_title(
        "Распределение водосборов по размерам площадей",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels
    for rect, cnt in zip(bars, size_counts.values, strict=False):
        pct = cnt / total_gauges * 100
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            f"{cnt}: ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(f"{geo_img_dir}/watershed_size_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    log.info("Watershed size distribution analysis complete")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Geographical Clustering Maps
    """)
    return


@app.cell
def _(common_index, log, pd):
    # Load HydroATLAS data
    geo_data = pd.read_csv(
        "../data/attributes/hydro_atlas_cis_camels.csv",
        index_col="gauge_id",
        dtype={"gauge_id": str},
    )
    static_parameters = [
        "for_pc_use",
        "crp_pc_use",
        "inu_pc_ult",
        "ire_pc_use",
        "lka_pc_use",
        "prm_pc_use",
        "pst_pc_use",
        "cly_pc_uav",
        "slt_pc_uav",
        "snd_pc_uav",
        "kar_pc_use",
        "urb_pc_use",
        "gwt_cm_sav",
        "lkv_mc_usu",
        "rev_mc_usu",
        "slp_dg_uav",
        "sgr_dk_sav",
        "ws_area",
        "ele_mt_uav",
    ]
    geo_subset = geo_data.loc[common_index, static_parameters]

    # Scale to 0-1 range
    geo_scaled = (geo_subset - geo_subset.min()) / (geo_subset.max() - geo_subset.min())

    available_features = geo_scaled.loc[:, static_parameters].columns
    log.info(f"Selected {len(available_features)} HydroATLAS features")
    print(f"Features: {list(available_features)}")

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
        "gwt_cm_uav": {
            "name": "Groundwater Table Depth",
            "description": "Depth to groundwater table in centimeters, sub-basin area-weighted average",
            "suffix": "uav",
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
        "slp_dg_uav": {
            "name": "Terrain Slope",
            "description": "Mean terrain slope in degrees, area-weighted average across the upstream watershed",
            "suffix": "uav",
            "category": "Topography & Physiography",
            "hydrological_impact": "Steep slopes accelerate runoff generation and reduce infiltration time. Controls flow velocity and time of concentration. Increases erosion potential and sediment transport. Critical for understanding flashiness and flood response timing.",
            "units": "degrees",
            "source": "SRTM, HydroSHEDS",
        },
        "sgr_dk_uav": {
            "name": "Stream Gradient",
            "description": "Stream channel gradient in decimal form, sub-basin area-weighted average",
            "suffix": "uav",
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
    russian_names = {
        "for_pc_use": "Лесистость",
        "crp_pc_use": "Пахотные земли",
        "inu_pc_ult": "Затапливаемые территории",
        "ire_pc_use": "Орошаемые территории",
        "lka_pc_use": "Озерность",
        "prm_pc_use": "Мерзлотные территории",
        "pst_pc_use": "Пастбища",
        "cly_pc_uav": "Глинистость",
        "slt_pc_uav": "Осадочные породы",
        "snd_pc_uav": "Песчаные породы",
        "kar_pc_use": "Карстовые породы",
        "urb_pc_use": "Урбанизированные территории",
        "gwt_cm_uav": "Глубина первого водоносного горизонта",
        "slp_dg_uav": "Средний уклон водосбора",
        "sgr_dk_uav": "Средний уклон речной сети",
        "lkv_mc_usu": "Объём озёр",
        "rev_mc_usu": "Объём водохранилищ",
        "ws_area": "Площадь водосбора",
        "ele_mt_uav": "Средняя высота водосбора",
    }
    # Create DataFrame for display
    feature_df = pd.DataFrame.from_dict(feature_descriptions, orient="index")
    feature_df.index.name = "HydroATLAS Code"
    feature_df = feature_df.reset_index()

    # Reorder columns
    feature_df = feature_df[
        [
            "HydroATLAS Code",
            "name",
            "description",
            "suffix",
            "units",
            "category",
            "hydrological_impact",
            "source",
        ]
    ]

    # Rename columns for clarity
    feature_df.columns = [
        "HydroATLAS Code",
        "Short Name",
        "Full Description",
        "Suffix",
        "Units",
        "Category",
        "Hydrological Impact",
        "Data Source",
    ]

    feature_df
    return (
        available_features,
        feature_descriptions,
        geo_scaled,
        geo_subset,
        russian_names,
    )


@app.cell
def _(dendrogram, geo_img_dir, geo_scaled, linkage, plt):
    cluster_number = 9
    # Compute linkage matrix
    Z_geo = linkage(geo_scaled.values, method="ward", metric="euclidean")

    # Plot dendrogram (without x-axis labels due to many gauges)
    geo_cluster_fig, geo_cluster_ax = plt.subplots(figsize=(16, 8))
    dendro = dendrogram(
        Z_geo,
        ax=geo_cluster_ax,
        above_threshold_color="#808080",
        color_threshold=Z_geo[-cluster_number + 1, 2],
        no_labels=True,  # Don't show gauge IDs on x-axis
    )

    geo_cluster_ax.set_xlabel("Catchments (unlabeled due to high count)", fontsize=12)
    geo_cluster_ax.set_ylabel("Ward distance", fontsize=12)
    geo_cluster_ax.set_title("Hierarchical Clustering Dendrogram (Ward Method)", fontsize=14, pad=15)
    geo_cluster_ax.axhline(
        y=Z_geo[-cluster_number + 1, 2],
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Cut at {cluster_number} clusters",
    )
    geo_cluster_ax.legend(fontsize=11)
    geo_cluster_ax.grid(True, alpha=0.3)

    geo_cluster_fig.savefig(
        f"{geo_img_dir}/geo_dendrogram_ward_{cluster_number}_clusters.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.tight_layout()
    plt.show()
    return Z_geo, cluster_number


@app.cell
def _(
    Z_geo,
    available_features,
    calinski_harabasz_score,
    cluster_number,
    davies_bouldin_score,
    fcluster,
    gauges,
    geo_scaled,
    geo_subset,
    silhouette_samples,
    silhouette_score,
):
    hierarchical_labels = fcluster(Z_geo, t=cluster_number, criterion="maxclust")

    # Add to dataframes
    geo_scaled["cluster_geo"] = hierarchical_labels
    gauges["cluster_geo"] = hierarchical_labels

    # Compute cluster centroids (0-1 normalized values)
    cluster_centroids = geo_scaled.groupby("cluster_geo")[available_features].mean()

    # Compute raw (original scale) centroids for interpretation
    geo_subset["cluster_geo"] = hierarchical_labels
    cluster_centroids_raw = geo_subset.groupby("cluster_geo")[available_features].mean()


    # CRITICAL: Use only feature columns, exclude cluster_geo
    scaled_values = geo_scaled[available_features].values
    silhouette_avg = silhouette_score(scaled_values, hierarchical_labels)
    silhouette_vals = silhouette_samples(scaled_values, hierarchical_labels)

    ch_score = calinski_harabasz_score(scaled_values, hierarchical_labels)
    db_score = davies_bouldin_score(scaled_values, hierarchical_labels)

    print(f"Clustering validation metrics (k={cluster_number}):")
    print(f"  Silhouette Score: {silhouette_avg:.3f}")
    print(f"  Calinski-Harabasz: {ch_score:.1f}")
    print(f"  Davies-Bouldin: {db_score:.3f}")

    # Per-cluster silhouette
    for cluster_id in range(1, cluster_number + 1):
        cluster_mask = hierarchical_labels == cluster_id
        cluster_sil = silhouette_vals[cluster_mask].mean()
        n_samples = cluster_mask.sum()
        print(f"  Cluster {cluster_id}: silhouette={cluster_sil:.3f}, n={n_samples}")
    return cluster_centroids, cluster_centroids_raw, hierarchical_labels


@app.cell
def _(
    available_features,
    cluster_centroids,
    cluster_centroids_raw,
    cluster_number,
    feature_descriptions,
    geo_img_dir,
    get_cluster_colors,
    get_cluster_markers,
    get_marker_size_corrections,
    hierarchical_labels,
    interpret_cluster_from_data,
    np,
    pd,
    plt,
    russian_names,
):
    # Get markers and colors for clusters
    markers_hc = get_cluster_markers(cluster_number)
    colors_hc = get_cluster_colors(cluster_number)
    marker_corrections = get_marker_size_corrections()


    def classify_geographical_cluster(
        normalized_row: pd.Series,
        raw_row: pd.Series,
        feature_descriptions: dict,
    ) -> tuple[str, str, str]:
        """Classify geographical cluster based on dominant landscape characteristics.

        Returns tuple of (short_name, cluster_type_ru, detailed_description).
        """
        # Key feature thresholds (normalized 0-1 scale)
        norm = normalized_row

        # Identify dominant landscape features (top 3 by normalized value)
        top_features = norm.nlargest(3)
        dominant_feature = top_features.index[0]
        dominant_value = top_features.iloc[0]

        # Extract raw values for interpretation
        raw = raw_row

        # Classification logic based on dominant features and combinations
        if norm["prm_pc_use"] > 0.5:  # Permafrost-dominated
            short_name = "Permafrost"
            cluster_type = "Криолитозона"
            description = (
                f"Криолитозонный ландшафт с {raw['prm_pc_use']:.1f}% площади под многолетней мерзлотой. "
                f"Экстремально ограниченная инфильтрация приводит к преобладанию поверхностного стока. "
                f"Характерна высокая сезонная контрастность водного режима с весенним половодьем и зимней меженью. "
                f"Средняя высота {raw['ele_mt_uav']:.0f} м, уклон {raw['slp_dg_uav']:.1f}°. "
                "Уязвимы к деградации мерзлоты при потеплении климата."
            )

        elif norm["for_pc_use"] > 0.6:  # Forest-dominated
            short_name = "Forest"
            cluster_type = "Лесной"
            description = (
                f"Лесной ландшафт с {raw['for_pc_use']:.1f}% лесистости. Высокая транспирация и "
                "перехват осадков кронами снижают поверхностный сток. Развитая корневая система "
                f"способствует инфильтрации. Средняя высота {raw['ele_mt_uav']:.0f} м обеспечивает "
                "достаточное увлажнение. Стабильный гидрологический режим с выраженным базисным стоком "
                "и умеренными паводками."
            )

        elif norm["lka_pc_use"] > 0.4 or norm["lkv_mc_usu"] > 0.5:  # Lake-dominated
            short_name = "Lacustrine"
            cluster_type = "Озёрный"
            description = (
                f"Озёрно-болотный ландшафт с {raw['lka_pc_use']:.1f}% озёрности и общим объёмом "
                f"озёр {raw['lkv_mc_usu']:.0f} млн м³. Естественное регулирование стока озёрами "
                "сглаживает внутригодовую изменчивость и демпфирует паводковые пики. Повышенное "
                "испарение с водной поверхности. Характерна низкая интенсивность водообмена и "
                "задержанный гидрологический отклик на осадки."
            )

        elif norm["urb_pc_use"] > 0.3 or norm["crp_pc_use"] > 0.5:  # Anthropogenic
            short_name = "Anthropogenic"
            cluster_type = "Антропогенный"
            description = (
                f"Антропогенно-трансформированный ландшафт: урбанизация {raw['urb_pc_use']:.1f}%, "
                f"пашни {raw['crp_pc_use']:.1f}%, орошение {raw['ire_pc_use']:.1f}%. Импервиозные "
                "поверхности и уплотнённые почвы генерируют быстрый поверхностный сток с высокими "
                "паводковыми пиками. Снижена инфильтрация и базисный сток. Водный режим сильно "
                "изменён водозабором, сбросами и регулированием."
            )

        elif norm["rev_mc_usu"] > 0.4:  # Regulated
            short_name = "Regulated"
            cluster_type = "Зарегулированный"
            description = (
                f"Зарегулированный ландшафт с объёмом водохранилищ {raw['rev_mc_usu']:.0f} млн м³. "
                "Естественный гидрологический режим трансформирован работой гидроузлов. Сглаживание "
                "сезонных колебаний, аккумуляция весеннего стока и его перераспределение в межень. "
                "Изменение температурного и ледового режима. Критически важен режим попусков для "
                "экологии нижнего бьефа."
            )

        elif norm["kar_pc_use"] > 0.3:  # Karst
            short_name = "Karst"
            cluster_type = "Карстовый"
            description = (
                f"Карстовый ландшафт с {raw['kar_pc_use']:.1f}% площади карстующихся пород. "
                "Доминирует подземный сток через систему трещин и пустот. Характерны потери стока "
                "в верховьях и выход мощных карстовых источников. Высокая доля базисного стока "
                f"обеспечивает стабильность водности. Глубина грунтовых вод {raw['gwt_cm_sav']:.0f} см. "
                "Сложная гидрогеологическая структура."
            )

        elif norm["slp_dg_uav"] > 0.6 or norm["ele_mt_uav"] > 0.7:  # Mountainous
            short_name = "Mountain"
            cluster_type = "Горный"
            description = (
                f"Горный ландшафт: средняя высота {raw['ele_mt_uav']:.0f} м, уклон {raw['slp_dg_uav']:.1f}°, "
                f"градиент русла {raw['sgr_dk_uav']:.4f}. Ускоренный поверхностный сток с коротким "
                "временем добегания. Высокая эрозионная активность и транспорт наносов. Вертикальная "
                "климатическая зональность обуславливает разнообразие источников питания. Характерны "
                "паводки от интенсивных осадков и таяния высокогорных снегов."
            )

        elif norm["snd_pc_uav"] > 0.5:  # Sandy soils
            short_name = "Sandy"
            cluster_type = "Песчаный"
            description = (
                f"Ландшафт с преобладанием песчаных почв ({raw['snd_pc_uav']:.1f}% песка). Высокая "
                "инфильтрационная способность способствует глубокому питанию грунтовых вод и снижает "
                "поверхностный сток. Характерен базисный тип питания рек с устойчивым стоком в межень. "
                f"Низкая влагоёмкость почв при глубине УГВ {raw['gwt_cm_sav']:.0f} см. Минимальная "
                "эрозия, но уязвимость к засухам."
            )

        elif norm["cly_pc_uav"] > 0.5:  # Clay soils
            short_name = "Clay"
            cluster_type = "Глинистый"
            description = (
                f"Ландшафт с тяжёлыми глинистыми почвами ({raw['cly_pc_uav']:.1f}% глины). Низкая "
                "водопроницаемость ограничивает инфильтрацию и генерирует интенсивный поверхностный сток. "
                "Высокая влагоёмкость но медленная отдача воды. Характерны паводки при насыщении почв и "
                "переувлажнение в понижениях. Повышенная мутность рек за счёт почвенной эрозии."
            )

        elif norm["inu_pc_ult"] > 0.3:  # Floodplain
            short_name = "Floodplain"
            cluster_type = "Пойменный"
            description = (
                f"Пойменный ландшафт с {raw['inu_pc_ult']:.1f}% затапливаемых территорий. Естественное "
                "регулирование паводков через разлив и временную аккумуляцию воды в пойме. Высокое "
                "биоразнообразие и продуктивность пойменных экосистем. Сложный водообмен между руслом, "
                "старицами и грунтовыми водами. Критическая роль в снижении пиков половодья и поддержании "
                "стока в межень."
            )

        else:  # Mixed/transitional
            # Identify top 2 contributors
            top1_name = russian_names.get(top_features.index[0], top_features.index[0])
            top2_name = russian_names.get(top_features.index[1], top_features.index[1])

            short_name = "Mixed"
            cluster_type = "Смешанный"
            description = (
                f"Переходный ландшафт со смешанными характеристиками. Доминируют: {top1_name} "
                f"({raw[top_features.index[0]]:.1f} {feature_descriptions[top_features.index[0]]['units']}) "
                f"и {top2_name} ({raw[top_features.index[1]]:.1f} "
                f"{feature_descriptions[top_features.index[1]]['units']}). "
                f"Средняя высота {raw['ele_mt_uav']:.0f} м, площадь водосбора {raw['ws_area']:.0f} км². "
                "Комбинация различных источников стока определяет умеренную сезонную изменчивость. "
                "Гидрологический режим формируется взаимодействием нескольких ландшафтных факторов."
            )

        return short_name, cluster_type, description


    # Generate cluster names AND detailed interpretations
    cluster_names_geo = pd.Series(index=range(1, cluster_number + 1), dtype=str)
    cluster_short_names_geo = pd.Series(index=range(1, cluster_number + 1), dtype=str)
    cluster_types_geo = pd.Series(index=range(1, cluster_number + 1), dtype=str)
    cluster_descriptions_geo = pd.Series(index=range(1, cluster_number + 1), dtype=str)

    for _cid in range(1, cluster_number + 1):
        _normalized_row = cluster_centroids.loc[_cid, :]
        _raw_row = cluster_centroids_raw.loc[_cid, :]

        # Get both old name and new classification
        cluster_names_geo[_cid] = interpret_cluster_from_data(
            _normalized_row, _raw_row, feature_descriptions
        )

        _short_name, _cluster_type, _description = classify_geographical_cluster(
            _normalized_row, _raw_row, feature_descriptions
        )

        cluster_short_names_geo[_cid] = _short_name
        cluster_types_geo[_cid] = _cluster_type
        cluster_descriptions_geo[_cid] = _description

    cluster_names_geo.index.name = "Cluster"

    print("Generated cluster classifications:")
    for _cid in range(1, cluster_number + 1):
        _n_catch = (hierarchical_labels == _cid).sum()
        print(f"\nCluster {_cid} (n={_n_catch}):")
        print(f"  Short Name: {cluster_short_names_geo[_cid]}")
        print(f"  Type: {cluster_types_geo[_cid]}")
        print(f"  Legacy Name: {cluster_names_geo[_cid]}")

    # Create display names with cluster ID prefix for map legend
    cluster_display_names = pd.Series(index=range(1, cluster_number + 1), dtype=str)
    for _cid in range(1, cluster_number + 1):
        cluster_display_names[_cid] = f"C{_cid}: {cluster_names_geo[_cid]}"

    # Create radar charts for each cluster
    _n_features_display = len(available_features)
    _angles_radar = np.linspace(0, 2 * np.pi, _n_features_display, endpoint=False).tolist()
    _angles_radar += _angles_radar[:1]  # Complete the circle

    # Setup subplot grid
    _n_cols = 3
    _n_rows = int(np.ceil(cluster_number / _n_cols))
    fig_radar, _axes_radar = plt.subplots(
        _n_rows,
        _n_cols,
        figsize=(25, _n_rows * 5),
        subplot_kw={"projection": "polar"},
    )
    _axes_radar = _axes_radar.flatten() if cluster_number > 1 else [_axes_radar]

    for _idx, _cid in enumerate(range(1, cluster_number + 1)):
        _ax = _axes_radar[_idx]

        # Get centroid values (0-1 normalized)
        _values_radar = cluster_centroids.loc[_cid, available_features].tolist()
        _values_radar += _values_radar[:1]  # Complete the circle

        # Use _cid - 1 to index colors (ensures Cluster 1 uses colors_hc[0])
        _color_idx = _cid - 1

        # Plot
        _ax.plot(
            _angles_radar,
            _values_radar,
            "o-",
            linewidth=2,
            color=colors_hc[_color_idx],
            label=cluster_names_geo[_cid],
        )
        _ax.fill(_angles_radar, _values_radar, alpha=0.25, color=colors_hc[_color_idx])

        # Styling
        _ax.set_xticks(_angles_radar[:-1])
        _ax.set_xticklabels(available_features, size=8)
        _ax.set_ylim(0, 1)
        _ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        _ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"], size=8)
        _ax.grid(True, alpha=0.3)

        # Title with cluster size and name
        _n_catchments = (hierarchical_labels == _cid).sum()
        _ax.set_title(
            f"Cluster {_cid}: {cluster_names_geo[_cid]}\n(n={_n_catchments})",
            size=10,
            weight="bold",
            pad=20,
        )

        # Add median line (0.5)
        _ax.axhline(y=0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    # Remove empty subplots
    for _idx in range(cluster_number, len(_axes_radar)):
        fig_radar.delaxes(_axes_radar[_idx])

    plt.suptitle(
        "Cluster Profiles: Normalized Feature Values (0-1 scale)",
        fontsize=16,
        y=1.00,
        weight="bold",
    )
    plt.tight_layout()
    fig_radar.savefig(
        f"{geo_img_dir}/geo_cluster_radar_{cluster_number}_profiles.png",
        dpi=300,
        bbox_inches="tight",
    )

    print("\nRadar chart complete.")

    fig_radar
    return (
        cluster_descriptions_geo,
        cluster_display_names,
        cluster_names_geo,
        cluster_short_names_geo,
        cluster_types_geo,
        colors_hc,
        marker_corrections,
        markers_hc,
    )


@app.cell
def _(
    OrderedDict,
    cluster_centroids,
    cluster_centroids_raw,
    cluster_descriptions_geo,
    cluster_names_geo,
    cluster_number,
    cluster_short_names_geo,
    cluster_types_geo,
    geo_table_dir,
    hierarchical_labels,
    log,
    pd,
):
    # Create comprehensive geographical cluster analysis DataFrame
    _geo_cluster_analysis = OrderedDict()

    for _cid in range(1, cluster_number + 1):
        _n_gauges_cluster = (hierarchical_labels == _cid).sum()
        _norm_row = cluster_centroids.loc[_cid, :]
        _raw_row = cluster_centroids_raw.loc[_cid, :]

        _geo_cluster_analysis[f"Cluster {_cid}"] = {
            "n_gauges": _n_gauges_cluster,
            "short_name": cluster_short_names_geo[_cid],
            "cluster_type": cluster_types_geo[_cid],
            "description": cluster_descriptions_geo[_cid],
            "legacy_name": cluster_names_geo[_cid],
            # Key raw attributes
            "forest_pct": _raw_row["for_pc_use"],
            "cropland_pct": _raw_row["crp_pc_use"],
            "permafrost_pct": _raw_row["prm_pc_use"],
            "lake_pct": _raw_row["lka_pc_use"],
            "urban_pct": _raw_row["urb_pc_use"],
            "mean_elevation_m": _raw_row["ele_mt_uav"],
            "mean_slope_deg": _raw_row["slp_dg_uav"],
            "watershed_area_km2": _raw_row["ws_area"],
            "clay_pct": _raw_row["cly_pc_uav"],
            "sand_pct": _raw_row["snd_pc_uav"],
            "karst_pct": _raw_row["kar_pc_use"],
            "reservoir_vol_mcm": _raw_row["rev_mc_usu"],
            "lake_vol_mcm": _raw_row["lkv_mc_usu"],
        }

    # Convert to DataFrame
    geo_analysis_df = pd.DataFrame.from_dict(_geo_cluster_analysis, orient="index")
    geo_analysis_df.index.name = "cluster_name"
    geo_analysis_df.reset_index(inplace=True)

    # Display detailed results
    log.info(f"Geographical cluster classification complete for {cluster_number} clusters")
    print("\n" + "=" * 120)
    print("GEOGRAPHICAL CLUSTER CLASSIFICATION RESULTS")
    print("=" * 120)

    for _, _row_geo in geo_analysis_df.iterrows():
        print(f"\n{_row_geo['cluster_name']} (n={_row_geo['n_gauges']} catchments)")
        print("─" * 120)
        print(f"  Short Name:        {_row_geo['short_name']}")
        print(f"  Landscape Type:    {_row_geo['cluster_type']}")
        print(f"\n  Description:")
        # Word wrap description at ~100 chars
        _desc_words = _row_geo['description'].split()
        _lines_desc = []
        _current_line = "    "
        for _word in _desc_words:
            if len(_current_line) + len(_word) + 1 <= 104:
                _current_line += _word + " "
            else:
                _lines_desc.append(_current_line.rstrip())
                _current_line = "    " + _word + " "
        _lines_desc.append(_current_line.rstrip())
        print("\n".join(_lines_desc))

        print(f"\n  Key Attributes:")
        print(f"    Elevation: {_row_geo['mean_elevation_m']:.0f} m  |  Slope: {_row_geo['mean_slope_deg']:.2f}°  |  Area: {_row_geo['watershed_area_km2']:.0f} km²")
        print(f"    Forest: {_row_geo['forest_pct']:.1f}%  |  Cropland: {_row_geo['cropland_pct']:.1f}%  |  Urban: {_row_geo['urban_pct']:.1f}%")
        print(f"    Permafrost: {_row_geo['permafrost_pct']:.1f}%  |  Lakes: {_row_geo['lake_pct']:.1f}%  |  Karst: {_row_geo['karst_pct']:.1f}%")

    # Save detailed results
    _geo_analysis_output = geo_table_dir / f"geo_cluster_analysis_{cluster_number}_detailed.csv"
    geo_analysis_df.to_csv(_geo_analysis_output, index=False, encoding="utf-8-sig")
    log.info(f"Saved geographical cluster analysis to {_geo_analysis_output}")

    print(f"\n{'=' * 120}")
    print(f"Results saved to: {_geo_analysis_output}")

    geo_analysis_df
    return


@app.cell
def _(
    basemap_data,
    cluster_display_names,
    cluster_number,
    colors_hc,
    gauges,
    geo_img_dir,
    hierarchical_labels,
    marker_corrections,
    markers_hc,
    russia_plots,
):
    # Prepare cluster names for geographic visualization
    gauge_clustered = gauges.copy()
    # Use display names with "C{id}:" prefix so sorting works correctly
    gauge_clustered["Cluster_Name"] = [
        cluster_display_names[cl] for cl in hierarchical_labels
    ]

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
        legend_cols=3,
        base_marker_size=25,
        base_linewidth=0.5,
    )

    fig_clusters_map.suptitle(
        f"Hydrological Catchment Clusters - {cluster_number} Clusters (Ward Method)",
        fontsize=16,
        y=0.98,
    )
    fig_clusters_map.savefig(
        f"{geo_img_dir}/geo_hierarchical_{cluster_number}clusters_map.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig_clusters_map
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Hydrological Clustering
    """)
    return


@app.cell
def _(gauges, log, median_filter, np, pd, savgol_filter, ws, xr):
    # Load geographical clustering results
    geo_clusters = pd.read_csv(
        "../res/chapter_one/geo_analysis/tables/geo_gauge_cluster_analysis_9_detailed.csv",
        index_col="gauge_id",
        dtype={"gauge_id": str},
    )

    # Extract cluster assignment
    geo_cluster_info = geo_clusters[["cluster_id", "cluster_name"]].copy()
    geo_cluster_info.rename(
        columns={"cluster_id": "geo_cluster_id", "cluster_name": "geo_cluster_name"},
        inplace=True,
    )

    print(f"Loaded geographical clustering for {len(geo_cluster_info)} gauges")
    print(f"Number of geographical clusters: {geo_cluster_info['geo_cluster_id'].nunique()}")
    print("\nGeographical cluster distribution:")
    print(geo_cluster_info["geo_cluster_name"].value_counts())
    # Load discharge time series for all gauges
    discharge_data = {}
    dis_column = "q_mm_day"

    for gauge_id in ws.index:
        try:
            with xr.open_dataset(f"../data/nc_all_q/{gauge_id}.nc") as ds:
                df = ds.to_dataframe()
            discharge_data[gauge_id] = df[[dis_column]].squeeze()

        except Exception as e:
            log.error(f"Error loading discharge for {gauge_id}: {e}")
            continue

    # Calculate median seasonal cycle for each gauge
    q_df = {}

    for gauge_id in discharge_data.keys():
        try:
            ts = discharge_data[gauge_id]
            # Group by day of year and take median across all years
            seasonal_cycle = ts.groupby([ts.index.month, ts.index.day]).median().values
            q_df[gauge_id] = seasonal_cycle

        except Exception as e:
            log.error(f"Error calculating seasonal cycle for {gauge_id}: {e}")
            continue

    # Convert to DataFrame (rows=days, columns=gauges)
    q_df = pd.DataFrame.from_dict(q_df, orient="columns")

    # Filter out gauges with extreme outliers (max > 50 mm/day)
    q_df = q_df.loc[:, q_df.max() < 50]

    # ===== SPIKE DETECTION AND SMOOTHING =====
    # Detect and interpolate anomalous spikes (likely data artifacts)


    q_df_smoothed = q_df.copy()

    for col in q_df_smoothed.columns:
        series = q_df_smoothed[col].values

        # Apply median filter to detect outliers (window=5 days)
        smoothed = median_filter(series, size=5, mode="wrap")

        # Calculate residuals
        residuals = np.abs(series - smoothed)
        threshold = residuals.std() * 3  # 3-sigma threshold

        # Identify spikes
        spike_mask = residuals > threshold

        if spike_mask.any():
            # Interpolate spikes using linear interpolation
            valid_indices = np.where(~spike_mask)[0]
            spike_indices = np.where(spike_mask)[0]

            if len(valid_indices) > 1:
                # Linear interpolation of spike values
                interpolated_values = np.interp(
                    spike_indices, valid_indices, series[valid_indices]
                )
                series[spike_indices] = interpolated_values

        # Savitzky-Golay: window=11 days, polynomial order=3
        # This preserves peaks but removes high-frequency noise
        series_smoothed = savgol_filter(series, window_length=11, polyorder=3, mode="wrap")

        q_df_smoothed[col] = series_smoothed

    # Replace original with smoothed
    q_df = q_df_smoothed.copy()

    print(f"Applied spike removal and smoothing to {len(q_df.columns)} gauges")
    print("Smoothing: median filter (spike detection) + Savitzky-Golay (curve flattening)")

    # Normalize each gauge's pattern to [0, 1]
    q_df_normalized = q_df.copy()
    q_df_normalized = (q_df_normalized - q_df_normalized.min()) / (
        q_df_normalized.max() - q_df_normalized.min()
    )

    # Transpose for clustering (rows=gauges, columns=days)
    q_df_clust = q_df_normalized.copy().T

    # Add spatial coordinates as features (optional: weight spatial proximity)
    for gauge_id in q_df_clust.index:
        q_df_clust.loc[gauge_id, "lat"] = gauges.loc[gauge_id, "geometry"].y
        q_df_clust.loc[gauge_id, "lon"] = gauges.loc[gauge_id, "geometry"].x

    # Remove any gauges with NaN values
    q_df_clust = q_df_clust.dropna()
    hydro_index = q_df_clust.index

    discharge_data = {k: v for k, v in discharge_data.items() if k in hydro_index}
    print(f"\nPrepared seasonal patterns for {len(q_df_clust)} gauges")
    print(f"Features per gauge: {q_df_clust.shape[1]} (366 days + 2 coords)")
    print(
        f"Normalized discharge range: [{q_df_normalized.min().min():.2f}, {q_df_normalized.max().max():.2f}]"
    )
    return q_df_clust, q_df_normalized


@app.cell
def _(dendrogram, hydro_img_dir, linkage, log, plt, q_df_clust):
    # Perform hierarchical clustering using Ward's method
    n_hydro_clusters = 10

    Z_hydro = linkage(q_df_clust.values, method="ward", metric="euclidean")

    # Create dendrogram
    _fig, _ax = plt.subplots(figsize=(16, 8))
    _dendro = dendrogram(
        Z_hydro,
        ax=_ax,
        above_threshold_color="#808080",
        color_threshold=Z_hydro[-n_hydro_clusters + 1, 2],
        no_labels=True,
        truncate_mode="level",
        p=0,
    )

    _ax.set_xlabel("Catchments (unlabeled due to high count)", fontsize=12)
    _ax.set_ylabel("Ward distance", fontsize=12)
    _ax.set_title(
        f"Hydrological Clustering Dendrogram - {n_hydro_clusters} Clusters (Ward Method)",
        fontsize=14,
        pad=15,
    )
    _ax.axhline(
        y=Z_hydro[-n_hydro_clusters + 1, 2],
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Cut at {n_hydro_clusters} clusters",
    )
    _ax.legend(fontsize=11)
    _ax.grid(True, alpha=0.3)


    _fig.savefig(
        hydro_img_dir / f"dendrogram_ward_{n_hydro_clusters}_hydro_clusters.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.tight_layout()
    plt.show()

    log.info(f"Hierarchical clustering complete: {n_hydro_clusters} clusters")
    return Z_hydro, n_hydro_clusters


@app.cell
def _(
    Z_hydro,
    basemap_data,
    fcluster,
    gauges,
    get_cluster_colors,
    get_cluster_markers,
    get_marker_size_corrections,
    hydro_img_dir,
    log,
    n_hydro_clusters,
    plt,
    q_df_clust,
    russia_plots,
):
    # Extract cluster labels using fcluster
    hydro_labels = fcluster(Z_hydro, t=n_hydro_clusters, criterion="maxclust")

    # Create GeoDataFrame with cluster assignments
    gauge_hydro = gauges.loc[q_df_clust.index, :].copy()
    gauge_hydro["hydro_cluster_id"] = hydro_labels
    gauge_hydro["Hydro cluster"] = [f"Cluster {cl}" for cl in hydro_labels]

    # Get colors and markers for visualization
    _markers_hc = get_cluster_markers(n_hydro_clusters)
    _colors_hc = get_cluster_colors(n_hydro_clusters)
    _marker_corrections = get_marker_size_corrections()

    # Plot spatial distribution of hydrological clusters
    fig_hydro_map = russia_plots(
        gdf_to_plot=gauge_hydro,
        basemap_data=basemap_data,
        distinction_col="Hydro cluster",
        markers_list=_markers_hc,
        color_list=_colors_hc,
        marker_size_corrections=_marker_corrections,
        title_text=f"Hydrological Clusters Based on Discharge Patterns ({n_hydro_clusters} clusters)",
        figsize=(16, 9),
        just_points=True,
        legend_cols=3,
        base_marker_size=28,
        base_linewidth=0.5,
    )

    fig_hydro_map.savefig(
        hydro_img_dir / f"hydro_clusters_{n_hydro_clusters}_map.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    log.info("Saved hydrological cluster spatial map")
    return gauge_hydro, hydro_labels


@app.cell
def _(
    FormatStrFormatter,
    OrderedDict,
    gauge_hydro,
    hydro_img_dir,
    log,
    n_hydro_clusters,
    np,
    pd,
    plt,
    q_df_normalized,
    re,
):
    # Prepare cluster-wise seasonal patterns
    cluster_groups = list(gauge_hydro.groupby("Hydro cluster").groups.items())


    def _cluster_num(key):
        """Extract numeric cluster ID from cluster name."""
        m = re.search(r"\d+", key)
        return int(m.group()) if m else float("inf")


    cluster_groups_sorted = sorted(cluster_groups, key=lambda x: _cluster_num(x[0]))

    # Build OrderedDict of cluster patterns
    cluster_q_mm = OrderedDict()
    for clust_name, member_ids in cluster_groups_sorted:
        clust_df = q_df_normalized.loc[:, member_ids].copy()
        clust_df.index = pd.date_range(start="2000-01-01", periods=366)
        cluster_q_mm[clust_name] = clust_df

    # Create grid layout for 10 clusters (2 rows x 5 cols)
    nrows = 2
    ncols = 5

    q_fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10))
    axes_flat = axes.flatten()

    for i, (clust_name, clust_data) in enumerate(cluster_q_mm.items()):
        if i >= len(axes_flat):
            break
        clust_ax = axes_flat[i]

        # Calculate daily statistics across all gauges in cluster
        mean_by_gauge = (
            clust_data.groupby([clust_data.index.month, clust_data.index.day])
            .median()
            .reset_index(drop=True)
        )

        # Calculate median, 25th and 75th percentiles
        clust_median = mean_by_gauge.median(axis=1).values.ravel()
        clust_p25 = mean_by_gauge.quantile(0.25, axis=1).values.ravel()
        clust_p75 = mean_by_gauge.quantile(0.75, axis=1).values.ravel()

        # LAYER 1 (BACKGROUND): Plot individual gauges (very transparent, gray-blue)
        for _gauge_id in mean_by_gauge.columns:
            clust_ax.plot(
                mean_by_gauge.index.values,
                mean_by_gauge[_gauge_id].values,
                color="blue",
                alpha=0.12,
                linewidth=0.5,
                zorder=1,
            )

        # LAYER 2 (MIDDLE): Plot 25-75 percentile spread (light red/salmon fill)
        clust_ax.fill_between(
            mean_by_gauge.index.values,
            clust_p25,
            clust_p75,
            color="salmon",
            alpha=0.35,
            label="25-75% spread",
            zorder=2,
        )

        # LAYER 3 (FOREGROUND): Plot cluster median (bold red line on top)
        clust_ax.plot(
            mean_by_gauge.index.values,
            clust_median,
            color="darkred",
            alpha=1.0,
            linewidth=2.8,
            label="Cluster median",
            zorder=3,
        )

        # Formatting
        clust_ax.grid(axis="both", which="both", alpha=0.3, zorder=0)
        clust_ax.set_xlim(0, 365)
        clust_ax.set_ylim(0, 1)
        clust_ax.set_yticks(np.arange(0, 1.25, 0.25))
        clust_ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        if i % ncols == 0:
            clust_ax.set_ylabel("Normalized runoff [0-1]")
        if i >= (nrows - 1) * ncols:
            clust_ax.set_xlabel("Day of year")

        for val in np.arange(0, 1.25, 0.25):
            clust_ax.axhline(val, c="black", linestyle="--", linewidth=0.2, zorder=0)

        clust_ax.set_title(f"{clust_name} — {len(clust_data.columns)} gauges")

        # Add legend only to first subplot
        if i == 0:
            clust_ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # Turn off any unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    q_fig.suptitle(
        "Hydrological Regime Types: Normalized Seasonal Discharge Patterns",
        fontsize=16,
        y=0.995,
    )
    q_fig.tight_layout()
    q_fig.savefig(
        hydro_img_dir / f"hydrograph_clusters_{n_hydro_clusters}.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    log.info("Hydrological regime visualization complete")
    return (cluster_q_mm,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Cluster interpretation
    """)
    return


@app.cell
def _(
    OrderedDict,
    cluster_q_mm,
    hydro_table_dir,
    log,
    n_hydro_clusters,
    np,
    pd,
):
    # Verify cluster_q_mm exists from previous cells
    if "cluster_q_mm" not in locals():
        raise RuntimeError("cluster_q_mm not found. Run NewChapterOne.ipynb cells first.")

    # Initialize storage for regime characteristics
    hydro_regime_info = OrderedDict()

    # Define seasonal DOY ranges (1-based indexing)
    # Winter: Dec 1 - Feb 28/29 (DOY 335-365, 1-59)
    # Spring: Mar 1 - May 31 (DOY 60-151)
    # Summer: Jun 1 - Aug 31 (DOY 152-243)
    # Autumn: Sep 1 - Nov 30 (DOY 244-334)


    def classify_hydrological_regime(
        cluster_median: np.ndarray,
        winter_ratio: float,
        spring_ratio: float,
        summer_ratio: float,
        autumn_ratio: float,
        peak_doy: int,
        cv: float,
    ) -> tuple[str, str, str]:
        """Classify hydrological regime using multi-indicator approach.

        Returns tuple of (short_name, regime_type_ru, detailed_description).
        """
        # Sort seasonal ratios to identify dominant and secondary contributors
        seasonal_dict = {
            "winter": winter_ratio,
            "spring": spring_ratio,
            "summer": summer_ratio,
            "autumn": autumn_ratio,
        }
        sorted_seasons = sorted(seasonal_dict.items(), key=lambda x: x[1], reverse=True)
        dominant_season = sorted_seasons[0][0]
        dominant_ratio = sorted_seasons[0][1]
        secondary_season = sorted_seasons[1][0]
        secondary_ratio = sorted_seasons[1][1]

        # Calculate regime complexity (Gini coefficient for seasonality)
        ratios = np.array([winter_ratio, spring_ratio, summer_ratio, autumn_ratio])
        regime_complexity = 1 - np.sum(ratios**2)  # Higher = more uniform distribution

        # Determine primary regime driver
        if dominant_ratio > 0.45:  # Strong single-season dominance
            regime_class = "simple"
        elif dominant_ratio > 0.35 and secondary_ratio > 0.25:
            regime_class = "mixed"
        else:
            regime_class = "complex"

        # Classify based on dominant season and peak timing
        season_names_en = {
            "winter": "Winter",
            "spring": "Spring",
            "summer": "Summer",
            "autumn": "Autumn",
        }
        season_names_ru = {
            "winter": "Зимний",
            "spring": "Весенний",
            "summer": "Летний",
            "autumn": "Осенний",
        }

        # Determine regime type with physical interpretation
        if dominant_season == "spring" and 60 <= peak_doy <= 151:
            # Nival regime: snowmelt-dominated
            if dominant_ratio > 0.45:
                short_name = "Nival"
                regime_type = "Снеговой"
                description = (
                    "Классический снеговой режим с ярко выраженным весенним половодьем. "
                    "Основной сток (>45%) формируется в период снеготаяния (март-май). "
                    "Характерен для водосборов с устойчивым снежным покровом и низкой "
                    "долей осадков в теплый период. Минимальный сток наблюдается зимой "
                    "при промерзании почв."
                )
            else:
                short_name = "Nivo-Pluvial"
                regime_type = "Снегово-дождевой"
                description = (
                    "Смешанный снегово-дождевой режим с весенним пиком стока от "
                    "снеготаяния и значительным вкладом дождевых паводков летом-осенью. "
                    f"Весенний сток составляет {spring_ratio:.1%}, осенний {autumn_ratio:.1%}. "
                    "Характерен для переходных климатических зон с умеренным снегонакоплением "
                    "и летними ливневыми осадками."
                )

        elif dominant_season == "summer" and 152 <= peak_doy <= 243:
            # Glacial/high-elevation snowmelt or monsoon
            if cv > 0.6:
                short_name = "Glacial"
                regime_type = "Ледниковый"
                description = (
                    "Ледниковый режим с максимумом стока в летние месяцы (июнь-август) "
                    "от таяния высокогорных снегов и ледников. Характеризуется высокой "
                    f"межгодовой изменчивостью (CV={cv:.2f}) и задержкой пика относительно "
                    "весеннего снеготаяния. Типичен для высокогорных водосборов (>2000 м) "
                    "с развитым оледенением."
                )
            else:
                short_name = "Summer Rain"
                regime_type = "Летне-дождевой"
                description = (
                    "Летне-дождевой режим с максимумом стока в период интенсивных летних "
                    f"осадков. Летний сток составляет {summer_ratio:.1%} годового объема. "
                    "Характерен для муссонных климатов или регионов с конвективными ливнями. "
                    "Относительно стабильный режим с умеренной внутригодовой изменчивостью."
                )

        elif dominant_season == "autumn" and 244 <= peak_doy <= 334:
            # Pluvial regime: rainfall-dominated
            short_name = "Pluvial"
            regime_type = "Дождевой"
            description = (
                "Дождевой режим с максимумом стока в осенний период от интенсивных осадков "
                f"и сниженного испарения. Осенний сток составляет {autumn_ratio:.1%}. "
                "Характерен для приморских регионов с циклональной активностью или зон с "
                "осенними муссонами. Снегонакопление незначительно, зимний минимум обусловлен "
                "промерзанием почв."
            )

        elif dominant_season == "winter":
            # Groundwater-fed or regulated
            short_name = "Baseflow"
            regime_type = "Грунтовое питание"
            description = (
                "Режим грунтового питания с относительно равномерным стоком и максимумом "
                f"в зимний период. Низкая изменчивость (CV={cv:.2f}) указывает на преобладание "
                "подземного стока. Характерен для карстовых районов, зон с глубоким залеганием "
                "грунтовых вод или зарегулированных рек. Поверхностный сток минимален."
            )

        else:
            # Complex/transitional regime
            short_name = "Complex"
            regime_type = "Сложный"
            description = (
                f"Сложный режим с относительно равномерным распределением стока по сезонам. "
                f"Доминирует {season_names_ru[dominant_season].lower()} период "
                f"({dominant_ratio:.1%}), вторичный пик в {season_names_ru[secondary_season].lower()} "
                f"сезон ({secondary_ratio:.1%}). Высокая сложность режима "
                f"(индекс={regime_complexity:.2f}) свидетельствует о множественных источниках "
                "питания и переходном характере гидрологических процессов."
            )

        # Add variability descriptor
        if cv < 0.3:
            variability = "стабильный"
        elif cv < 0.5:
            variability = "умеренно изменчивый"
        elif cv < 0.7:
            variability = "изменчивый"
        else:
            variability = "высокоизменчивый"

        # Append variability to type name
        regime_type = f"{regime_type} ({variability})"

        return short_name, regime_type, description


    for _clust_name, _clust_df in cluster_q_mm.items():
        # Compute cluster median pattern (366 days)
        cluster_median = _clust_df.median(axis=1).values

        # Find peak DOY (1-366)
        peak_doy = int(np.argmax(cluster_median)) + 1  # Convert 0-indexed to 1-indexed
        peak_value = float(cluster_median[peak_doy - 1])

        # Calculate seasonal ratios
        # Winter: DOY 335-365 + 1-59
        winter_indices = list(range(334, 365)) + list(range(59))
        winter_mean = float(cluster_median[winter_indices].mean())

        # Spring: DOY 60-151
        spring_mean = float(cluster_median[59:151].mean())

        # Summer: DOY 152-243
        summer_mean = float(cluster_median[151:243].mean())

        # Autumn: DOY 244-334
        autumn_mean = float(cluster_median[243:334].mean())

        # Total for normalization
        total_seasonal = winter_mean + spring_mean + summer_mean + autumn_mean

        # Normalized ratios
        winter_ratio = winter_mean / total_seasonal if total_seasonal > 0 else 0.0
        spring_ratio = spring_mean / total_seasonal if total_seasonal > 0 else 0.0
        summer_ratio = summer_mean / total_seasonal if total_seasonal > 0 else 0.0
        autumn_ratio = autumn_mean / total_seasonal if total_seasonal > 0 else 0.0

        # Calculate coefficient of variation
        cv = (
            float(cluster_median.std() / cluster_median.mean())
            if cluster_median.mean() > 0
            else 0.0
        )

        # Classify using improved multi-indicator approach
        short_name, regime_type, description = classify_hydrological_regime(
            cluster_median, winter_ratio, spring_ratio, summer_ratio, autumn_ratio, peak_doy, cv
        )

        # Store results
        hydro_regime_info[_clust_name] = {
            "short_name": short_name,
            "regime_type": regime_type,
            "description": description,
            "peak_doy": peak_doy,
            "peak_value": peak_value,
            "winter_ratio": winter_ratio,
            "spring_ratio": spring_ratio,
            "summer_ratio": summer_ratio,
            "autumn_ratio": autumn_ratio,
            "cv": cv,
            "n_gauges": len(_clust_df.columns),
        }

    # Convert to DataFrame for display
    regime_df = pd.DataFrame.from_dict(hydro_regime_info, orient="index")
    regime_df.index.name = "cluster_name"
    regime_df.reset_index(inplace=True)

    # Reorder columns for better presentation
    column_order = [
        "cluster_name",
        "n_gauges",
        "short_name",
        "regime_type",
        "description",
        "peak_doy",
        "peak_value",
        "winter_ratio",
        "spring_ratio",
        "summer_ratio",
        "autumn_ratio",
        "cv",
    ]
    regime_df = regime_df[column_order]

    log.info(f"Extracted regime info for {len(regime_df)} hydrological clusters")
    print("\nHydrological Regime Classification Results:")
    print("=" * 100)

    for _, row in regime_df.iterrows():
        print(f"\n{row['cluster_name']} (n={row['n_gauges']} gauges)")
        print(f"{'─' * 100}")
        print(f"  Short Name:     {row['short_name']}")
        print(f"  Regime Type:    {row['regime_type']}")
        print(f"\n  Description:")
        print(f"    {row['description']}")
        print(f"\n  Quantitative Metrics:")
        print(f"    Peak DOY:       {row['peak_doy']} (normalized value: {row['peak_value']:.3f})")
        print(
            f"    Seasonal Ratios - Winter: {row['winter_ratio']:.1%} | "
            f"Spring: {row['spring_ratio']:.1%} | "
            f"Summer: {row['summer_ratio']:.1%} | "
            f"Autumn: {row['autumn_ratio']:.1%}"
        )
        print(f"    Coefficient of Variation: {row['cv']:.3f}")

    # Save to CSV for further use
    regime_output_path = hydro_table_dir / f"hydrological_regime_classification_{n_hydro_clusters}.csv"
    regime_df.to_csv(regime_output_path, index=False, encoding="utf-8-sig")
    log.info(f"Saved regime classification to {regime_output_path}")

    print(f"\n{'=' * 100}")
    print(f"Results saved to: {regime_output_path}")

    regime_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Hybrid clustering
    """)
    return


@app.cell
def _(
    basemap_data,
    gauges,
    geo_scaled,
    hydro_labels,
    pd,
    plot_hybrid_spatial_clusters,
    q_df_clust,
    ws,
):
    # Create hybrid labels (Ф#-Г# format)
    geo_labels = [f'{cl}' for cl in geo_scaled['cluster_geo']]
    _hydro_labels = [f'{cl}' for cl in hydro_labels]
    # Create mapping for common gauges
    common_gauges = q_df_clust.index.to_list()
    geo_mapping = pd.Series(geo_labels, index=geo_scaled.index)
    hydro_mapping = pd.Series(_hydro_labels, index=q_df_clust.index)

    # Build gauge mapping with hybrid labels
    gauge_mapping = pd.DataFrame(
        {
            "gauge_id": common_gauges,
            "geo_cluster": geo_mapping.loc[common_gauges].values,
            "hydro_cluster": hydro_mapping.loc[common_gauges].values,
        }
    )

    # Create hybrid labels
    gauge_mapping["hybrid_combo"] = gauge_mapping.apply(
        lambda row: f"Ф{row['geo_cluster']}-Г{row['hydro_cluster']}", axis=1
    )

    print(f"\nRaw hybrid combinations: {gauge_mapping['hybrid_combo'].nunique()}")

    # Consolidate hybrid combinations using proper strategy
    # Strategy: Keep top combinations, merge rare ones by geo cluster
    combo_counts = gauge_mapping["hybrid_combo"].value_counts()
    target_n_classes = 15
    min_gauges = 5

    print(f"Combination counts:\n{combo_counts}\n")

    # Keep significant combinations
    significant_combos = combo_counts[combo_counts >= min_gauges].head(target_n_classes)
    print(f"Significant combinations (≥{min_gauges} gauges): {len(significant_combos)}")

    # Create mapping
    combo_mapping = {}

    # Assign significant combos directly
    for combo in significant_combos.index:
        combo_mapping[combo] = combo

    # For rare combinations, merge by geo cluster
    rare_combos = combo_counts[~combo_counts.index.isin(significant_combos.index)]

    for combo in rare_combos.index:
        geo_cluster = combo.split("-")[0]  # Extract Ф1, Ф2, etc.

        # Find significant combos with same geo cluster
        geo_matches = [c for c in significant_combos.index if c.startswith(geo_cluster + "-")]

        if geo_matches:
            # Assign to most common geo-matched combo
            combo_mapping[combo] = geo_matches[0]
        else:
            # Group as "Ф#-Mixed"
            combo_mapping[combo] = f"{geo_cluster}-Mixed"

    # Apply mapping
    gauge_mapping["hybrid_class"] = gauge_mapping["hybrid_combo"].map(combo_mapping)

    print(f"\nConsolidated to {gauge_mapping['hybrid_class'].nunique()} hybrid classes")
    print("\nClass distribution:")
    print(gauge_mapping["hybrid_class"].value_counts())

    # Summary table of hybrid classes
    class_summary = (
        gauge_mapping.groupby("hybrid_class")
        .agg(
            {
                "gauge_id": "count",
                "geo_cluster": lambda x: sorted(set(x)),
                "hydro_cluster": lambda x: sorted(set(x)),
            }
        )
        .rename(columns={"gauge_id": "n_gauges"})
    )

    class_summary = class_summary.sort_values("n_gauges", ascending=False)
    class_summary["geo_clusters"] = class_summary["geo_cluster"].apply(
        lambda x: ", ".join([f"Ф{i}" for i in x])
    )
    class_summary["hydro_clusters"] = class_summary["hydro_cluster"].apply(
        lambda x: ", ".join([f"Г{i}" for i in x])
    )

    print("\nHybrid Class Summary:")
    print("=" * 80)
    print(class_summary[["n_gauges", "geo_clusters", "hydro_clusters"]].to_string())
    print("=" * 80)
    print(f"\nTotal classes: {len(class_summary)}")
    print(f"Total gauges: {class_summary['n_gauges'].sum()}")
    print(f"Mean gauges per class: {class_summary['n_gauges'].mean():.1f}")
    print(f"Median gauges per class: {class_summary['n_gauges'].median():.1f}")

    # Plot hybrid spatial distribution with watersheds
    fig_hybrid_ws = plot_hybrid_spatial_clusters(
        watersheds=ws,
        gauges=gauges,
        gauge_mapping=gauge_mapping,
        hybrid_col="hybrid_class",
        basemap=basemap_data,
        output_path="../res/chapter_one/map_hybrid_classes_watersheds.png",
        title=f"Hybrid Classification - Watersheds ({len(class_summary)} classes)",
        show_watersheds=True,
        show_gauges=True,
    )
    fig_hybrid_ws
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Plot Function Update
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
