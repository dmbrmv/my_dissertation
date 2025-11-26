# Plotting Functions Usage Guide

## Overview
This guide provides practical examples for using the refactored plotting functions with the new unified parameter interface.

---

## 1. `plot_spatial_clusters()` - Spatial cluster visualization

### Basic Usage
```python
from src.plots.cluster_plots import plot_spatial_clusters
import geopandas as gpd

# Load your data
geo_df = gpd.read_file("path/to/geodata.shp")
basemap = gpd.read_file("path/to/basemap.shp")

# Simple plot
fig = plot_spatial_clusters(
    gdf=geo_df,
    basemap=basemap,
    cluster_col="cluster",
    title="Geographical Clusters",
    output_path="results/spatial_clusters.png"
)
```

### Advanced Usage with Custom Colors and Histogram
```python
# Define custom colors and markers
custom_colors = ["#FF5733", "#33FF57", "#3357FF", "#FF33A1", "#FFC300"]
custom_markers = ["o", "s", "^", "v", "D"]

fig = plot_spatial_clusters(
    gdf=geo_df,
    basemap=basemap,
    cluster_col="cluster",
    
    # Color customization
    color_list=custom_colors,  # Overrides cmap_name
    cmap_name="viridis",  # Used if color_list not provided
    skip_colors=0,
    
    # Marker customization
    markers_list=custom_markers,
    base_marker_size=30,
    marker_size_variants=[1.0, 1.4, 1.8],  # For >12 categories
    linewidth_variants=[1.0, 1.6, 2.2],
    marker_size_corrections={"s": 0.85, "^": 1.15},  # Visual compensation
    
    # Histogram
    with_histogram=True,
    histogram_col="cluster",
    histogram_rect=(0.05, 0.05, 0.30, 0.24),  # [x, y, width, height]
    histogram_bar_colors=custom_colors,
    histogram_xticklabels=["A", "B", "C", "D", "E"],
    histogram_label_rotation=45.0,
    
    # Legend
    legend_cols=3,
    legend_auto_position=True,
    legend_fontsize=10,
    
    # Basemap styling
    basemap_color="gray",
    basemap_edgecolor="black",
    basemap_linewidth=0.3,
    basemap_alpha=0.8,
    
    title="Advanced Spatial Clusters",
    figsize=(10, 8),
    output_path="results/spatial_clusters_advanced.png"
)
```

---

## 2. `plot_hybrid_spatial_clusters()` - Hybrid watershed/gauge visualization

### Basic Usage
```python
from src.plots.cluster_plots import plot_hybrid_spatial_clusters

fig = plot_hybrid_spatial_clusters(
    watersheds=watersheds_gdf,
    gauges=gauges_gdf,
    gauge_mapping=mapping_dict,  # {gauge_id: watershed_id}
    hybrid_col="hybrid_class",
    basemap=basemap_gdf,
    title="Hybrid Classification",
    show_watersheds=True,
    show_gauges=True,
    output_path="results/hybrid_clusters.png"
)
```

### Advanced Usage
```python
fig = plot_hybrid_spatial_clusters(
    watersheds=watersheds_gdf,
    gauges=gauges_gdf,
    gauge_mapping=mapping_dict,
    hybrid_col="hybrid_class",
    basemap=basemap_gdf,
    
    # Colors and markers
    color_list=["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"],
    markers_list=["o", "s", "^", "D"],
    marker_size_variants=[1.0, 1.4, 1.8],
    base_linewidth=0.35,
    linewidth_variants=[1.0, 1.6, 2.2],
    
    # Histogram
    with_histogram=True,
    histogram_col="hybrid_class",
    histogram_rect=(0.02, 0.02, 0.35, 0.25),
    histogram_bar_colors=["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"],
    histogram_label_rotation=30.0,
    
    # Legend
    legend_cols=2,
    legend_auto_position=True,
    
    # Display options
    show_watersheds=True,
    show_gauges=True,
    watershed_alpha=0.3,
    
    title="Hybrid Classification (Watersheds + Gauges)",
    figsize=(12, 8),
    output_path="results/hybrid_advanced.png"
)
```

---

## 3. `hex_model_distribution_plot()` - Hexagonal grid aggregation

### Basic Usage
```python
from src.plots.hex_maps import hex_model_distribution_plot

fig, ax, hex_grid, counts = hex_model_distribution_plot(
    watersheds=watersheds_gdf,
    basemap_data=basemap_gdf,
    model_col="model_type",
    model_dict={"Model A": 1, "Model B": 2, "Model C": 3},
    title="Model Distribution"
)
```

### Advanced Usage
```python
fig, ax, hex_grid, counts = hex_model_distribution_plot(
    watersheds=watersheds_gdf,
    basemap_data=basemap_gdf,
    model_col="model_type",
    model_dict={"GR4J": 1, "HBV": 2, "LSTM": 3, "Ambiguous": 4},
    
    # Hex grid parameters
    r_km=75.0,  # Hex radius in km (None = auto-calculate)
    target_ws_per_hex=6.0,  # Target watersheds per hex
    min_overlap_share=0.15,  # Minimum overlap fraction
    dominant_threshold=0.33,  # Minimum frequency for non-ambiguous
    ambiguous_label="Неоднозначно",
    
    # Colors
    color_list=["#3498DB", "#E74C3C", "#2ECC71", "#95A5A6"],
    cmap_name="turbo",  # Used if color_list not provided
    skip_colors=0,
    
    # Histogram (RENAMED PARAMETERS)
    with_histogram=True,  # Was 'histogram'
    histogram_col="model_type",  # NEW parameter
    histogram_rect=(0.05, 0.05, 0.30, 0.24),
    histogram_xticklabels=["GR4J", "HBV", "LSTM", "Mixed"],  # NEW parameter
    histogram_label_rotation=45.0,
    histogram_count_format="%d",  # NEW parameter
    
    # Legend (RENAMED PARAMETERS)
    legend_show=True,  # Was 'legend'
    legend_cols=2,  # Was 'legend_columns'
    legend_kwargs={"fontsize": 9},
    
    # Basemap (NEW PARAMETERS)
    basemap_color="grey",
    basemap_edgecolor="black",
    basemap_linewidth=0.4,
    basemap_alpha=0.8,
    
    rus_extent=(50, 140, 32, 90),  # [lon_min, lon_max, lat_min, lat_max]
    figsize=(10.0, 5.0),
    title="Model Distribution by Hex Grid"
)

# Access results
print(f"Total hexes: {len(hex_grid)}")
print(f"Category counts:\n{counts}")
```

---

## 4. `russia_plots()` - General Russia map plotting

### Point-based Plotting
```python
from src.plots.maps import russia_plots

fig = russia_plots(
    gdf_to_plot=gauges_gdf,
    basemap_data=basemap_gdf,
    distinction_col="cluster",
    just_points=True,  # Enable point plotting mode
    
    # Colors and markers
    color_list=["#FF5733", "#33FF57", "#3357FF"],
    markers_list=["o", "s", "^"],
    base_marker_size=30,
    base_linewidth=0.35,
    marker_size_corrections={"s": 0.8},
    
    # Histogram
    with_histogram=True,
    specific_xlabel=["Cluster A", "Cluster B", "Cluster C"],
    
    # Legend
    legend_cols=3,
    
    title_text="Gauge Clusters",
    figsize=(4.88189, 3.34646),
    rus_extent=[50, 140, 32, 90]
)
```

### Polygon-based Plotting
```python
fig = russia_plots(
    gdf_to_plot=watersheds_gdf,
    basemap_data=basemap_gdf,
    distinction_col="",
    metric_col="nse",  # Continuous metric
    just_points=False,  # Polygon mode
    
    # Colormap for continuous data
    cmap_name="RdYlGn",
    cmap_lims=(0.0, 1.0),
    list_of_limits=[0.0, 0.4, 0.6, 0.8, 1.0],
    
    with_histogram=True,
    title_text="Model Performance (NSE)",
    figsize=(6, 4)
)
```

---

## Key Parameter Changes (Migration Guide)

### hex_model_distribution_plot
| Old Parameter | New Parameter | Notes |
|--------------|---------------|-------|
| `histogram` | `with_histogram` | Boolean flag |
| `legend_columns` | `legend_cols` | Integer |
| `legend` | `legend_show` | Boolean flag |
| N/A | `color_list` | Explicit color override |
| N/A | `histogram_col` | Custom histogram column |
| N/A | `histogram_xticklabels` | Custom x-labels |
| N/A | `histogram_count_format` | Bar label format |
| N/A | `basemap_color` | Basemap fill color |
| N/A | `basemap_edgecolor` | Basemap edge color |
| N/A | `basemap_linewidth` | Basemap line width |

### All Functions
- **`colors`** → **`color_list`** (explicit color array)
- **`histogram`** → **`with_histogram`** (consistent naming)
- **`legend_columns`** → **`legend_cols`** (shorter name)

---

## Common Patterns

### Pattern 1: Custom Colors Override Colormap
```python
# color_list takes precedence over cmap_name
fig = plot_spatial_clusters(
    ...,
    color_list=["#FF0000", "#00FF00", "#0000FF"],  # Used
    cmap_name="viridis"  # Ignored when color_list provided
)
```

### Pattern 2: Marker Cycling for Large Category Sets
```python
# For >12 categories, markers repeat with size/linewidth variants
fig = plot_spatial_clusters(
    ...,
    markers_list=["o", "s", "^"],  # Only 3 markers
    marker_size_variants=[1.0, 1.4, 1.8],  # 3 variants
    # Result: 3 × 3 = 9 distinguishable marker combinations
)
```

### Pattern 3: Histogram Positioning
```python
# histogram_rect = [x, y, width, height] in axes coordinates (0-1)
fig = plot_spatial_clusters(
    ...,
    with_histogram=True,
    histogram_rect=(0.05, 0.05, 0.30, 0.24),  # Lower-left corner
    # OR
    histogram_rect=(0.65, 0.70, 0.30, 0.24)   # Upper-right corner
)
```

---

## Shared Utilities Reference

All functions now use `src/plots/styling_utils.py`:

- **`get_russia_projection()`** - Albers Equal Area CRS
- **`sort_plot_categories()`** - Letter → numeric → alphabetical sorting
- **`get_unified_colors()`** - Color sampling/cycling
- **`get_marker_system()`** - Marker cycling logic
- **`create_legend_handles()`** - Line2D legend entries
- **`render_basemap()`** - Basemap rendering
- **`add_histogram_inset()`** - Histogram overlays

---

## Troubleshooting

### Issue: Colors not matching between map and histogram
**Solution:** Ensure `histogram_bar_colors` matches `color_list`:
```python
colors = ["#FF0000", "#00FF00", "#0000FF"]
fig = plot_spatial_clusters(
    ...,
    color_list=colors,
    histogram_bar_colors=colors  # Must match
)
```

### Issue: Too many categories (>36)
**Solution:** Increase `marker_size_variants` and `linewidth_variants`:
```python
fig = plot_spatial_clusters(
    ...,
    markers_list=["o", "s", "^", "v", "<", ">", "P", "X", "D", "*", "h", "H"],  # 12 markers
    marker_size_variants=[1.0, 1.3, 1.6, 1.9],  # 4 variants
    linewidth_variants=[1.0, 1.4, 1.8, 2.2]     # 4 variants
    # Result: 12 × 4 = 48 combinations
)
```

### Issue: Histogram labels overlapping
**Solution:** Rotate labels or reduce font size:
```python
fig = plot_spatial_clusters(
    ...,
    histogram_label_rotation=45.0,  # Rotate 45°
    histogram_rect=(0.05, 0.05, 0.40, 0.24)  # Widen histogram
)
```

---

## Complete Example Workflow

```python
# Complete example for Chapter One analysis
import geopandas as gpd
from src.plots.cluster_plots import plot_spatial_clusters, plot_hybrid_spatial_clusters
from src.plots.hex_maps import hex_model_distribution_plot

# 1. Load data
geo_scaled = gpd.read_file("data/geo_clusters.shp")
watersheds = gpd.read_file("data/watersheds.shp")
gauges = gpd.read_file("data/gauges.shp")
basemap = gpd.read_file("data/russia_basemap.shp")

# 2. Define consistent styling
n_clusters = 6
colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
markers = ["o", "s", "^", "v", "D", "P"]
marker_corrections = {"s": 0.85, "^": 1.1, "v": 1.05}

# 3. Create spatial cluster map
fig_spatial = plot_spatial_clusters(
    gdf=geo_scaled,
    basemap=basemap,
    cluster_col="cluster",
    color_list=colors,
    markers_list=markers,
    marker_size_corrections=marker_corrections,
    with_histogram=True,
    histogram_col="cluster",
    legend_cols=3,
    title=f"Geographical Clusters (n={n_clusters})",
    output_path="res/chapter_one/geo_clusters.png"
)

# 4. Create hybrid map
gauge_mapping = {g: w for g, w in zip(gauges["gauge_id"], gauges["watershed_id"])}
fig_hybrid = plot_hybrid_spatial_clusters(
    watersheds=watersheds,
    gauges=gauges,
    gauge_mapping=gauge_mapping,
    hybrid_col="hybrid_class",
    basemap=basemap,
    color_list=colors,
    markers_list=markers,
    with_histogram=True,
    legend_cols=2,
    title="Hybrid Classification",
    show_watersheds=True,
    show_gauges=True,
    output_path="res/chapter_one/hybrid_clusters.png"
)

# 5. Create hex grid model distribution
fig_hex, ax, grid, counts = hex_model_distribution_plot(
    watersheds=watersheds,
    basemap_data=basemap,
    model_col="best_model",
    model_dict={"GR4J": 1, "HBV": 2, "LSTM": 3},
    color_list=["#3498DB", "#E74C3C", "#2ECC71"],
    with_histogram=True,
    histogram_xticklabels=["GR4J", "HBV", "LSTM"],
    legend_cols=3,
    title="Best Model Distribution"
)

print("All plots generated successfully!")
```

---

## Notes

- All functions maintain backward compatibility through wrapper functions
- Parameter validation raises `ValueError` with actionable messages
- Colormap sampling uses `skip_colors` to avoid extreme colors
- Category sorting: letter prefix (`a)`, `b)`) → numeric suffix → alphabetical
- Projection: Albers Equal Area (central_longitude=100, standard_parallels=(50,70))

For more details, see:
- `src/plots/styling_utils.py` - Shared utilities
- `src/plots/cluster_plots.py` - Cluster plotting functions
- `src/plots/hex_maps.py` - Hex grid functions
- `src/plots/maps.py` - General mapping functions
