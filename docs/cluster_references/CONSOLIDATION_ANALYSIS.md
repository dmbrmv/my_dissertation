# Hybrid Cluster Consolidation Analysis

## Problem Statement

**Issue**: NewChapterOne.ipynb produces **25 hybrid classes** instead of the expected **~16 classes**

**Goal**: Understand why ClusterExperiments produces ~16 classes and apply the same methodology

## Root Cause Analysis

### Key Differences Between Approaches

| Aspect | ClusterExperiments (✓ ~16 classes) | NewChapterOne (✗ 25 classes) |
|--------|-------------------------------------|-------------------------------|
| **Consolidation Method** | Hex-based spatial aggregation FIRST, then consolidation | Direct gauge-level consolidation |
| **Function Used** | `consolidate_hybrid_combinations()` from analytics module | Inline consolidation logic |
| **Spatial Context** | Uses 80km hex grid to aggregate similar neighboring regions | No spatial aggregation |
| **Target Parameter** | `target_n_classes=15` | `target_n_classes=16` |
| **Merging Strategy** | Merges by spatial proximity in hexes | Merges by frequency threshold only |

### Why Hex-Based Aggregation Works Better

1. **Spatial Coherence**: Combines geographically close watersheds with similar characteristics
2. **Reduces Noise**: Filters out isolated singleton combinations that create artificial diversity
3. **Meaningful Grouping**: Creates classes that represent regional patterns, not just statistical combinations

### NewChapterOne's Consolidation Logic Issue

The current consolidation in NewChapterOne:

```python
# Keep top N combinations
if len(common_combos) > target_n_classes:
    top_combos = combo_counts.head(target_n_classes).index.tolist()
    merge_combos = [c for c in combo_counts.index if c not in top_combos]
```

**Problem**: This keeps top 16 **raw combinations** but then adds "Ф#-Mixed" classes for each geo cluster, resulting in:
- 16 preserved combinations
- Up to 9 "Ф#-Mixed" classes (one per geo cluster)
- **Total: 16 + 9 = 25 classes**

### ClusterExperiments' Hex-Based Approach

```python
# Step 1: Aggregate watersheds to hexes by geo cluster
hexes_with_geo = aggregate_clusters_to_hex(watersheds, hexes, "geo_cluster")

# Step 2: Map hydro clusters onto geo-clustered hexes
hexes_with_both = map_secondary_cluster_to_hex(
    watersheds, hexes_with_geo, "geo_cluster", "hydro_cluster"
)

# Step 3: Assign gauges to hex-based hybrid classes
gauges_with_combos = assign_gauges_to_hybrid_classes(
    gauges, watersheds, hexes_with_combos, k_neighbors=5
)

# Step 4: Consolidate to target N classes
gauges_final = consolidate_hybrid_combinations(
    gauges_with_combos, target_n_classes=15, min_gauges_per_class=5
)
```

**Result**: ~15-16 spatially coherent classes

## Solution: HybridClusterAnalysis.ipynb

Created new notebook that follows ClusterExperiments methodology:

### Features

1. **Minimal Output**: Only essential diagnostics, no verbose logging
2. **Hex-Based Aggregation**: Uses 80km hex grid for spatial consolidation
3. **Proper Function Usage**: Leverages existing analytics module functions
4. **Russian Naming**: Applies Ф#-Г# convention throughout
5. **DOY Peak Timing**: Extracts regime characteristics from discharge patterns
6. **Full Export**: Generates CSV mapping + YAML reference

### Workflow

```
Load Geo Clusters (9) → Load Hydro Clusters (10) → Create Raw Combos (Ф#-Г#)
    ↓
Build Hex Grid (80km) → Aggregate Geo → Map Hydro → Hex-Level Combos
    ↓
Assign Gauges to Hexes → Consolidate to ~15 Classes → Apply Russian Names
    ↓
Extract DOY Timing → Export CSV + YAML
```

### Expected Output

- **Final classes**: 15-16 (not 25)
- **Naming**: Ф1-Г1, Ф1-Г3, Ф2-Г5, Ф3-Mixed, etc.
- **Spatial coherence**: Classes represent regional patterns
- **Regime info**: Peak DOY and regime type for each hydro cluster

## Verification Steps

1. Run HybridClusterAnalysis.ipynb
2. Check final class count in Step 5 output
3. Verify class distribution is reasonable (mean ~50-70 gauges/class)
4. Confirm no excessive "Mixed" classes
5. Validate CSV export has both `hybrid_combo_ru` and `hybrid_class_ru`

## Files

- **New Notebook**: `notebooks/HybridClusterAnalysis.ipynb` (streamlined, minimal output)
- **Reference**: `notebooks/ClusterExperiments.ipynb` (original working version)
- **Problematic**: `notebooks/NewChapterOne.ipynb` (produces 25 classes)

## Recommendation

Use **HybridClusterAnalysis.ipynb** for production hybrid classification. It follows proven methodology from ClusterExperiments with minimal diagnostic output.
