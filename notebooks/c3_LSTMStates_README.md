# LSTM States Analysis - Documentation

## Overview

This analysis examines how LSTM hidden states (`h_n`) and cell states (`c_n`) internally represent **meteorological processes NOT included as model inputs**.

## Data Structure

**States array shape**: `(996, 730, 256)`
- **996**: Number of gauges
- **730**: Timesteps (2019-2020 test period)
- **256**: Hidden units

Each gauge has its **own** state trajectory, enabling proper per-gauge correlation analysis.

## Key Files

| File | Description |
|------|-------------|
| `c3_ExtractLSTMStates.ipynb` | Extracts per-gauge states from LSTM model (run on CUDA machine) |
| `c3_LSTMStates_v2.ipynb` | Main analysis notebook (can run on Mac) |
| `all_gauges_states.npz` | Extracted states: `h_n`, `c_n`, `gauge_ids` |

## Scientific Background

### LSTM Theory: h_n vs c_n

| State | Name | Theoretical Role | Expected Behavior |
|-------|------|------------------|-------------------|
| **c_n** | Cell State | Long-term memory | Captures **slowly-varying, persistent** patterns (seasonal cycles) |
| **h_n** | Hidden State | Short-term output | More **responsive to recent inputs** (quick fluctuations) |

### Hypothesis

- **h_n** should correlate better with quick/variable processes (evaporation, subsurface)
- **c_n** should correlate better with seasonal/persistent patterns (SWE, snow_depth)

### Processes Analyzed (NOT in model inputs)

| Parameter | Source | Physical Meaning | Temporal Scale |
|-----------|--------|------------------|----------------|
| Evaporation | GLEAM | Water loss to atmosphere | Short-term |
| SWE | ERA5-Land | Snow water storage | Seasonal |
| Snow Depth | ERA5-Land | Snow accumulation | Seasonal |
| Subsurface | ERA5-Land | Groundwater/baseflow | Short-term |

## Methodology

### 1. Per-Gauge Correlation Analysis

**Critical improvement over previous version**:
- **Before**: One state trajectory correlated with 996 different gauges' meteo data (WRONG)
- **Now**: Each gauge's state trajectory correlated with its OWN meteo data (CORRECT)

For each gauge:
1. Get gauge's state trajectory: `(730 timesteps, 256 cells)`
2. Load gauge's meteo data: `(730 timesteps,)`
3. Compute Pearson correlation for each cell: `r = corr(cell_trajectory, meteo_trajectory)`

### 2. Signed Correlations

Preserves positive/negative relationships:
- **Positive correlation**: Cell activates when process increases
- **Negative correlation**: Cell inhibits when process increases

Groups: `{process}_pos` or `{process}_neg`

### 3. Z-Score Ranking for Cell Assignment

**Problem**: Snow processes often have higher raw correlations, dominating assignments

**Solution**: Normalize correlations within each process using Z-scores:
```
z_score[cell, process] = (r[cell, process] - mean_r[process]) / std_r[process]
```

This ensures weaker-correlation processes (subsurface, evaporation) still get cell representation.

### 4. Per-Gauge Dominant Process

For each gauge, determine which process has the strongest representation using Z-score normalized comparisons.

## Output Files

### Tables (`res/chapter_three/tables/`)

| File | Content |
|------|---------|
| `lstm_v2_hn_cell_assignment.csv` | h_n cell-to-process assignments |
| `lstm_v2_cn_cell_assignment.csv` | c_n cell-to-process assignments |
| `lstm_v2_hn_cn_comparison.csv` | h_n vs c_n overlap statistics |
| `lstm_v2_gauge_dominant.csv` | Per-gauge dominant process |
| `lstm_v2_hypothesis_test.csv` | Temporal scale hypothesis results |
| `lstm_v2_cluster_process_crosstab.csv` | Hybrid cluster x process counts |

### Figures (`res/chapter_three/images/`)

| File | Content |
|------|---------|
| `lstm_v2_hn_cell_grid.png` | 16x16 grid of h_n cell assignments |
| `lstm_v2_hn_dominant_process_map.png` | Spatial map of dominant processes |
| `lstm_v2_hn_cn_comparison_grid.png` | Side-by-side h_n vs c_n grids |
| `lstm_v2_cluster_process_heatmap.png` | Cluster x process percentage heatmap |

## Interpreting Results

### Cell Grid

Each cell in the 16x16 grid represents one of 256 LSTM hidden units. Colors indicate which meteorological process that cell best represents.

### Hypothesis Test Results

Check `lstm_v2_hypothesis_test.csv`:
- If `Matches Hypothesis = True` for most processes, the LSTM follows expected temporal dynamics
- `h_n Mean |r|` vs `c_n Mean |r|` shows which state has stronger correlations

### Cluster Analysis

Check `lstm_v2_cluster_process_crosstab.csv`:
- Do gauges in same hybrid cluster have similar dominant processes?
- High within-cluster homogeneity suggests LSTM organization aligns with hydrological regimes

## Common Issues

### 1. Missing meteo data for some gauges
- Solution: Gauges with missing data are excluded from correlation analysis
- Check: Printed "Valid gauges" count should be close to 996

### 2. All cells assigned to same process
- Cause: One process dominates (usually snow)
- Solution: Z-score normalization addresses this

### 3. Low correlations overall
- Check: `MIN_CORRELATION = 0.3` threshold
- Cells below threshold assigned to "inactive" group

## Code References

### Loading states
```python
states = np.load("all_gauges_states.npz")
h_states_all = states["h_n"]  # (996, 730, 256)
c_states_all = states["c_n"]  # (996, 730, 256)
gauge_ids = states["gauge_ids"]  # (996,)
```

### Computing correlation for one gauge
```python
gauge_idx = 0
gauge_states = h_states_all[gauge_idx]  # (730, 256)
meteo_data = load_meteo_data(gauge_ids[gauge_idx], "evaporation")  # (730,)
correlations = compute_cell_correlations(gauge_states, meteo_data)  # (256,)
```

### Finding dominant process
```python
# For each gauge, find process with max Z-score normalized correlation
gauge_zscores = (gauge_max_corr - gauge_max_corr.mean()) / gauge_max_corr.std()
dominant_process = gauge_zscores.idxmax()
```

## Historical Context

Previous version (`example_states.npz`) had shape `(731, 256)` which represented only ONE gauge's time series. This was incorrectly correlated with all 996 gauges' meteo data.

The corrected version extracts states for ALL gauges: `(996, 730, 256)`.

---

*Last updated: 2026-01-28*
