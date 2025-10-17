# Random Forest Regressor with Enhanced Features for Hydrological Modeling

## Overview

This implementation extends the baseline Random Forest Regressor (RFR) with advanced feature engineering specifically designed for streamflow prediction in hydrological applications. The enhanced approach integrates:

1. **Potential Evapotranspiration (PET)** — Water balance component
2. **Cyclic temporal encoding** — Seasonal pattern representation
3. **Static watershed attributes** — Spatial catchment characteristics
4. **Rolling window aggregations** — Temporal memory and dependencies

These enhancements enable the model to capture physical processes, seasonal patterns, spatial heterogeneity, and temporal dynamics—critical for accurate streamflow prediction and **ungauged basin regionalization**.

---

## Feature Engineering Pipeline

### 1. Dynamic Meteorological Features

#### Base Features
- **Precipitation** (`prcp_{dataset}`) — mm/day
- **Temperature** (`t_min_e5l`, `t_max_e5l`) — °C
- **PET** (`pet_mm_day`) — Potential evapotranspiration [mm/day] via **Oudin (2005) formula**

#### Rolling Window Aggregations
For each base feature, we compute rolling statistics over multiple window sizes:
- **Windows**: 1, 2, 4, 8, 16, 32 days
- **Precipitation & PET**: Cumulative sum (captures antecedent water input/loss)
- **Temperature**: Mean (captures recent thermal conditions)

**Rationale**: Streamflow responds to precipitation and temperature patterns over days to weeks due to:
- Soil moisture storage and drainage
- Snowmelt accumulation
- Groundwater recharge lag
- Catchment routing time

**Feature count**: ~40 dynamic features per dataset
- 3 base features × 6 windows × 2 (sum + original) + temperature means

---

### 2. Potential Evapotranspiration (PET)

**Method**: Oudin (2005) formula
```
PET = (R_a / λ) × (T_mean + 5) / 100
```
where:
- `R_a` = Extraterrestrial radiation [MJ/m²/day] (function of latitude, day of year)
- `λ` = Latent heat of vaporization (2.45 MJ/kg)
- `T_mean` = Mean air temperature [°C]

**Implementation**: `src/models/gr4j/pet.py::pet_oudin()`

**Input requirements**:
- Mean temperature time series
- Day of year (1–366)
- Gauge latitude (decimal degrees)

**Why PET matters**:
- Water balance: Streamflow = Precipitation - Evapotranspiration - Storage
- Seasonal variability: High PET in summer → lower runoff ratios
- Climate sensitivity: PET links temperature to water availability

---

### 3. Cyclic Temporal Encoding (Day of Year)

**Transformation**:
```python
doy_sin = sin(2π × day_of_year / 366)
doy_cos = cos(2π × day_of_year / 366)
```

**Why cyclic encoding**:
- **Continuity**: Day 365 and Day 1 are adjacent (not 364 days apart)
- **Seasonality**: Captures annual patterns (snowmelt, monsoons, summer low flows)
- **Machine learning**: Tree-based models benefit from explicit seasonal signals

**Feature count**: 2 features (`doy_sin`, `doy_cos`)

**Reference**: Scikit-learn Time-related Feature Engineering  
https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html

---

### 4. Static Watershed Attributes (HydroATLAS)

**Source**: `data/attributes/hydro_atlas_cis_camels.csv`

**Selected features (17 total)**:

| Feature | Description | Units | Hydrological Relevance |
|---------|-------------|-------|------------------------|
| `for_pc_sse` | Forest cover | % | Interception, infiltration |
| `crp_pc_sse` | Cropland | % | Surface runoff, irrigation |
| `pst_pc_sse` | Pasture | % | Moderate infiltration |
| `urb_pc_sse` | Urban area | % | Impervious surface, fast runoff |
| `inu_pc_ult` | Inland water | % | Lake/wetland storage |
| `ire_pc_sse` | Irrigated area | % | Water abstraction |
| `lka_pc_use` | Lake area | % | Flow regulation, evaporation |
| `lkv_mc_usu` | Lake volume | million m³ | Storage capacity |
| `rev_mc_usu` | Reservoir volume | million m³ | Anthropogenic regulation |
| `cly_pc_sav` | Clay content | % | Low permeability, high storage |
| `slt_pc_sav` | Silt content | % | Moderate permeability |
| `snd_pc_sav` | Sand content | % | High permeability, low storage |
| `gwt_cm_sav` | Groundwater depth | cm | Baseflow potential |
| `kar_pc_sse` | Karst area | % | Subsurface flow, springs |
| `prm_pc_sse` | Permeable surface | % | Infiltration capacity |
| `ws_area` | Watershed area | km² | Flow volume, routing time |
| `ele_mt_sav` | Mean elevation | m | Temperature, precipitation gradient |

**Why static attributes matter**:
- **Spatial patterns**: Catchment characteristics explain inter-basin variability
- **Ungauged basins**: Transfer learning via physical similarity
- **Process representation**: Land cover, soil, and topography control runoff generation

**Feature count**: 17 features (time-invariant, tiled across timesteps)

---

## Model Architecture

### Random Forest Regressor Configuration

**Framework**: Scikit-learn `RandomForestRegressor`

**Hyperparameter optimization**: Optuna multi-objective
- **Objectives**:
  1. KGE (Kling-Gupta Efficiency) — Overall performance
  2. Composite low-flow metric (logNSE + invNSE) — Low-flow accuracy
  3. Composite high-flow metric (NSE + PFE) — Peak flow accuracy
  4. Percent bias (PBIAS) — Volume conservation

**Hyperparameter search space**:
- `n_estimators`: 200–1000 (100-step intervals)
- `max_depth`: None or 10–50 (5-step intervals)
- `min_samples_split`: 2–20
- `min_samples_leaf`: 1–10
- `max_features`: "sqrt", "log2", or {0.3, 0.5, 0.7, 1.0}
- `bootstrap`: True/False
- `max_samples`: 0.5–1.0 (if bootstrap=True)
- `min_impurity_decrease`: 0.0–0.01 (log scale)

**Regularization**: Prevents overfitting via depth limits, sample constraints, and impurity thresholds

---

## Total Feature Count

| Feature Category | Count | Description |
|------------------|-------|-------------|
| Base dynamic | 3 | prcp, t_min, t_max |
| PET | 1 | Oudin formula |
| Rolling windows | ~40 | 6 windows × (3 base + PET) × agg types |
| Cyclic temporal | 2 | doy_sin, doy_cos |
| Static attributes | 17 | HydroATLAS catchment characteristics |
| **Total** | **~63** | Per gauge-dataset combination |

*Actual count may vary by dataset availability and missing data*

---

## Usage

### Single Gauge Training (Testing)

```bash
# Activate conda environment
conda activate camels_ru

# Run single gauge test
cd /home/dmbrmv/Development/Dissertation
python scripts/rfr_train.py
```

### Multi-Gauge Parallel Optimization

```python
from pathlib import Path
from src.models.rfr.parallel import run_parallel_optimization, process_rfr_gauge
from src.readers.geom_reader import load_geodata

# Load gauge geometries
_, gauges = load_geodata(folder_depth=".")

# Configure
datasets = ["e5l", "gpcp", "e5", "mswep"]
static_file = Path("data/attributes/hydro_atlas_cis_camels.csv")
static_parameters = [
    "for_pc_sse", "crp_pc_sse", "inu_pc_ult", "ire_pc_sse",
    "lka_pc_use", "prm_pc_sse", "pst_pc_sse", "cly_pc_sav",
    "slt_pc_sav", "snd_pc_sav", "kar_pc_sse", "urb_pc_sse",
    "gwt_cm_sav", "lkv_mc_usu", "rev_mc_usu", "ws_area", "ele_mt_sav"
]

# Run optimization
run_parallel_optimization(
    gauge_ids=gauge_ids,
    process_gauge_func=process_rfr_gauge,
    n_processes=4,
    datasets=datasets,
    calibration_period=("2008-01-01", "2018-12-31"),
    validation_period=("2019-01-01", "2020-12-31"),
    save_storage=Path("data/optimization/rfr_results"),
    gauges_gdf=gauges,
    static_file=static_file,
    static_columns=static_parameters,
    rolling_windows=[1, 2, 4, 8, 16, 32],
    n_trials=100,
    timeout=3600,
    n_jobs=-1,
)
```

---

## Outputs

### Model Artifacts

For each `{gauge_id}_{dataset}` combination:
```
data/optimization/rfr_results/{gauge_id}/{gauge_id}_{dataset}/
├── best_model.joblib           # Trained RandomForestRegressor
├── optimization_study.pkl      # Optuna study (Pareto front)
├── best_parameters.json        # Hyperparameters
├── validation_metrics.json     # KGE, NSE, PBIAS, etc.
└── pareto_front.csv           # All non-dominated solutions
```

### Metrics

**Validation metrics** (JSON):
```json
{
  "KGE": 0.85,
  "NSE": 0.80,
  "PBIAS": -5.2,
  "RMSE": 1.23,
  "MAE": 0.87,
  "n_features_dynamic": 43,
  "n_features_static": 17,
  "n_features_total": 60,
  "latitude": 55.123,
  "n_train_samples": 4018,
  "n_val_samples": 730
}
```

---

## Ungauged Basin Prediction Strategy

### Spatial Regionalization Approach

1. **Train gauge-specific models** with full feature set (current implementation)
2. **Extract spatial patterns**:
   - Feature importance analysis per gauge
   - Cluster catchments by static attributes (hierarchical/k-means)
   - Identify homogeneous regions
3. **Regional transfer learning**:
   - **Option A**: Train one model per region using pooled data
   - **Option B**: Weighted ensemble of donor gauges (nearest neighbors in attribute space)
   - **Option C**: Meta-learning (train model to predict parameters)
4. **Prediction for ungauged basins**:
   - Extract static attributes from HydroATLAS/GIS
   - Assign to region or find donor gauges
   - Apply regional model or ensemble

### Key Advantages

- **Physical consistency**: Static attributes encode runoff generation mechanisms
- **Data efficiency**: Leverage multi-site information
- **Scalability**: Predict anywhere with available static data
- **Interpretability**: Feature importance reveals controlling factors

### Validation

- **Spatial leave-one-out**: Hold out entire catchments (not just temporal data)
- **Geographic stratification**: Test on basins distant from training gauges
- **Metrics**: KGE, NSE, PBIAS (same as gauged basins)

---

## Comparison with Conceptual Models

| Aspect | RFR (Enhanced) | GR4J/HBV |
|--------|----------------|----------|
| **Calibration** | Automatic (Optuna) | Manual/Optuna required |
| **Parameters** | 60+ features, auto-tuned | 4–6 fixed parameters |
| **Spatial transfer** | Static attributes enable regionalization | No spatial information |
| **Interpretability** | Feature importance | Physical equations |
| **PET** | Computed explicitly | Implicit in parameters |
| **Computation** | O(n × trees × features) | O(n × timesteps) |
| **Overfitting risk** | Regularization needed | Low (few parameters) |

**Trade-offs**:
- **RFR**: Better accuracy, data-hungry, less physically interpretable
- **Conceptual**: Physically grounded, parameter-sparse, manual tuning tedious

---

## References

1. **Oudin et al. (2005)**: PET formula  
   *Journal of Hydrology*, 311(1-4), 299-311.
2. **HydroATLAS**: Global watershed attributes  
   https://www.hydrosheds.org/hydroatlas
3. **Scikit-learn**: Machine learning library  
   https://scikit-learn.org/
4. **Optuna**: Hyperparameter optimization framework  
   https://optuna.org/

---

## Authors & Acknowledgments

**Implementation**: Dissertation project on hydrological modeling

**Data sources**:
- Discharge: Russian observational network
- Meteorology: ERA5-Land, GPCP, MSWEP
- Static attributes: HydroATLAS, CAMELS-style dataset

**License**: Academic use only

---

## Next Steps

- [ ] Run optimization for all gauges (parallel execution)
- [ ] Analyze feature importance across catchments
- [ ] Implement spatial regionalization (clustering + transfer learning)
- [ ] Validate on held-out basins (spatial cross-validation)
- [ ] Compare with GR4J/HBV benchmarks
- [ ] Generate publication-quality plots (feature importance, spatial maps)
