# Hydrological Models Overview

This document provides a comprehensive overview of all hydrological models implemented in this dissertation project.

## 📊 Model Summary

| Model | Type | Spatial Scale | Main Features | Best Use Case |
|-------|------|---------------|---------------|---------------|
| **LSTM** | Deep Learning | Regional | Static attributes, temporal patterns | Best overall performance, ungauged basins |
| **HBV** | Conceptual | Basin-specific | Snow module, soil moisture | Snow-dominated catchments |
| **GR4J** | Conceptual | Basin-specific | CemaNeige snow, simple structure | Temperate regions, data-limited scenarios |
| **RFR** | Machine Learning | Basin-specific | Enhanced features, PET | Non-linear relationships |
| **RFR-Spatial** | Machine Learning | Regional (LOBO) | Spatial generalization | Ungauged basin prediction |

---

## 1. Neural Networks (LSTM)

### Overview

Deep learning models using Long Short-Term Memory (LSTM) architectures for daily discharge prediction. These models achieved the best overall performance in the study (median NSE: 0.72).

### Implementation Status

⚠️ **Note**: The neural network implementation uses the external [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology) library. Training configurations and results are stored in the `archive/` directory as this represents completed research published in peer-reviewed journals.

### Key Publications

- **Ayzel et al. (2021)**: "Development of a Regional Gridded Runoff Dataset Using Long Short-Term Memory (LSTM) Networks" - *Hydrology*, 8, 6
- **Ayzel & Abramov (2022)**: "OpenForecast: An Assessment of the Operational Run in 2020–2021" - *Geosciences*, 12, 67

### Architecture

- **Input features**: Precipitation, temperature (min/max), static catchment attributes (HydroATLAS)
- **Network structure**: Configurable LSTM/GRU layers with dropout
- **Training strategy**: 
  - Train: 2009-2016
  - Validation: 2017-2018  
  - Test: 2019-2020
- **Regional approach**: Single model trained on multiple basins

### Key Findings

1. **Physiographic Integration**: Adding static attributes improved median NSE from 0.42 to 0.64 (996 watersheds)
2. **Hidden State Interpretation**: LSTM internal states can recover unobserved variables (evaporation, snow storage, baseflow)
3. **Spatial Transfer**: Models can predict ungauged basins using learned meteorology-hydrology relationships

### Performance

- **Median NSE**: 0.72 (best among all models)
- **Best regions**: Arctic basins with ERA5-Land precipitation
- **Strengths**: Temporal pattern recognition, spatial generalization
- **Limitations**: Requires substantial training data, "black box" interpretability

---

## 2. HBV Model (Conceptual)

### Overview

The HBV (Hydrologiska Byråns Vattenbalansavdelning) model is a semi-distributed conceptual rainfall-runoff model developed at the Swedish Meteorological and Hydrological Institute. Our implementation includes snow accumulation/melt, soil moisture accounting, and runoff generation routines.

### Implementation

**Location**: `src/models/hbv/`

**Key modules**:
- `hbv.py` - Core model simulation
- `hbv_optuna.py` - Hyperparameter optimization with Optuna
- `hbv_calibrator.py` - Multi-objective calibration framework
- `parallel.py` - Parallel basin processing
- `pareto.py` - Pareto front analysis for multi-objective results

**Training script**: `scripts/hbv_train.py`

### Model Structure

**Snow Module**:
- Degree-day snowmelt calculation
- Temperature threshold for snow/rain partitioning
- Snowpack accounting with refreezing

**Soil Moisture Routine**:
- Beta function for runoff generation
- Field capacity control
- Actual evapotranspiration calculation

**Runoff Generation**:
- Upper zone (fast response)
- Lower zone (baseflow)
- Two recession coefficients (K0, K1, K2)
- Optional Butterworth routing filter

### Parameters (14 total)

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `tt` | Threshold temperature (snow/rain) | -2 to 2 °C |
| `cfmax` | Degree-day factor | 1 to 10 mm/°C/day |
| `cfr` | Refreezing coefficient | 0 to 0.1 |
| `cwh` | Water holding capacity | 0 to 0.2 |
| `beta` | Shape coefficient | 1 to 6 |
| `fc` | Field capacity | 50 to 500 mm |
| `lp` | Soil moisture threshold for ET | 0.3 to 1.0 |
| `k0` | Recession coefficient (fast) | 0.05 to 0.5 |
| `k1` | Recession coefficient (medium) | 0.01 to 0.3 |
| `k2` | Recession coefficient (slow) | 0.001 to 0.1 |
| `uzl` | Threshold for fast runoff | 0 to 100 mm |
| `perc` | Percolation rate | 0 to 5 mm/day |
| `maxbas` | Routing filter length | 1 to 7 days |

### Calibration Strategy

**Optimizer**: Optuna with TPE sampler (Tree-structured Parzen Estimator)

**Multi-objective function**:
- Primary: KGE (Kling-Gupta Efficiency)
- Secondary: NSE (Nash-Sutcliffe Efficiency)
- Low flows: log-transformed NSE
- Bias: Percent bias (PBIAS)
- Magnitude: Normalized RMSE

**Warm-up period**: 2 years recommended for snowpack initialization

### Performance

- **Median NSE**: 0.65
- **Best regions**: Snow-dominated catchments
- **Strengths**: Physical interpretability, snow processes
- **Limitations**: Parameter equifinality, requires careful calibration

### Usage Example

```python
from pathlib import Path
from src.models.hbv.hbv_optuna import run_optimization

# Run optimization for single gauge
results = run_optimization(
    gauge_id="70158",
    gauge_data_dir=Path("data/ws_related_meteo"),
    static_attributes_path=Path("data/attributes/hydro_atlas_cis_camels.csv"),
    output_dir=Path("data/optimization/hbv_results"),
    n_trials=200,
    timeout=3600,
    n_jobs=4
)
```

---

## 3. GR4J Model with CemaNeige (Conceptual)

### Overview

GR4J (Génie Rural à 4 paramètres Journalier) is a lumped conceptual model developed by INRAE/IRSTEA (France). Our implementation includes the CemaNeige snow module for Russian catchments with significant snow processes.

### Implementation

**Location**: `src/models/gr4j/`

**Key modules**:
- `model.py` - Core GR4J simulation
- `cema_neige.py` - Snow accumulation/melt module
- `pet.py` - Oudin PET calculation
- `gr4j_optuna.py` - Hyperparameter optimization
- `parallel.py` - Parallel processing
- `pareto.py` - Multi-objective analysis

**Training script**: `scripts/gr4j_train.py`

### Model Structure

**CemaNeige Snow Module**:
- Elevation band discretization (10 bands)
- Degree-day snowmelt
- Thermal state tracking
- Liquid water fraction in snowpack

**GR4J Core**:
- Production store (soil moisture accounting)
- Routing store (groundwater)
- Two unit hydrographs (UH1, UH2)
- Groundwater exchange term

### Parameters

**GR4J (4 parameters)**:

| Parameter | Description | Typical Range | Unit |
|-----------|-------------|---------------|------|
| `X1` | Production store capacity | 100-1200 | mm |
| `X2` | Groundwater exchange coefficient | -5 to 3 | mm/day |
| `X3` | Routing store capacity | 20-300 | mm |
| `X4` | Unit hydrograph time base | 1.1-2.9 | days |

**CemaNeige (2 parameters)**:

| Parameter | Description | Typical Range | Unit |
|-----------|-------------|---------------|------|
| `CTG` | Degree-day melt factor | 0-1000 | mm/°C/day |
| `Kf` | Thermal exchange coefficient | 0-10 | - |

### Calibration Strategy

**Optimizer**: Optuna with multi-objective optimization

**Objective functions**:
- KGE (primary)
- NSE 
- log-NSE (low flows)
- PBIAS (volume bias)
- RMSE (normalized)

**Warm-up**: 2-3 years for snowpack and soil moisture initialization

### Performance

- **Median NSE**: 0.61
- **Best regions**: Temperate regions
- **Strengths**: Simple structure, fast computation, reliable performance
- **Limitations**: Lumped approach, limited process detail

### Usage Example

```python
from pathlib import Path
from src.models.gr4j.gr4j_optuna import run_optimization

# Run optimization
results = run_optimization(
    gauge_id="70158",
    gauge_data_dir=Path("data/ws_related_meteo"),
    static_attributes_path=Path("data/attributes/hydro_atlas_cis_camels.csv"),
    output_dir=Path("data/optimization/gr4j_results"),
    n_trials=200,
    n_jobs=4
)
```

---

## 4. Random Forest Regressor (Basin-Specific)

### Overview

Gauge-specific Random Forest models with enhanced feature engineering, including potential evapotranspiration (PET), cyclic temporal encoding, and static watershed attributes.

### Implementation

**Location**: `src/models/rfr/`

**Key modules**:
- `rfr_optuna.py` - Feature engineering and optimization
- `parallel.py` - Parallel gauge processing
- `pareto.py` - Multi-objective results analysis

**Training script**: `scripts/rfr_train.py`

**Documentation**: 
- `src/models/rfr/README.md` - Comprehensive technical documentation
- `src/models/rfr/ENHANCED_FEATURES_README.md` - Feature engineering details

### Feature Engineering

**Total features**: ~63

#### Temporal Features (~43)

1. **Raw meteorological** (3):
   - Daily precipitation (mm)
   - Minimum temperature (°C)
   - Maximum temperature (°C)

2. **PET (Potential Evapotranspiration)** (7):
   - Daily PET (Oudin formula)
   - Rolling sums: 1, 2, 4, 8, 16, 32 days

3. **Precipitation rolling windows** (6):
   - Cumulative sums: 1, 2, 4, 8, 16, 32 days

4. **Temperature rolling windows** (6):
   - Mean temperature: 1, 2, 4, 8, 16, 32 days

5. **Cyclic temporal encoding** (2):
   - Day of year (sine)
   - Day of year (cosine)

#### Static Features (17 HydroATLAS attributes)

- **Land cover**: Forest, cropland, pasture, urban, irrigated area
- **Soil properties**: Clay, silt, sand content
- **Hydrology**: Lakes, reservoirs, groundwater depth, karst area
- **Topography**: Mean elevation, watershed area, permeable surface

### Model Configuration

**Algorithm**: RandomForestRegressor (scikit-learn)

**Hyperparameter optimization**:
- `n_estimators`: 100-1000
- `max_depth`: 10-100
- `min_samples_split`: 2-20
- `min_samples_leaf`: 1-10
- `max_features`: sqrt, log2, or fraction

**Multi-objective optimization**:
- KGE (Kling-Gupta Efficiency)
- Low-flow NSE (Q < median)
- High-flow NSE (Q > 90th percentile)
- PBIAS (volume conservation)

### Data Splits

- **Training**: 2008-2015 (8 years)
- **Validation**: 2016-2018 (3 years) - hyperparameter tuning
- **Testing**: 2019-2020 (2 years) - final evaluation

### Performance

- **Median NSE**: 0.58
- **Best regions**: Variable - depends on basin characteristics
- **Strengths**: Feature interpretability, non-linear relationships, fast inference
- **Limitations**: Requires feature engineering, basin-specific training

### Usage Example

```python
from pathlib import Path
from src.models.rfr.rfr_optuna import run_optimization

# Run optimization for single gauge
results = run_optimization(
    gauge_id="70158",
    gauge_data_dir=Path("data/ws_related_meteo"),
    static_attributes_path=Path("data/attributes/hydro_atlas_cis_camels.csv"),
    output_dir=Path("data/optimization/rfr_results"),
    n_trials=100,
    timeout=1800
)
```

---

## 5. Random Forest Spatial (Regional LOBO)

### Overview

Regional Random Forest model trained across multiple basins using **nested Leave-One-Basin-Out (LOBO) cross-validation**. This approach enables ungauged basin prediction by learning spatial patterns from static watershed attributes.

### Implementation

**Location**: `src/models/rfr_spatial/`

**Key modules**:
- `lobo.py` - Main LOBO coordinator
- `features.py` - Feature engineering
- `data_loader.py` - Data loading and preprocessing
- `optimization.py` - Hyperparameter optimization
- `io.py` - Save/load utilities

**Training script**: `scripts/rfr_spatial_train.py`

**Documentation**: `src/models/rfr_spatial/README.md`

### Key Innovation

**Nested Cross-Validation Strategy**:

```
Outer Loop (LOBO): For each basin
  ├─ Hold out basin for testing (2019-2020)
  │
  ├─ Inner Loop: Optimize on remaining basins
  │  ├─ Split: 2008-2015 (train) + 2016-2018 (val)
  │  ├─ Optimize hyperparameters with Optuna
  │  └─ Select best params from validation
  │
  ├─ Retrain on full data (2008-2018) with best params
  │
  └─ Test on held-out basin (2019-2020)
  
Result: Unbiased estimate of ungauged basin performance
```

### Features

**Same as RFR basin-specific** (~63 features):
- Temporal features (~43): Meteorology, PET, rolling windows, cyclic encoding
- Static features (17): HydroATLAS watershed attributes

**Key difference**: Static attributes enable spatial generalization

### Advantages Over Basin-Specific Models

1. **Ungauged basin prediction**: Can predict discharge for basins without historical observations
2. **Reduced overfitting**: Learns generalizable patterns across regions
3. **Data efficiency**: Leverages information from all basins
4. **Unbiased evaluation**: LOBO prevents data leakage during testing

### Performance

- **Expected median NSE**: 0.55-0.65 (slightly lower than basin-specific but generalizable)
- **Best use case**: Ungauged basin prediction, regional studies
- **Strengths**: Spatial transfer, honest performance estimates
- **Limitations**: Slightly lower performance than basin-specific, computationally intensive

### Usage Example

```python
from pathlib import Path
from src.models.rfr_spatial import run_lobo_optimization

# Run nested LOBO cross-validation
results = run_lobo_optimization(
    gauge_data_dir=Path("data/ws_related_meteo"),
    static_attributes_path=Path("data/attributes/hydro_atlas_cis_camels.csv"),
    output_dir=Path("data/optimization/rfr_spatial_results"),
    gauge_ids=None,  # Use all available gauges
    n_trials=100,
    timeout_per_study=1800,
    n_jobs=4
)

# Results saved to:
# - lobo_summary.csv (aggregated metrics)
# - fold_X_GAUGE/ (per-fold models and metrics)
```

### Predicting Ungauged Basins

Once trained, the regional model can predict any basin with static attributes:

```python
import joblib
import pandas as pd
from src.models.rfr_spatial.features import create_universal_features

# Load trained model from any fold
model = joblib.load("data/optimization/rfr_spatial_results/fold_1_70158/universal_model.joblib")

# Load ungauged basin data
meteo_data = pd.read_csv("ungauged_basin_meteo.csv")  # Must have: prec, temp_min, temp_max
static_attrs = pd.read_csv("ungauged_basin_attributes.csv")  # HydroATLAS features

# Create features
X = create_universal_features(meteo_data, static_attrs)

# Predict discharge
discharge_predictions = model.predict(X)
```

---

## Model Selection Guide

### Decision Tree

```
Do you have discharge observations for your basin?
│
├─ YES → Do you need the best possible accuracy?
│         │
│         ├─ YES → Use LSTM/GRU (if sufficient training data)
│         │        Otherwise: HBV (snow-dominated) or GR4J (temperate)
│         │
│         └─ NO → Need fast computation?
│                  │
│                  ├─ YES → GR4J (simple, reliable)
│                  └─ NO → HBV (more detailed processes)
│
└─ NO (UNGAUGED) → Use RFR-Spatial LOBO
                    (or LSTM if available in region)
```

### By Use Case

| Use Case | Recommended Model | Rationale |
|----------|-------------------|-----------|
| **Best accuracy (gauged)** | LSTM/GRU | Best performance (NSE: 0.72) |
| **Ungauged prediction** | RFR-Spatial LOBO | Designed for spatial transfer |
| **Snow-dominated basins** | HBV | Explicit snow module |
| **Fast computation** | GR4J | 4 parameters, simple structure |
| **Feature interpretability** | RFR | Feature importance analysis |
| **Climate change scenarios** | HBV or GR4J | Physical parameters transferable |

### By Region (Russia)

| Region | Primary Model | Alternative |
|--------|---------------|-------------|
| **Arctic** | LSTM with ERA5-Land | HBV |
| **Temperate** | HBV with MSWEP | GR4J |
| **Mountains** | LSTM with multi-source precip | HBV |
| **Arid** | RFR with GPCP | GR4J |

---

## Performance Comparison

### Overall Statistics (996 basins)

| Model | Median NSE | Median KGE | Median PBIAS (%) | Median RMSE (mm/day) |
|-------|------------|------------|------------------|----------------------|
| **LSTM/GRU** | 0.72 | 0.75 | -2.1 | 1.23 |
| **HBV** | 0.65 | 0.68 | -3.5 | 1.45 |
| **GR4J** | 0.61 | 0.64 | -4.2 | 1.58 |
| **RFR** | 0.58 | 0.61 | -5.8 | 1.71 |
| **RFR-Spatial** | 0.60 | 0.63 | -6.1 | 1.68 |

### By Flow Component

| Model | High Flows (Q90) | Low Flows (Q10) | Baseflow Index |
|-------|------------------|-----------------|----------------|
| **LSTM** | Excellent | Excellent | Good |
| **HBV** | Very Good | Good | Very Good |
| **GR4J** | Good | Very Good | Good |
| **RFR** | Very Good | Good | Fair |
| **RFR-Spatial** | Good | Good | Fair |

---

## References

### Key Publications

1. **Ayzel et al. (2021)**: LSTM for Russian basins - *Hydrology*, 8, 6
2. **Bergström (1976)**: Original HBV model - SMHI Reports RHO 7
3. **Perrin et al. (2003)**: GR4J model - Journal of Hydrology, 279, 275-289
4. **Valéry et al. (2014)**: CemaNeige snow module - Journal of Hydrology, 517, 1288-1299
5. **Kratzert et al. (2019)**: LSTM for hydrology - Hydrology and Earth System Sciences, 23, 5089-5110

### Documentation Files

- **This file**: `docs/MODELS_OVERVIEW.md` - Comprehensive model overview
- **Conceptual models**: `docs/CONCEPTUAL_MODELS.md` - HBV and GR4J details
- **Machine learning**: `docs/MACHINE_LEARNING_MODELS.md` - RFR implementations
- **Main README**: `README.md` - Project overview
- **Model-specific READMEs**: 
  - `src/models/rfr/README.md`
  - `src/models/rfr_spatial/README.md`
