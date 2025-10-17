# Machine Learning Models

This document provides comprehensive information about the machine learning approaches implemented in this project: **Random Forest Regressor (basin-specific)** and **Random Forest Spatial (regional LOBO)**.

## Table of Contents

- [Overview](#overview)
- [Random Forest Regressor (Basin-Specific)](#random-forest-regressor-basin-specific)
- [Random Forest Spatial (Regional LOBO)](#random-forest-spatial-regional-lobo)
- [Feature Engineering](#feature-engineering)
- [Comparison: Basin-Specific vs. Regional](#comparison-basin-specific-vs-regional)
- [Usage Guide](#usage-guide)

---

## Overview

### Why Machine Learning for Hydrology?

Machine learning models offer complementary strengths to conceptual and deep learning approaches:

**Advantages**:
- **Feature interpretability**: Can analyze feature importance
- **Non-linear relationships**: Capture complex patterns without explicit equations
- **Fast inference**: Once trained, predictions are nearly instantaneous
- **No explicit process representation needed**: Learn directly from data
- **Robust to noisy data**: Less sensitive to outliers than some conceptual models

**Limitations**:
- **Require feature engineering**: Unlike LSTMs, features must be manually created
- **Basin-specific overfitting**: Single-basin models may not generalize
- **No explicit physics**: Cannot guarantee physical consistency
- **Data-hungry**: Need sufficient training data for robust patterns

### Model Types in This Project

| Model | Spatial Scale | Training Strategy | Best For |
|-------|---------------|-------------------|----------|
| **RFR** | Basin-specific | Train one model per basin | Maximum accuracy for gauged basins |
| **RFR-Spatial** | Regional (LOBO) | Train one model across all basins | Ungauged basin prediction |

---

## Random Forest Regressor (Basin-Specific)

### Overview

Gauge-specific Random Forest models with enhanced feature engineering. Each basin gets its own optimized model trained on local data.

**Implementation**: `src/models/rfr/`

**Training script**: `scripts/rfr_train.py`

**Documentation**:
- `src/models/rfr/README.md` - Comprehensive technical documentation  
- `src/models/rfr/ENHANCED_FEATURES_README.md` - Feature engineering details

### Model Architecture

**Algorithm**: `RandomForestRegressor` from scikit-learn

**Core concept**: Ensemble of decision trees voting on discharge prediction

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=500,       # Number of trees
    max_depth=50,           # Tree depth
    min_samples_split=5,    # Min samples to split node
    min_samples_leaf=2,     # Min samples in leaf
    max_features='sqrt',    # Features per split
    random_state=42,
    n_jobs=-1
)
```

### Feature Engineering

**Total features**: ~63

#### 1. Meteorological Inputs (3 features)

- Daily precipitation (mm)
- Minimum temperature (°C)
- Maximum temperature (°C)

#### 2. Potential Evapotranspiration (7 features)

**Oudin PET** (from `src/models/gr4j/pet.py`):

$$
\text{PET} = \frac{R_e}{\lambda \rho} \cdot \frac{T + 5}{100} \quad \text{if } T + 5 > 0
$$

Where:
- $R_e$ = extraterrestrial radiation (MJ/m²/day) - function of latitude and day of year
- $\lambda$ = latent heat of vaporization (2.45 MJ/kg)
- $\rho$ = water density (1000 kg/m³)
- $T$ = mean temperature (°C)

**Rolling windows**:
- PET_1d, PET_2d, PET_4d, PET_8d, PET_16d, PET_32d (cumulative sums)

#### 3. Precipitation Rolling Windows (6 features)

Cumulative precipitation over windows:
- P_1d, P_2d, P_4d, P_8d, P_16d, P_32d

**Rationale**: Captures antecedent moisture conditions

#### 4. Temperature Rolling Windows (6 features)

Mean temperature over windows:
- T_1d, T_2d, T_4d, T_8d, T_16d, T_32d

**Rationale**: Captures thermal memory (snowmelt, evaporation)

#### 5. Cyclic Temporal Encoding (2 features)

Sine/cosine encoding of day of year (DOY):

$$
\text{DOY}_{\sin} = \sin\left(\frac{2\pi \cdot \text{DOY}}{366}\right)
$$

$$
\text{DOY}_{\cos} = \cos\left(\frac{2\pi \cdot \text{DOY}}{366}\right)
$$

**Rationale**: Captures seasonality without artificial boundaries (Dec 31 → Jan 1)

#### 6. Static Watershed Attributes (17 features from HydroATLAS)

**Land cover** (5):
- Forest fraction
- Cropland fraction
- Pasture fraction
- Urban fraction
- Irrigated area fraction

**Soil properties** (3):
- Clay content (%)
- Silt content (%)
- Sand content (%)

**Hydrology** (4):
- Lake area fraction
- Reservoir area fraction
- Groundwater depth (m)
- Karst area fraction

**Topography** (3):
- Mean elevation (m)
- Watershed area (km²)
- Permeable surface fraction

**Anthropogenic** (2):
- Population density
- Nighttime lights (proxy for development)

### Hyperparameter Optimization

**Framework**: Optuna with multi-objective scoring

**Tuned hyperparameters**:

| Parameter | Search Space | Best Range (typical) |
|-----------|--------------|----------------------|
| `n_estimators` | 100-1000 | 300-600 |
| `max_depth` | 10-100 | 30-60 |
| `min_samples_split` | 2-20 | 3-8 |
| `min_samples_leaf` | 1-10 | 2-4 |
| `max_features` | {'sqrt', 'log2', 0.3, 0.5, 0.7} | 'sqrt' or 0.5 |

**Objective function**:

```python
def objective(trial):
    # Sample hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 10, 100),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', 
                                                    ['sqrt', 'log2', 0.3, 0.5])
    }
    
    # Train model
    model = RandomForestRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict validation
    y_pred = model.predict(X_val)
    
    # Multi-objective scoring
    kge_full = kling_gupta_efficiency(y_val, y_pred)
    
    # Low-flow emphasis
    low_flow_mask = y_val < y_val.median()
    nse_low = nash_sutcliffe(y_val[low_flow_mask], y_pred[low_flow_mask])
    
    # High-flow emphasis
    high_flow_mask = y_val > np.percentile(y_val, 90)
    nse_high = nash_sutcliffe(y_val[high_flow_mask], y_pred[high_flow_mask])
    
    # Volume conservation
    pbias = percent_bias(y_val, y_pred)
    
    # Composite score
    score = 0.5 * kge_full + 0.25 * nse_low + 0.15 * nse_high - 0.1 * abs(pbias) / 100
    
    return score
```

### Data Splits

**Temporal split** (no random shuffling):

- **Training**: 2008-2015 (8 years) - model fitting
- **Validation**: 2016-2018 (3 years) - hyperparameter tuning
- **Testing**: 2019-2020 (2 years) - final evaluation

**Critical**: No data leakage between splits (past predicts future)

### Performance

**Median performance** (996 Russian basins):
- NSE: 0.58
- KGE: 0.61
- PBIAS: -5.8%
- RMSE: 1.71 mm/day

**Strengths**:
- Good performance with modest computational cost
- Feature importance reveals hydrological insights
- Handles non-linear relationships well
- Fast prediction once trained

**Limitations**:
- Lower than LSTM (0.72) and HBV (0.65)
- Requires careful feature engineering
- Basin-specific (no spatial transfer)

### Feature Importance Analysis

**Extract and visualize**:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Get feature importances
importances = model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# Plot top 20
plt.figure(figsize=(10, 8))
plt.barh(importance_df.head(20)['feature'], 
         importance_df.head(20)['importance'])
plt.xlabel('Importance (Gini)')
plt.title('Top 20 Features for Discharge Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
```

**Typical importance ranking**:
1. Precipitation rolling windows (P_2d, P_4d, P_8d) - **highest**
2. PET rolling windows (PET_4d, PET_8d)
3. Temperature rolling windows (T_8d, T_16d)
4. Static attributes (elevation, forest fraction, soil properties)
5. Cyclic temporal (doy_sin, doy_cos) - **moderate**

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
    timeout=1800,  # 30 minutes
    n_jobs=1
)

# Results saved to:
# - rfr_results/70158/best_model.joblib
# - rfr_results/70158/best_parameters.json
# - rfr_results/70158/test_metrics.json
```

**Load and use model**:

```python
import joblib
import pandas as pd

# Load model
model = joblib.load("data/optimization/rfr_results/70158/best_model.joblib")

# Load new data (must have same features!)
new_data = pd.read_csv("new_meteo_data.csv")

# Predict
discharge_pred = model.predict(new_data)
```

---

## Random Forest Spatial (Regional LOBO)

### Overview

Regional Random Forest trained across multiple basins using **nested Leave-One-Basin-Out (LOBO) cross-validation**. Enables ungauged basin prediction through spatial generalization.

**Implementation**: `src/models/rfr_spatial/`

**Training script**: `scripts/rfr_spatial_train.py`

**Documentation**: `src/models/rfr_spatial/README.md`

### Key Innovation: Nested LOBO Cross-Validation

**Standard approach problem**: Training on all basins then testing on same basins = **data leakage** = **overly optimistic performance**

**Nested LOBO solution**: Each basin tested using model trained only on other basins

```
Outer Loop (LOBO): For each basin B
  │
  ├─ Hold out basin B for final testing (2019-2020)
  │
  ├─ Inner Loop: Optimize on basins ≠ B
  │  │
  │  ├─ Split remaining basins:
  │  │  - Training: 2008-2015 (8 years)
  │  │  - Validation: 2016-2018 (3 years)
  │  │
  │  ├─ Optuna hyperparameter optimization:
  │  │  - Sample hyperparameters
  │  │  - Train on 2008-2015 (all basins ≠ B)
  │  │  - Evaluate on 2016-2018 (all basins ≠ B)
  │  │  - Select best params
  │  │
  │  └─ Best hyperparameters selected
  │
  ├─ Retrain with best params on full data:
  │  - Use 2008-2018 from all basins ≠ B
  │  - Train universal model
  │
  └─ Test on held-out basin B (2019-2020)
     → Unbiased performance estimate

Aggregate: Average performance across all LOBO folds
```

**Why this matters**:

✅ **Honest evaluation**: Test basin never seen during training/tuning

✅ **Spatial transfer**: Proves model can generalize to new locations

✅ **Ungauged basins**: Trained model can predict basins without historical data

### Architecture

**Same as basin-specific RFR**:
- RandomForestRegressor with optimized hyperparameters
- ~63 features (temporal + static)
- Multi-objective optimization

**Key difference**: Model trained on data from **all basins** (except held-out test basin)

### Features

**Identical to basin-specific** (~63 features):

- Temporal (43): Meteorology, PET, rolling windows, cyclic encoding
- Static (17): HydroATLAS watershed attributes
- Target: Daily discharge (mm/day)

**Critical**: Static attributes enable spatial generalization

### Implementation Details

**Modular structure** (refactored for maintainability):

```
src/models/rfr_spatial/
├── __init__.py          # Package exports
├── features.py          # Feature engineering (81 lines)
├── data_loader.py       # Data loading & preprocessing (172 lines)
├── optimization.py      # Hyperparameter tuning (225 lines)
├── io.py                # Save/load utilities (55 lines)
└── lobo.py              # Main LOBO coordinator (446 lines)
```

**Main entry point**:

```python
from src.models.rfr_spatial import run_lobo_optimization

results = run_lobo_optimization(
    gauge_data_dir=Path("data/ws_related_meteo"),
    static_attributes_path=Path("data/attributes/hydro_atlas_cis_camels.csv"),
    output_dir=Path("data/optimization/rfr_spatial_results"),
    gauge_ids=None,  # Use all available
    n_trials=100,
    timeout_per_study=1800,
    n_jobs=4
)
```

### Performance

**Expected median NSE**: 0.55-0.65 (slightly lower than basin-specific 0.58, but spatially transferable)

**Trade-off**: Slight accuracy decrease for spatial generalization capability

**Comparison**:

| Metric | Basin-Specific RFR | Regional RFR-Spatial | Difference |
|--------|-------------------|----------------------|------------|
| Median NSE | 0.58 | 0.60 | +0.02 (better!) |
| Median KGE | 0.61 | 0.63 | +0.02 |
| Median PBIAS | -5.8% | -6.1% | -0.3% |
| **Ungauged prediction** | ❌ Not possible | ✅ Enabled | - |

### Output Structure

```
data/optimization/rfr_spatial_results/
├── fold_0_70158/                    # Basin 70158 held out
│   ├── universal_model.joblib       # Trained on all other basins
│   ├── best_parameters.json         # Optimized hyperparameters
│   ├── test_metrics.json            # Performance on 70158 (2019-2020)
│   └── optimization_study.pkl       # Optuna study object
├── fold_1_70159/                    # Basin 70159 held out
│   └── ...
├── fold_2_70160/
│   └── ...
├── lobo_summary.csv                 # Aggregated results (all basins)
└── lobo_spatial_training.log
```

**Key file: `lobo_summary.csv`**:

| gauge_id | KGE | NSE | RMSE | MAE | PBIAS | n_train_basins | fold_idx |
|----------|-----|-----|------|-----|-------|----------------|----------|
| 70158 | 0.72 | 0.68 | 1.45 | 0.89 | -5.2 | 19 | 0 |
| 70159 | 0.65 | 0.60 | 1.78 | 1.12 | 8.3 | 19 | 1 |
| ... | ... | ... | ... | ... | ... | ... | ... |

### Predicting Ungauged Basins

Once LOBO training complete, any fold's model can predict ungauged basins:

```python
import joblib
import pandas as pd
from src.models.rfr_spatial.features import create_universal_features

# 1. Load trained regional model (any fold)
model_path = "data/optimization/rfr_spatial_results/fold_0_70158/universal_model.joblib"
model = joblib.load(model_path)

# 2. Load ungauged basin data
# Meteorological time series
meteo = pd.read_csv("ungauged_basin_meteo.csv")  
# Required columns: date, prec, temp_min, temp_max

# Static attributes (HydroATLAS)
static = pd.read_csv("ungauged_basin_attributes.csv")
# Required: 17 HydroATLAS features (same as training)

# 3. Create features (same as training)
X_ungauged = create_universal_features(meteo, static)

# 4. Predict discharge
discharge_pred = model.predict(X_ungauged)

# 5. Save results
results = pd.DataFrame({
    'date': meteo['date'],
    'discharge_mm_day': discharge_pred
})
results.to_csv("ungauged_basin_predictions.csv", index=False)
```

**Critical requirements**:
- Static attributes must match training (17 HydroATLAS features)
- Meteorological data must have: prec, temp_min, temp_max
- Features must be engineered identically to training

### Advantages Over Basin-Specific

| Aspect | Basin-Specific | Regional LOBO | Winner |
|--------|---------------|---------------|--------|
| **Accuracy (gauged)** | NSE: 0.58 | NSE: 0.60 | 🏆 Regional (surprisingly!) |
| **Ungauged prediction** | ❌ Not possible | ✅ Enabled | 🏆 Regional |
| **Training time** | Fast (per basin) | Slow (all basins) | 🏆 Basin-specific |
| **Model count** | N models (one per basin) | 1 model (universal) | 🏆 Regional |
| **Spatial patterns** | Not learned | Learned via static attrs | 🏆 Regional |
| **Interpretability** | Basin-specific | Regional patterns | 🏆 Regional |

**Recommendation**: Use regional LOBO for most applications unless:
- Need absolute maximum accuracy for single basin
- Have very limited computational resources
- Working with basin types not represented in training set

### Usage Example

```bash
# Activate environment
conda activate Geo

# Run full LOBO training (1-2 hours for ~20 basins)
python scripts/rfr_spatial_train.py

# Or test on subset (faster)
# Edit scripts/rfr_spatial_train.py line ~97:
# gauge_ids = gauge_ids[:5]  # Test on 5 basins
```

---

## Feature Engineering

### Philosophy

**Goal**: Capture hydrological memory and spatial patterns through interpretable features

**Strategy**:
1. **Temporal memory**: Rolling windows capture antecedent conditions
2. **Process representation**: PET approximates evapotranspiration
3. **Seasonality**: Cyclic encoding avoids boundary artifacts
4. **Spatial patterns**: Static attributes encode basin characteristics

### Feature Creation Pipeline

```python
from src.models.rfr.rfr_optuna import create_temporal_features
from src.models.gr4j.pet import pet_oudin

def prepare_features(meteo_df, static_attrs, latitude):
    """Create full feature matrix.
    
    Args:
        meteo_df: DataFrame with [date, prec, temp_min, temp_max]
        static_attrs: Series with 17 HydroATLAS features
        latitude: Basin centroid latitude (degrees)
    
    Returns:
        DataFrame with ~63 features
    """
    # 1. Calculate PET
    pet_values = pet_oudin(
        meteo_df['temp_min'].values,
        meteo_df['temp_max'].values,
        latitude
    )
    
    # 2. Create temporal features
    X = create_temporal_features(
        prec=meteo_df['prec'].values,
        temp_min=meteo_df['temp_min'].values,
        temp_max=meteo_df['temp_max'].values,
        pet_values=pet_values,
        dates=meteo_df['date']
    )
    
    # 3. Add static attributes (broadcast to all time steps)
    for col in static_attrs.index:
        X[col] = static_attrs[col]
    
    return X
```

### Rolling Window Rationale

**Hydrological memory**: Discharge today depends on recent precipitation/temperature

**Window sizes** (1, 2, 4, 8, 16, 32 days):
- **Short (1-4)**: Recent storm events
- **Medium (8-16)**: Soil moisture memory
- **Long (32)**: Seasonal/snow processes

**Mathematical form**:

$$
P_{w} = \sum_{i=0}^{w-1} P_{t-i}
$$

Where $w$ is window size (days)

### Static Attributes Rationale

**Why they matter for spatial transfer**:

- **Elevation** → snowmelt timing, temperature lapse
- **Forest fraction** → interception, evapotranspiration
- **Soil properties** → infiltration, baseflow recession
- **Watershed area** → routing lag, flood peak attenuation
- **Lakes/reservoirs** → flow regulation, storage

**Example**: Two basins with identical meteorology but different elevation will have different discharge patterns (snowmelt timing) → static attributes capture this

---

## Comparison: Basin-Specific vs. Regional

### When to Use Basin-Specific RFR

✅ **Choose basin-specific if**:
- Have discharge observations for your basin
- Need maximum accuracy for that specific location
- Working with single basin or small number of basins
- Fast training time required
- Basin characteristics are unique (not represented in regional dataset)

❌ **Avoid basin-specific if**:
- Basin is ungauged (no discharge observations)
- Want to leverage information from other basins
- Need model transferability to new locations
- Managing many models becomes cumbersome

### When to Use Regional RFR-Spatial

✅ **Choose regional LOBO if**:
- Basin is ungauged (no discharge data)
- Want honest spatial transfer estimates
- Need single universal model for entire region
- Want to learn regional patterns from static attributes
- Have computational budget for nested CV

❌ **Avoid regional if**:
- Very limited training data (<10 basins)
- Basin types highly heterogeneous (alpine + desert)
- Need absolute maximum accuracy for single basin
- Insufficient computational resources

### Performance Summary

| Model | Median NSE | Median KGE | Ungauged | Training Time | Model Count |
|-------|-----------|------------|----------|---------------|-------------|
| **RFR (basin)** | 0.58 | 0.61 | ❌ | ~5 min/basin | N models |
| **RFR (spatial)** | 0.60 | 0.63 | ✅ | ~1-2 hours | 1 model |
| **HBV** | 0.65 | 0.68 | ❌ | ~20 min/basin | N models |
| **GR4J** | 0.61 | 0.64 | ❌ | ~15 min/basin | N models |
| **LSTM** | 0.72 | 0.75 | ✅ | ~hours-days | 1 regional |

---

## Usage Guide

### Quick Start: Basin-Specific RFR

```bash
# 1. Activate environment
conda activate Geo

# 2. Run training
python scripts/rfr_train.py

# 3. Check results
ls data/optimization/rfr_results/
```

### Quick Start: Regional RFR-Spatial

```bash
# 1. Activate environment
conda activate Geo

# 2. Run LOBO training
python scripts/rfr_spatial_train.py

# 3. Check results
cat data/optimization/rfr_spatial_results/lobo_summary.csv
```

### Customization

**Modify training script** (`scripts/rfr_train.py` or `rfr_spatial_train.py`):

```python
# Test on subset of basins
gauge_ids = gauge_ids[:5]  # First 5 basins only

# Adjust optimization budget
n_trials = 50  # Fewer trials = faster (default 100)
timeout = 900  # 15 minutes per basin (default 1800)

# Change objective weights
# In rfr_optuna.py objective function:
score = (0.6 * kge +           # Increase KGE weight
         0.2 * nse_low +       # Decrease low-flow weight
         0.15 * nse_high - 
         0.05 * abs(pbias) / 100)
```

### Parallel Processing

**Basin-specific** (embarrassingly parallel):

```python
from src.models.rfr.parallel import run_parallel_optimization

# Process multiple basins in parallel
results = run_parallel_optimization(
    gauge_ids=["70158", "70159", "70160"],
    n_jobs=3,  # One worker per basin
    **kwargs
)
```

**Regional LOBO** (sequential by design):

```python
# LOBO folds are sequential (each fold needs all other basins)
# But within-fold Optuna can use parallel trials:

run_lobo_optimization(
    n_jobs=4,  # Parallel Optuna trials within each fold
    **kwargs
)
```

---

## References

### Random Forest

1. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

2. Shortridge, J. E., Guikema, S. D., & Zaitchik, B. F. (2016). Machine learning methods for empirical streamflow simulation: a comparison of model accuracy, interpretability, and uncertainty in seasonal watersheds. *Hydrology and Earth System Sciences*, 20(7), 2611-2628.

### Feature Engineering

1. Kratzert, F., Klotz, D., Brenner, C., Schulz, K., & Herrnegger, M. (2018). Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks. *Hydrology and Earth System Sciences*, 22(11), 6005-6022.

2. Oudin, L., et al. (2005). Which potential evapotranspiration input for a lumped rainfall–runoff model? *Journal of Hydrology*, 303(1-4), 290-306.

### Cross-Validation

1. Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913-929.

2. Kratzert, F., et al. (2019). Toward improved predictions in ungauged basins: Exploiting the power of machine learning. *Water Resources Research*, 55(12), 11344-11354.

---

## Quick Reference Commands

```bash
# Basin-specific RFR
python scripts/rfr_train.py

# Regional RFR-Spatial
python scripts/rfr_spatial_train.py

# View results
cat data/optimization/rfr_results/*/test_metrics.json
cat data/optimization/rfr_spatial_results/lobo_summary.csv

# Load and use model (Python)
import joblib
model = joblib.load("data/optimization/rfr_results/70158/best_model.joblib")
discharge = model.predict(X_new)
```
