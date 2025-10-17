# Regional RFR for Ungauged Basin Prediction

## Overview

This module implements a **universal (spatial) Random Forest Regressor** for predicting streamflow in ungauged basins using **nested Leave-One-Basin-Out (LOBO) cross-validation**.

Unlike gauge-specific models that train one model per catchment, the regional approach trains a single model across multiple basins. This enables:

1. **Spatial generalization**: Learn patterns that transfer across catchments
2. **Ungauged prediction**: Predict streamflow where no observations exist
3. **Reduced overfitting**: Pool data from many basins for robust training
4. **Physical interpretation**: Identify which features control spatial variability

---

## Nested Cross-Validation Strategy

### Outer Loop: Leave-One-Basin-Out (LOBO)
- Each basin is held out once for testing (2019-2020)
- Remaining basins used for training and hyperparameter tuning
- Provides unbiased estimate of spatial generalization

### Inner Loop: Temporal Split
- **Training**: 2008-2015 (8 years)
- **Validation**: 2016-2018 (3 years) — used for hyperparameter selection
- **Test**: 2019-2020 (2 years) — held-out basin only

### Why Nested CV?
- Prevents **information leakage** from test basins during hyperparameter tuning
- Provides **honest assessment** of spatial transferability
- Standard approach in ML for spatial prediction problems

---

## Feature Engineering

### Temporal Features (Dynamic)
Computed per basin from meteorological data:

1. **Raw meteorological variables**: precipitation, T_min, T_max
2. **PET (Potential Evapotranspiration)**: Oudin (2005) formula using latitude
3. **Rolling aggregations** (1, 2, 4, 8, 16, 32 days):
   - Precipitation: cumulative sum (water input)
   - PET: cumulative sum (water loss)
   - Temperature: mean (thermal conditions)
4. **Cyclic temporal encoding**: day-of-year (sin/cos) to capture seasonality

**Total dynamic features**: ~43

### Static Features (Spatial)
Watershed attributes from HydroATLAS (17 features):

- **Land cover**: forest, cropland, pasture, urban, irrigated area
- **Soil properties**: clay, silt, sand content
- **Hydrology**: lakes, reservoirs, groundwater depth, karst
- **Topography**: elevation, watershed area, permeable surface

These features enable the model to learn which catchment characteristics influence hydrological response.

**Total features per timestep**: ~60 (43 dynamic + 17 static)

---

## Implementation Details

### Data Structure
Each gauge contributes a time series matrix:
```
Shape: (n_timesteps, 60_features)
- Rows: daily timesteps (2008-2020)
- Columns: 43 temporal + 17 static features
- Target: q_mm_day (discharge in mm/day)
```

For training, data from multiple basins are **stacked vertically**:
```
Train data shape: (n_basins × n_timesteps, 60_features)
Example: 20 basins × 2922 days (2008-2015) = (58,440, 60)
```

### Hyperparameter Search
Same multi-objective optimization as gauge-specific model:

1. **KGE** (Kling-Gupta Efficiency): overall performance
2. **Composite low-flow**: log-NSE + inverse NSE
3. **Composite high-flow**: NSE + peak flow error
4. **PBIAS**: volume conservation

Uses Optuna's TPE sampler for efficient Pareto frontier exploration.

### Model Training
1. **Inner optimization**: Train on pooled data from training basins (2008-2015), validate on 2016-2018
2. **Select best hyperparameters**: Maximize weighted sum of objectives
3. **Final model**: Retrain on full data (2008-2018) from training basins
4. **Test**: Predict on held-out basin (2019-2020)

---

## Usage

### Environment Setup
```bash
conda activate camels_ru
cd /home/dmbrmv/Development/Dissertation
```

### Run Full LOBO Cross-Validation
```bash
python scripts/rfr_spatial_train.py
```

**Expected runtime**: ~1-2 hours per fold (depends on n_trials, n_basins, hardware)

### Configuration
Edit `scripts/rfr_spatial_train.py` to customize:

```python
# Reduce for quick testing
n_trials = 50          # Optuna trials per fold (default: 50)
timeout = 1800         # 30 minutes per fold

# Limit to subset of gauges for testing
gauge_ids = gauge_ids[:5]  # Test on 5 basins only
```

---

## Output Structure

```
data/optimization/rfr_spatial_results/
├── fold_1_70158/
│   ├── universal_model.joblib       # Trained RF model (can predict any basin)
│   ├── best_parameters.json         # Optimized hyperparameters
│   ├── test_metrics.json            # Performance on held-out basin 70158
│   └── optimization_study.pkl       # Full Optuna study
├── fold_2_70159/
│   └── ...
├── lobo_summary.csv                 # Aggregated metrics across all folds
└── ...
```

### Interpreting Results

**lobo_summary.csv** contains one row per basin with columns:
- `KGE`, `NSE`, `RMSE`, `MAE`, `PBIAS`: test metrics (2019-2020)
- `n_train_basins`: number of basins used for training
- `n_train_samples`: total training samples (pooled across basins)
- `test_basin`: held-out basin ID
- `fold_idx`: fold number

**Example analysis**:
```python
import pandas as pd

results = pd.read_csv("data/optimization/rfr_spatial_results/lobo_summary.csv", index_col=0)

# Overall performance
print(results[["KGE", "NSE", "PBIAS"]].describe())

# Spatial patterns
import matplotlib.pyplot as plt
results["KGE"].hist(bins=20)
plt.xlabel("KGE")
plt.title("Distribution of Test Performance Across Basins")
plt.show()
```

---

## Comparison: Regional vs Gauge-Specific

| Aspect | Gauge-Specific RFR | Regional RFR (LOBO) |
|--------|-------------------|-------------------|
| **Training** | One model per basin | One universal model |
| **Data usage** | Single basin time series | Pooled multi-basin data |
| **Ungauged prediction** | ❌ Not possible | ✅ Yes (via static attrs) |
| **Sample size** | Limited by basin length | Amplified by n_basins |
| **Overfitting risk** | Higher (fewer samples) | Lower (more samples) |
| **Calibration time** | Fast per basin | Slower (larger dataset) |
| **Spatial interpretation** | None | Feature importance shows spatial controls |

**When to use regional RFR**:
- Need predictions for ungauged basins
- Have multiple similar catchments
- Want to identify spatial controls on hydrology
- Have limited data per basin (spatial pooling helps)

**When to use gauge-specific RFR**:
- Have long time series (>10 years) per basin
- Basins are very heterogeneous (different climate zones)
- Need maximum accuracy for specific gauged locations
- Computational resources are limited

---

## Expected Performance

Based on hydrological ML literature, expect:

- **KGE**: 0.6–0.8 (median ~0.7)
- **NSE**: 0.5–0.7 (lower than gauge-specific due to spatial challenge)
- **PBIAS**: ±15% (acceptable for regional models)

Performance depends on:
1. **Basin similarity**: Homogeneous regions → better transfer
2. **Sample size**: More basins → better spatial patterns
3. **Static attribute quality**: HydroATLAS completeness matters
4. **Climate variability**: Stable regimes → easier prediction

---

## Predicting for Truly Ungauged Basins

After LOBO validation, train a **final global model** on all basins:

```python
from src.models.rfr_spatial.rfr_lobo import train_universal_model
import numpy as np

# Load all basins (2008-2018)
# ... (load features for all gauges)

# Train on full dataset
global_model = train_universal_model(
    x_train=X_all_basins,
    y_train=y_all_basins,
    params=best_params_from_lobo,  # Average of best params across folds
    n_jobs=-1
)

# Predict for ungauged basin with known static attributes
ungauged_static_attrs = [...]  # 17 HydroATLAS features
ungauged_temporal_features = create_universal_features(
    data=ungauged_meteo_data,
    gauge_id="ungauged_001",
    dataset="e5l",
    latitude=55.0,
    static_attrs=ungauged_static_attrs,
    rolling_windows=[1, 2, 4, 8, 16, 32]
)

# Generate predictions
X_ungauged = ungauged_temporal_features.drop(columns=["q_mm_day"]).values
y_pred = global_model.predict(X_ungauged)
```

---

## Limitations & Future Work

### Current Limitations
1. **Single meteorological dataset**: Uses ERA5-Land only (could ensemble multiple)
2. **Temporal stationarity**: Assumes 2008-2018 patterns apply to 2019-2020
3. **Static attributes**: Limited to 17 HydroATLAS features (could add more)
4. **No uncertainty quantification**: Point predictions only (could use quantile RF)

### Future Enhancements
1. **Spatial clustering**: Group basins by similarity, train regional models per cluster
2. **Transfer learning**: Pre-train on large basins, fine-tune on small basins
3. **Feature importance analysis**: Identify dominant controls (land cover vs soil vs topography)
4. **Ensemble predictions**: Combine gauge-specific + regional models
5. **Uncertainty estimates**: Quantile Random Forest or bootstrap ensembles

---

## References

1. **Kratzert et al. (2019)**. "Toward Improved Predictions in Ungauged Basins: Exploiting the Power of Machine Learning."  
   *Water Resources Research*, 55(12), 11344-11354.

2. **Oudin et al. (2005)**. "Which potential evapotranspiration input for a lumped rainfall–runoff model?"  
   *Journal of Hydrology*, 311(1-4), 299-311.

3. **Addor et al. (2017)**. "The CAMELS data set: catchment attributes and meteorology for large-sample studies."  
   *Hydrology and Earth System Sciences*, 21(10), 5293-5313.

4. **HydroATLAS**. Linke et al. (2019). Global hydro-environmental sub-basin and river reach characteristics at high spatial resolution.  
   *Scientific Data*, 6(1), 283.

---

## Troubleshooting

### Error: "Need at least 2 gauges for LOBO cross-validation"
- Ensure `data/nc_all_q/` contains ≥2 NetCDF files
- Check gauge IDs match between data files and HydroATLAS CSV

### Error: "Missing columns for gauge X with dataset Y"
- Verify meteorological dataset exists in NetCDF files
- Check column naming: `prcp_e5l`, `t_min_e5l`, `t_max_e5l`

### Error: "Gauge X not in static attributes"
- Ensure `gauge_id` column in `hydro_atlas_cis_camels.csv` uses string format
- Verify gauge IDs match exactly (case-sensitive)

### Memory issues
- Reduce `n_basins` (test on subset first)
- Reduce `n_estimators` in hyperparameter space (lower bound to 100)
- Use fewer `n_trials` per fold

### Slow performance
- Reduce `n_trials` (50 is reasonable for quick runs)
- Reduce `timeout` (30 min per fold is sufficient)
- Use fewer `rolling_windows` (try [2, 8, 32] only)
- Enable multiprocessing: `n_jobs=-1` in RF training

---

## Contact

For questions or issues:
1. Check logs: `logs/rfr_spatial_train.log`, `logs/rfr_lobo.log`
2. Review this README and code comments
3. Compare with gauge-specific RFR implementation (`src/models/rfr/`)

---

**Implementation Date**: October 17, 2025  
**Status**: ✅ Complete and ready for testing
