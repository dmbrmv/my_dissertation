# Model Inference Scripts

This directory contains scripts for generating final time series predictions using calibrated model parameters.

## Overview

Each script loads pre-calibrated parameters from `data/optimization/` and generates predictions saved to `data/predictions/`.

## Scripts

### 1. `gr4j_predict.py` - GR4J Conceptual Model
**Input:** Calibrated parameters from `data/optimization/gr4j_simple/`
**Output:** `data/predictions/gr4j/{gauge_id}/{gauge_id}_{dataset}_predictions.parquet`

**Parameters (7 total):**
- `x1`: Maximum capacity of production store (mm)
- `x2`: Groundwater exchange coefficient (mm/day)
- `x3`: One-day ahead maximum capacity of routing store (mm)
- `x4`: Time base of unit hydrograph (days)
- `ctg`: CemaNeige thermal coefficient
- `kf`: CemaNeige snowfall correction factor
- `tt`: CemaNeige threshold temperature (°C)

**Usage:**
```bash
conda activate camels_ru
python scripts/predict/gr4j_predict.py
```

---

### 2. `hbv_predict.py` - HBV Conceptual Model
**Input:** Calibrated parameters from `data/optimization/hbv_simple/`
**Output:** `data/predictions/hbv/{gauge_id}/{gauge_id}_{dataset}_predictions.parquet`

**Parameters (16 total):**
- Soil moisture: `parBETA`, `parFC`, `parLP`
- Response routing: `parPERC`, `parUZL`, `parK0`, `parK1`, `parK2`, `parMAXBAS`
- Evapotranspiration: `parCET`, `parPCORR`
- Snow routing: `parTT`, `parCFMAX`, `parCFR`, `parCWH`, `parSFCF`

**Usage:**
```bash
conda activate camels_ru
python scripts/predict/hbv_predict.py
```

---

### 3. `rfr_predict.py` - Random Forest Regressor (Temporal)
**Input:** Calibrated hyperparameters from `data/optimization/rfr_simple/`
**Output:** `data/predictions/rfr/{gauge_id}/{gauge_id}_{dataset}_predictions.parquet`

**Features:**
- **Temporal:** PET (Oudin), cyclic day-of-year, rolling windows [1,2,4,8,16,32]
- **Static (17 attributes):** Land cover, soil properties, topography, hydrogeology

**Hyperparameters:**
- `n_estimators`: Number of trees in forest
- `max_depth`: Maximum depth of each tree
- `min_samples_split`: Minimum samples to split node
- `min_samples_leaf`: Minimum samples in leaf node
- `max_features`: Features to consider for split ('sqrt', 'log2', 0.5, 0.7)

**Usage:**
```bash
conda activate camels_ru
python scripts/predict/rfr_predict.py
```

---

### 4. `rfr_spatial_predict.py` - Random Forest Regressor (Spatial/LOBO)
**Input:** N/A (retrains for each basin using optimized hyperparameters)
**Output:** `data/predictions/rfr_spatial/{gauge_id}/{gauge_id}_{dataset}_predictions.parquet`

**Special Notes:**
- Uses **Leave-One-Basin-Out (LOBO)** cross-validation strategy
- For each test gauge:
  1. Trains RF on ALL other gauges (2008-2018)
  2. Predicts on held-out gauge (2019-2020)
- Tests **spatial generalization** (ungauged basin prediction)
- Currently uses fixed hyperparameters; can be extended to load from LOBO calibration

**Usage:**
```bash
conda activate camels_ru
python scripts/predict/rfr_spatial_predict.py
```

---

## Output Format

All scripts save predictions in **Parquet format** (efficient, compressed, type-safe) with the following structure:

| Column    | Type     | Description                        |
|-----------|----------|------------------------------------|
| `date`    | datetime | Timestamp                          |
| `q_obs`   | float64  | Observed discharge (mm/day)        |
| `q_sim`   | float64  | Simulated discharge (mm/day)       |
| `dataset` | string   | Meteorological dataset (e5l, etc.) |
| `gauge_id`| string   | Gauge identifier                   |

**Metrics** are saved alongside predictions:
- `{gauge_id}_{dataset}_prediction_metrics.json`

Metrics include: **KGE, NSE, RMSE, MAE, PBIAS, R², correlation, low-flow bias, high-flow bias**.

---

## Configuration

### Datasets
All scripts support multiple meteorological datasets:
- `e5l`: ERA5-Land
- `gpcp`: Global Precipitation Climatology Project
- `e5`: ERA5
- `mswep`: Multi-Source Weighted-Ensemble Precipitation

### Periods
- **Training (conceptual models):** 2010-2018 with 2-year warm-up
- **Training (RF models):** 2008-2018
- **Validation/Test:** 2019-2020

### File Formats
Scripts support three output formats (default: `parquet`):
- `parquet`: Recommended (fast, compressed, type-safe)
- `csv`: Human-readable (larger files)
- `netcdf`: CF-compliant (for geospatial analysis)

---

## Dependencies

Required packages (from `camels_ru` environment):
- `pandas`, `numpy`, `xarray`
- `geopandas`, `shapely`
- `scikit-learn` (RF models only)
- `pyarrow` or `fastparquet` (for Parquet I/O)

Model implementations:
- `src.models.gr4j`: GR4J model
- `src.models.hbv`: HBV model
- `src.models.gr4j.pet`: PET calculation (Oudin formula)
- `src.utils.metrics`: Evaluation metrics
- `src.utils.logger`: Logging utilities
- `src.readers.geom_reader`: Gauge geometry loader

---

## Workflow

### Step 1: Run Calibration (if not done yet)
```bash
# GR4J
python scripts/gr4j_train_simple.py

# HBV
python scripts/hbv_train_simple.py

# RF (temporal)
python scripts/rfr_train_simple.py

# RF (spatial)
python scripts/rfr_spatial_train_simple.py
```

### Step 2: Run Inference
```bash
# All models
python scripts/predict/gr4j_predict.py
python scripts/predict/hbv_predict.py
python scripts/predict/rfr_predict.py
python scripts/predict/rfr_spatial_predict.py
```

### Step 3: Analyze Results
```python
import pandas as pd

# Load predictions
gr4j = pd.read_parquet("data/predictions/gr4j/gauge_123/gauge_123_e5l_predictions.parquet")

# Quick plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(gr4j["date"], gr4j["q_obs"], label="Observed", alpha=0.7)
plt.plot(gr4j["date"], gr4j["q_sim"], label="Simulated", alpha=0.7)
plt.legend()
plt.show()
```

---

## Notes

1. **Warm-up periods:** GR4J and HBV use 2-year warm-up to initialize internal states
2. **Static attributes:** RF models require `data/attributes/hydro_atlas_cis_camels.csv`
3. **Gauge geometries:** All scripts require gauge shapefiles (loaded via `geom_reader`)
4. **Error handling:** Scripts log errors to `logs/` but continue processing other gauges
5. **Parallelization:** Not implemented in prediction scripts (fast enough serially)

---

## Troubleshooting

### Missing parameters
**Error:** `Parameters not found: data/optimization/.../params.json`
**Solution:** Run the corresponding training script first

### Missing static attributes
**Error:** `Static attributes file not found`
**Solution:** Ensure `data/attributes/hydro_atlas_cis_camels.csv` exists

### Type errors (Pyright)
**Issue:** Some sklearn type stubs may be incomplete
**Solution:** Add `# type: ignore[import-untyped]` for sklearn imports (already done)

### Memory issues (LOBO)
**Issue:** `rfr_spatial_predict.py` loads all gauges simultaneously
**Solution:** Process gauges in batches or reduce rolling windows

---

## Citation

If using these models, please cite:
- **GR4J:** Perrin et al. (2003), doi:10.1016/S0022-1694(03)00225-7
- **HBV:** Bergström (1992), doi:10.1016/0022-1694(92)90177-T
- **RF:** Breiman (2001), doi:10.1023/A:1010933404324
- **LOBO:** Kratzert et al. (2019), doi:10.5194/hess-23-5089-2019
