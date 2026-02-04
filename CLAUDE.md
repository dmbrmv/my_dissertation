# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhD dissertation comparing hydrological models (LSTM, HBV, GR4J, Random Forest) for daily river discharge prediction across Russian river basins. Core workflow: Load meteorological grids → Extract watershed features → Train/calibrate models → Evaluate against observations.

## Development Commands

```bash
# Package management (Pixi)
pixi run edit              # Start Marimo notebook editor
pixi run present           # Presentation mode

# Code quality
ruff check src/ --fix      # Lint with auto-fix
ruff format src/           # Format code
pyright src/               # Type check

# Model training
python scripts/hbv_train_simple.py      # HBV calibration (parallel)
python scripts/gr4j_train_simple.py     # GR4J calibration
python scripts/rfr_train_simple.py      # Random Forest

# Testing
pytest test/
```

## Architecture: "Notebooks as Thin Clients"

**Critical rule**: Notebooks orchestrate workflows; all complex logic lives in `src/`.

```
One-off exploration (≤5 lines)? → Inline in notebook
Reused across ≥2 notebooks? → Extract to src/
Dissertation-critical (models, metrics, data I/O)? → MANDATORY: src/ with tests
```

### Mandatory Notebook Structure
1. **Cell 1**: Imports only (no computation)
2. **Cell 2**: Configuration (paths, constants, `setup_logger()`)
3. **Cell 3+**: Execution delegating to `src/` functions

### Source Code Organization

```
src/
├── analytics/          # Statistical analysis, clustering
├── constants/          # Feature definitions
├── geometry/           # Watershed geometry operations
├── grids/              # NetCDF grid processing
├── hydro/              # Flow extremes, hydrological signatures
├── loaders/            # Data loading pipelines
├── models/
│   ├── gr4j/           # GR4J + CemaNeige (4-param + snow)
│   ├── hbv/            # HBV (16-param conceptual model)
│   ├── rfr/            # Random Forest (basin-specific)
│   └── rfr_spatial/    # Random Forest with spatial transfer (LOBO)
├── plots/              # Visualization functions
├── readers/            # Data loading (hydro, metadata, attributes)
├── timeseries_stats/   # Metrics (NSE, KGE, flow signatures)
└── utils/              # Logger and utilities
```

## Code Quality Standards

Enforced via `pyproject.toml`:
- **Line length**: 105 chars
- **Imports**: Google-style, sorted (stdlib → 3rd party → local)
- **Docstrings**: Google-style, required for public functions
- **Quotes**: Double quotes everywhere
- **Type hints**: Required for all public functions
- **Complexity**: Cyclomatic complexity ≤10
- **No print()**: Use `setup_logger()` from `src.utils.logger` in `src/` code

## Data Conventions

| Type | Format | Units | Location |
|------|--------|-------|----------|
| Meteorological | NetCDF | mm/day (prcp), °C (temp) | `data/nc_all_q/{gauge_id}.nc` |
| Discharge | CSV | q_mm_day | `data/hydro_csv/{gauge_id}.csv` |
| Watersheds | GeoJSON/Shapefile | — | `data/watersheds/` |
| Results | CSV/Pickle | — | `res/chapter_{one,two,three}/` |
| Logs | TXT | — | `logs/` |

- **Date indices**: pandas DatetimeIndex, daily frequency
- **CRS**: EPSG:4326 for all vector data
- **Missing values**: Use `np.nan`, not sentinel values

## Key Patterns

### Logging (MANDATORY)
```python
from src.utils.logger import setup_logger
logger = setup_logger("task_name", log_file="logs/task.log")
logger.info("Processing gauge 1234")
```
Never use `logging.getLogger()` directly or `print()` in `src/`.

### Loading Gauge Data
```python
import xarray as xr
with xr.open_dataset(f"data/nc_all_q/{gauge_id}.nc") as ds:
    df = ds.to_dataframe()
# df columns: prcp, t_min, t_max, evap, q_mm_day
```

### Model Calibration
- **Primary metric**: KGE (Kling-Gupta Efficiency)
- **Optimizer**: Optuna (1500 trials HBV, 1000 trials GR4J)
- **Parallelization**: ProcessPoolExecutor for training multiple gauges

### Geospatial Data
```python
import geopandas as gpd
watersheds = gpd.read_file("data/watersheds.geojson")
assert watersheds.crs.to_epsg() == 4326, "Expected EPSG:4326"
```

## Common Pitfalls

1. Complex logic in notebooks (extract to `src/`)
2. Using `print()` in `src/` code (use logger)
3. String path manipulation (use `pathlib.Path`)
4. Loading geospatial data without CRS validation
5. Missing warmup period for conceptual models (2 years default)
6. Not using context managers for `xr.open_dataset()`

## Model Summary

| Model | Params | Best For | Script |
|-------|--------|----------|--------|
| HBV | 16 | Snow-dominated basins | `scripts/hbv_train_simple.py` |
| GR4J | 4 | Temperate regions, fast computation | `scripts/gr4j_train_simple.py` |
| Random Forest | Auto | Feature interpretation, basin-specific | `scripts/rfr_train_simple.py` |
| RFR-Spatial | Auto | Ungauged basin prediction (LOBO) | — |
| LSTM | — | Best accuracy (archived, uses NeuralHydrology) | `archive/` |
