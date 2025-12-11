# Copilot Instructions: Hydrological Modeling Dissertation

## Project Overview

This is a PhD dissertation project on hydrological modeling across Russian river basins using deep learning, conceptual models, and machine learning approaches. The codebase compares multiple modeling frameworks (LSTM, HBV, GR4J, Random Forest) for river flow prediction under data-limited conditions.

**Core workflow**: Load meteorological grids (ERA5, MSWEP) → Extract watershed features → Train/calibrate models → Evaluate against observations → Visualize results by region/cluster

## Critical Setup

- **Python environment**: Always use `conda activate geo` (see `dis_env_mac.yml`)
- **Running Python code**: Use conda environment, never bare `python`
- **Project structure**: Notebooks are thin orchestration layers; all reusable logic lives in `src/`

## Architecture: "Notebooks as Thin Clients"

### The Golden Rule
Notebooks (`notebooks/`) orchestrate workflows. All complex logic, plotting functions, data loaders, and models belong in `src/`. This is non-negotiable.

**Decision tree for code placement**:
```
One-off exploration in single notebook? → Inline with docstring (≤5 lines)
Reused across ≥2 notebooks? → Extract to src/
Dissertation-critical (models, metrics, data I/O)? → MANDATORY: src/ with tests
```

### Mandatory Notebook Structure
Every notebook MUST follow this cell order (enforced by linting):

1. **Cell 1 - Imports ONLY**: No computation. Only `sys.path.append("../")` allowed for path setup.
2. **Cell 2 - Configuration**: Constants, paths, logger initialization via `setup_logger()`.
3. **Cell 3+ - Execution**: Each cell = one logical step delegating to `src/` functions.

**Example notebook skeleton**:
```python
# CELL 1: Imports
import sys
sys.path.append("../")
from src.loaders.discharge_loader import load_discharge_data
from src.plots.cluster_plots import plot_dendrogram
from src.utils.logger import setup_logger

# CELL 2: Config
DATA_DIR = Path("../data")
logger = setup_logger("chapter_one", log_file="../logs/chapter_one.log")

# CELL 3+: Execution
discharge_df = load_discharge_data(DATA_DIR / "nc_all_q", gauge_ids=["1234", "5678"])
```

### Forbidden in Notebooks
- Functions >5 lines (extract to `src/`)
- Multi-line data structures >20 lines (move to `src/constants/` or JSON)
- `print()` for inspection (use `logger.info()` or display DataFrames directly)
- Multiple `sys.path` manipulations or `%cd` magic commands
- Out-of-order cell execution (notebooks must be reproducible top-to-bottom)

## Key Modules & Data Flow

### 1. Data Loading Pipeline

**Meteorological Data** (`src/readers/hydro_data_reader.py`):
- NetCDF grids stored per gauge: `data/nc_all_q/{gauge_id}.nc`
- Variables: `prcp`, `t_min`, `t_max`, `evap` (columns standardized across datasets)
- Always use `xr.open_dataset()` with context manager
- Example: `with xr.open_dataset(f"data/nc_all_q/{gauge_id}.nc") as ds: df = ds.to_dataframe()`

**Discharge Data** (`src/loaders/`):
- CSV format: `data/hydro_csv/{gauge_id}.csv` with columns `[date, q_mm_day]`
- Always check for missing values via `find_valid_gauges()`

**Static Attributes** (`src/readers/hydro_atlas_reader.py`):
- HydroATLAS features for each watershed (area, slope, land cover, climate indices)
- Used by Random Forest models for spatial generalization

### 2. Model Implementations

All models live in `src/models/` with parallel execution support:

**HBV** (`src/models/hbv/hbv.py`):
- 16-parameter conceptual model with snow routine
- Calibration: Single-objective KGE optimization via Optuna (1500 trials)
- Run script: `scripts/hbv_train_simple.py` with parallel ProcessPoolExecutor
- Inputs: `temp`, `prcp`, `evap` (PET via Oudin formula)

**GR4J** (`src/models/gr4j/model.py`):
- 4-parameter daily rainfall-runoff + CemaNeige snow module
- Calibration: Single-objective KGE (1000 trials sufficient)
- Run script: `scripts/gr4j_train_simple.py`
- Faster than HBV but requires warmup period (2 years default)

**Random Forest** (`src/models/rfr/`):
- Two variants:
  - `rfr/`: Basin-specific models (one per gauge)
  - `rfr_spatial/`: Regional LOBO (Leave-One-Basin-Out) with static attributes
- Feature engineering: Rolling windows [1,2,4,8,16,32] days for temporal dependencies
- Multi-objective optimization: KGE + composite high/low flow metrics

### 3. Metrics & Evaluation

**Primary metrics** (`src/timeseries_stats/metrics.py`):
- `kling_gupta_efficiency()`: Primary calibration target (KGE)
- `nash_sutcliffe_efficiency()`: NSE for comparison
- `percent_bias()`: PBIAS for water balance

**Flow regime analysis** (`src/timeseries_stats/metrics_enhanced.py`):
- `composite_high_flow_metric()`: High flow performance
- `composite_low_flow_metric()`: Low flow bias + correlation
- Base Flow Index via Lyne-Hollick filter

### 4. Plotting Functions

All plotting **MUST** use functions from `src/plots/`:
- `cluster_plots.py`: Dendrograms, cluster heatmaps
- `maps.py`: Watershed boundary maps with results
- `hex_maps.py`: Hexbin spatial aggregation
- `numeric_plots.py`: Hydrographs, scatter plots, flow duration curves

**Example usage**:
```python
from src.plots.numeric_plots import plot_hydrograph
plot_hydrograph(observed, simulated, gauge_id="1234", save_path=OUTPUT_DIR / "plot.png")
```

### 5. Logging System

**MANDATORY**: Always use `setup_logger()` from `src/utils/logger.py`:
```python
from src.utils.logger import setup_logger
logger = setup_logger("task_name", log_file="logs/task.log", level="INFO")
logger.info("Processing gauge 1234")
```

**Never**:
- Use `logging.getLogger()` directly
- Use `print()` in `src/` code (T20 rule violation)

Logs go to `logs/` directory with emoji + color formatting for console output.

## Development Workflows

### Running Model Training

1. **Parallel calibration** (recommended):
   ```bash
   conda activate geo
   python scripts/hbv_train_simple.py  # Calibrates all gauges in parallel
   ```

2. **Results saved to**: `res/hbv/` (or `gr4j/`, `rfr/`) with subdirs per dataset

3. **Monitoring**: Check `logs/{model}_simple_optim.log` for progress

### Code Quality Checks

Enforced by `pyproject.toml` with Ruff + Pyright:

```bash
ruff check src/ --fix          # Lint with auto-fix
ruff format src/               # Format code
pyright src/                   # Type check
```

**Key rules**:
- Line length: 90 chars
- Import order: Standard lib → Third party → First party (`floodforecast`)
- Double quotes everywhere
- No `print()` in `src/` (use logging)
- Cyclomatic complexity ≤10
- Type hints required for all public functions

### Testing

Run tests with:
```bash
pytest test/{hbv,gr4j,rfr}/  # Model-specific tests
```

## Common Patterns

### 1. Loading Gauge Data for Model Training
```python
import xarray as xr
import pandas as pd

with xr.open_dataset(f"data/nc_all_q/{gauge_id}.nc") as ds:
    df = ds.to_dataframe()

# df has columns: prcp, t_min, t_max, evap, q_mm_day
# Index: DatetimeIndex
```

### 2. Feature Engineering for Random Forest
```python
from src.models.rfr.rfr_optuna import create_temporal_features

# Adds rolling windows + cyclic day-of-year encoding
features_df = create_temporal_features(
    meteo_df, 
    rolling_windows=[1, 2, 4, 8, 16, 32],
    meteo_vars=["prcp_e5l", "t_min_e5l", "t_max_e5l"]
)
```

### 3. Model Calibration with Optuna
```python
import optuna
from src.models.hbv import hbv
from src.timeseries_stats.metrics import kling_gupta_efficiency

def objective(trial):
    # Sample parameters from bounds
    params = [trial.suggest_float(f"p{i}", *bound) for i, bound in enumerate(hbv.bounds())]
    
    # Run simulation
    q_sim = hbv.simulation(data, params)
    
    # Calculate KGE
    kge = kling_gupta_efficiency(observed, q_sim)
    return kge

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1500)
```

### 4. Geospatial Data Handling
```python
import geopandas as gpd

# ALWAYS validate CRS on load (enforced in src/loaders/)
watersheds = gpd.read_file("data/watersheds.geojson")
assert watersheds.crs is not None, "Missing CRS"
assert watersheds.crs.to_epsg() == 4326, f"Expected EPSG:4326, got {watersheds.crs}"
```

## Data Conventions

- **Date indices**: Always pandas DatetimeIndex, daily frequency
- **Runoff units**: `q_mm_day` (discharge normalized by catchment area)
- **Precipitation**: `prcp` in mm/day
- **Temperature**: `t_min`, `t_max` in °C
- **PET**: `evap` in mm/day (Oudin formula for conceptual models)
- **CRS**: EPSG:4326 for all vector data
- **Missing values**: Use `np.nan`, not sentinel values

## File Paths

Critical directories:
- `data/nc_all_q/`: Per-gauge NetCDF files with meteo + discharge
- `data/watersheds/`: GeoJSON/shapefiles for catchment boundaries
- `res/chapter_{one,two,three}/`: Results organized by dissertation chapter
  - Each chapter has: `tables/` (CSV/Excel outputs), `images/` (plots/figures)
  - Example: `res/chapter_one/tables/model_metrics.csv`, `res/chapter_one/images/hydrograph_1234.png`
- `logs/`: All log files
- `scripts/`: Entry points for model training
- `notebooks/`: Analysis notebooks (one per chapter/task)

## Documentation References

For detailed model documentation, see:
- `docs/CONCEPTUAL_MODELS.md`: HBV and GR4J implementation details
- `docs/MACHINE_LEARNING_MODELS.md`: Random Forest variants
- `docs/PLOTTING_FUNCTIONS_USAGE.md`: Visualization guidelines

## Common Pitfalls to Avoid

1. **Never** put complex logic in notebooks (violates architecture)
2. **Never** use `print()` in `src/` code (use logger)
3. **Never** manipulate paths with strings (use `pathlib.Path`)
4. **Never** load data without validating CRS for geospatial data
5. **Never** commit notebooks with outputs (clear before commit)
6. **Never** run Python without activating `geo` conda environment
7. **Always** use `setup_logger()`, not `logging.getLogger()`
8. **Always** include warmup period (2 years) for conceptual models
9. **Always** use context managers for `xr.open_dataset()`
10. **Always** validate notebook sequential execution (no hidden state)
