---
applyTo: "*.py, *.ipynb"
---
# Hydrological Modeling & Earth Sciences — Python 3.12 Codex (Ruff/Pyright)

## Prime directive
Ship the simplest correct, reproducible, lint-clean solution optimized for 
hydrological modeling workflows. Verify imports, typing, determinism, and test 
coverage before replying.

**IMPORTANT:** Do NOT create summary.md, report.md, or any documentation files 
unless explicitly requested by the user. Focus on code and technical solutions only.
In order to run python use conda env: `conda activate geo`

---

## Architecture Decision Tree: Code Placement Logic

**Use this decision tree to resolve "simplest solution" vs. "layered architecture" conflicts:**

```
┌─ Is this a one-off exploratory task (single notebook, throwaway analysis)?
│  └─ YES → Implement inline in notebook WITH function wrapper + docstring
│  └─ NO  → Continue ↓
│
├─ Will this code be reused across ≥2 notebooks or scripts?
│  └─ YES → Extract to src/ with full typing and tests
│  └─ NO  → Continue ↓
│
├─ Is this dissertation-critical logic (modeling, data processing, metrics)?
│  └─ YES → MANDATORY: src/ with layered architecture + unit tests
│  └─ NO  → Continue ↓
│
└─ Is function >5 lines OR contains nested loops/complex logic?
   └─ YES → Extract to src/ (notebook lint violation)
   └─ NO  → Inline acceptable with docstring
```

**Golden Rules:**
- **Data I/O (loaders, readers)**: ALWAYS in `src/loaders/` or `src/readers/`
- **Plotting functions**: ALWAYS in `src/plots/`
- **Metrics/validation**: ALWAYS in `src/utils/` or `src/hydro/`
- **Model implementations**: ALWAYS in `src/models/`
- **Notebooks**: Orchestration, parameter tuning, visualization ONLY

---

## Notebook Philosophy: The "Top-Down Structure-Invariant" Pattern

**CRITICAL: Notebooks are structure-invariant thin clients with MANDATORY cell ordering.**

### **Mandatory Notebook Structure (Non-Negotiable)**

```python
# ============================================================================
# CELL 1: IMPORTS ONLY (No computation, no path manipulation beyond sys.path)
# ============================================================================
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append("../")  # ONLY acceptable path manipulation

from src.analytics.chapter_one import cluster_analysis
from src.loaders.discharge_loader import load_discharge_data
from src.plots.cluster_plots import plot_dendrogram
from src.utils.logger import setup_logger

# Configure matplotlib ONCE
plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# ============================================================================
# CELL 2: CONFIGURATION & LOGGING (Constants, paths, logger setup)
# ============================================================================
# Runtime configuration
N_CLUSTERS = 10
TARGET_CRS = "EPSG:4326"
DATA_DIR = Path("../data")
OUTPUT_DIR = Path("../res/chapter_one")

# Setup logging (MANDATORY: Use setup_logger, never logging.getLogger directly)
logger = setup_logger("chapter_one", log_file="../logs/chapter_one.log")
logger.info(f"Notebook initialized: {N_CLUSTERS} clusters, CRS={TARGET_CRS}")

# ============================================================================
# CELL 3+: EXECUTION CELLS (Load → Process → Visualize)
# Each cell: ONE logical step, delegates to src/ functions
# ============================================================================
```

### **Notebook Enforcement Rules**

1. **Cell 1 is imports ONLY**: No computation, no file I/O, no config. Only `sys.path.append("../")` allowed.

2. **Cell 2 is configuration ONLY**: Constants, paths, logger initialization. No data loading.

3. **Function Length Limit**: Any function definition in a notebook **>5 lines** (excluding docstring) is a **LINT VIOLATION**. Extract to `src/`.

4. **No Multi-Line Data Structures**: Dictionaries/lists **>20 lines** (like `feature_descriptions`) MUST be moved to:
   - `src/constants/*.py` (static metadata)
   - JSON/YAML files in `data/` (loaded via `src.loaders`)
   - **Exception**: Plotting configuration dicts ≤10 lines inline acceptable

5. **Zero Hidden State**: Every cell must be executable **independently after Cell 1-2**. No dependencies on out-of-order execution.

6. **Determinism Validation**: Final cell MUST include execution counter validation:
   ```python
   # MANDATORY: Place as last cell
   if "_exec_counter" not in dir():
       _exec_counter = []
   _exec_counter.append(len(_exec_counter) + 1)
   expected = list(range(1, len(_exec_counter) + 1))
   assert _exec_counter == expected, f"Out-of-order execution: {_exec_counter}"
   assert Path.cwd().name == "notebooks", f"Wrong directory: {Path.cwd()}"
   print("✓ Notebook executed sequentially from clean state")
   ```

### **Jupyter Specific Standards**

1. **No `sys.path` manipulation** beyond `sys.path.append("../")` in Cell 1
2. **No `%cd` magic commands** (breaks path reproducibility)
3. **No `print()` for data inspection** — use `logger.info()` or display DataFrames directly
4. **Clear outputs before commit** unless plot is required for documentation
5. **Apply Ruff formatting** to all code cells (use `ruff format` on extracted .py representation)
6. **MANDATORY logger usage**: Cell 2 MUST initialize logger via `setup_logger(function_name, log_file=...)`. Never use `logging.getLogger()` directly in notebooks.

## Language & formatting (enforced by pyproject.toml)
- **Python 3.12** semantics; Ruff formatter with **90-char line length**.
- **Double quotes** for all strings/docstrings (triple double for multi-line).
- **Google-style docstrings**; module/package docstrings optional (D100/D104 ignored).
- Code examples in docstrings: ≤72 chars per line (formatted via `docstring-code-format`).
- `pathlib.Path` for all filesystem paths; f-strings for interpolation only.
- Avoid trailing commas in single-line constructs unless required by Ruff.

## Ruff linting rules (auto-fixed where possible)
Configured ruleset: `["E", "W", "F", "I", "N", "D", "UP", "B", "S", "C4", "Q", "T20", "A", "C90", "PIE"]`

**Critical rules:**
- **E/W/F**: pycodestyle + Pyflakes — no unused imports/vars, no bare `except`.
- **I (isort)**: Import order with first-party = `["floodforecast"]` (update if renamed).
- Combine "as" imports; force sort within sections; split on trailing comma.
- **N**: pep8-naming (snake_case funcs/vars, CapWords classes, UPPER_CASE constants).
- **D**: Google docstrings; one-line summaries; D100/D104 suppressed.
- **UP**: pyupgrade for 3.12+ (e.g., `list[str]` over `List[str]`).
- **B**: bugbear checks (e.g., no mutable defaults, no `==` for singletons).
- **Q**: double quotes everywhere.
- **T20**: **No `print()` in `src/**`**; use `logging.getLogger(__name__)`.  
✅ Prints allowed in: `scripts/**`, `archive/**`, `**/*.ipynb`.
- **A**: Never shadow builtins (`list`, `dict`, `id`, `filter`, etc.).
- **C90**: Cyclomatic complexity ≤ 10; refactor when threshold exceeded.
- **S (bandit)**: No `eval`/`exec`, insecure temp files, or `shell=True`; sanitize inputs.

**Per-file exceptions (from pyproject.toml):**
- `test/**`: S101 (assert), D100-D103, T20 allowed.
- `scripts/**`, `**/*.ipynb`: T20 allowed; D100-D104 ignored.

## Pyright type checking (`typeCheckingMode = "basic"`)
- **Type-hint pervasively**; prefer precise types over `Any`.
- Return types **required** for public functions/methods; avoid implicit `Optional`.
- Use `Literal`/`Enum` for constrained choices; annotate generators/iterables explicitly.
- NumPy/xarray: use `np.ndarray`, `xr.DataArray`, `xr.Dataset` from stubs.
- Pandas: annotate as `pd.DataFrame`, `pd.Series` with column/index hints where feasible.
- **Ensure Pyright "basic" passes**: resolve all imports, stub warnings acceptable.

## Project architecture (hydrological modeling context)
src/ ├── geometry/ # Watershed delineation, spatial ops ├── grids/ # Meteorological grid processing (NetCDF/GDAL) ├── loaders/ # Data ingestion (discharge, static attributes) ├── models/ # HBV, GR4J, LSTM implementations ├── readers/ # Format-specific readers (HydroATLAS, AIS, etc.) └── utils/ # Metrics, logging, validation archive/ # Legacy experiments (lenient linting) scripts/ # CLI tools (prints allowed) test/ # Unit tests (assert allowed) notebooks/ # Analysis notebooks (prints + minimal docs OK)


**Code organization:**
- Functions ≤40 LOC; single responsibility; isolate I/O from logic.
- Layered workflow: **ingestion → preprocessing → modeling → evaluation**.
- Config via `TypedDict`/`dataclasses`; avoid hidden globals or module-level state.
- CLI scripts: use `if __name__ == "__main__":` guard; leverage `argparse`/`click`.
- Never log secrets; sanitize paths/env vars in logs.

## Hydrology & Earth Science Specifics: Runtime Validation Requirements

### **Data Structures**
- **Multi-dimensional gridded data**: `xarray.Dataset`/`DataArray` with labeled dims 
  (time, lat, lon, basin_id).
- **Tabular data**: Prefer `polars.DataFrame` for large CSVs; pandas for small/legacy data.
- **Large/OOM datasets**: Use Dask-backed xarray (`chunks`) or Zarr storage.
- **Geospatial**: `geopandas.GeoDataFrame` + `shapely` for vectors; `rasterio`/`rioxarray` 
  for rasters.

---

### **MANDATORY: Geospatial CRS Validation (Runtime Assertions)**

**Rule**: All geospatial data loaders MUST validate CRS on load. No exceptions.

**Implementation Pattern (REQUIRED in `src/loaders/` and `src/readers/`):**

```python
from typing import Literal
import geopandas as gpd
import logging

def load_geodata(
    path: Path,
    expected_crs: str | None = None,
    target_crs: str = "EPSG:4326",
    logger: logging.Logger | None = None,
) -> gpd.GeoDataFrame:
    """Load geospatial data with mandatory CRS validation.
    
    Args:
        path: Path to geospatial file (GeoPackage, Shapefile, etc.)
        expected_crs: Expected CRS string. If None, any defined CRS accepted.
        target_crs: CRS to reproject to (default WGS84)
        logger: Logger for CRS transformations
        
    Returns:
        GeoDataFrame reprojected to target_crs
        
    Raises:
        ValueError: If CRS is undefined or mismatches expected_crs
        
    Example:
        >>> gdf = load_geodata(
        ...     Path("watersheds.gpkg"),
        ...     expected_crs="EPSG:32637",  # UTM Zone 37N
        ...     target_crs="EPSG:4326"
        ... )
    """
    gdf = gpd.read_file(path)
    
    # MANDATORY: Assert CRS is defined
    if gdf.crs is None:
        raise ValueError(
            f"Undefined CRS in {path.name}. "
            f"Set CRS explicitly: gdf.set_crs('EPSG:XXXX', inplace=True)"
        )
    
    # OPTIONAL: Validate expected CRS
    if expected_crs and str(gdf.crs) != expected_crs:
        raise ValueError(
            f"CRS mismatch in {path.name}: "
            f"expected {expected_crs}, got {gdf.crs}"
        )
    
    # MANDATORY: Log reprojection
    if str(gdf.crs) != target_crs:
        if logger:
            logger.info(f"Reprojecting {path.name}: {gdf.crs} → {target_crs}")
        gdf = gdf.to_crs(target_crs)
    
    return gdf
```

**Validation in Notebooks:**
```python
# Cell 3: Load geospatial data with CRS validation
ws, gauges = load_geodata(folder_depth="../")  # Uses validated loader

# ASSERT: Verify CRS consistency across datasets
assert ws.crs is not None, "Watershed CRS undefined"
assert gauges.crs is not None, "Gauge CRS undefined"
assert str(ws.crs) == str(gauges.crs), f"CRS mismatch: {ws.crs} vs {gauges.crs}"
logger.info(f"✓ CRS validated: {ws.crs}")
```

---

### **MANDATORY: Temporal Axis Validation (Runtime Assertions)**

**Rule**: All xarray DataArrays/Datasets with time dimensions MUST validate temporal integrity on load.

**Implementation Pattern (REQUIRED in `src/loaders/`):**

```python
from typing import Literal
import xarray as xr
import pandas as pd
import numpy as np
import logging

def validate_temporal_axis(
    da: xr.DataArray,
    freq: Literal["D", "H", "M", "MS"] | None = None,
    calendar: Literal["standard", "proleptic_gregorian"] = "proleptic_gregorian",
    allow_gaps: bool = False,
    logger: logging.Logger | None = None,
) -> xr.DataArray:
    """Validate temporal axis integrity with mandatory assertions.
    
    Args:
        da: DataArray with 'time' dimension
        freq: Expected frequency (D=daily, H=hourly, M=monthly, MS=month-start)
        calendar: CF-compliant calendar type
        allow_gaps: If False, raise on non-monotonic or gapped time series
        logger: Logger for validation messages
        
    Returns:
        Validated DataArray (unchanged if valid)
        
    Raises:
        ValueError: If time axis validation fails
        
    Example:
        >>> discharge = validate_temporal_axis(
        ...     xr.open_dataarray("discharge.nc"),
        ...     freq="D",
        ...     calendar="proleptic_gregorian",
        ...     allow_gaps=False
        ... )
    """
    if "time" not in da.dims:
        raise ValueError(f"Missing 'time' dimension. Found dims: {list(da.dims)}")
    
    time_coord = da["time"]
    
    # MANDATORY: Assert datetime64 type
    if not pd.api.types.is_datetime64_any_dtype(time_coord):
        raise ValueError(
            f"Time coordinate must be datetime64, got {time_coord.dtype}. "
            f"Decode with xr.decode_cf() or xr.open_dataset(decode_times=True)"
        )
    
    # MANDATORY: Check monotonic increasing
    time_values = time_coord.values
    if not np.all(time_values[:-1] <= time_values[1:]):
        raise ValueError(
            f"Non-monotonic time axis detected. "
            f"Sort with da.sortby('time') or check for duplicates."
        )
    
    # MANDATORY: Validate calendar attribute
    calendar_attr = time_coord.attrs.get("calendar", "standard")
    if calendar_attr not in {"standard", "proleptic_gregorian"}:
        raise ValueError(
            f"Unsupported calendar: {calendar_attr}. "
            f"Convert to 'proleptic_gregorian' with xr.coding.times"
        )
    
    # OPTIONAL: Check time gaps (if freq specified and gaps not allowed)
    if freq and not allow_gaps:
        time_diff = pd.Series(time_values).diff().dropna()
        expected_delta = pd.Timedelta(1, unit=freq)
        
        # Allow 1-hour tolerance for daily data (handles DST/rounding)
        tolerance = pd.Timedelta("1H") if freq == "D" else expected_delta * 0.01
        
        gaps = time_diff[time_diff > (expected_delta + tolerance)]
        if not gaps.empty:
            raise ValueError(
                f"Time gaps detected (expected freq={freq}): "
                f"{len(gaps)} gaps, largest={gaps.max()}. "
                f"Fill gaps with .resample() or set allow_gaps=True"
            )
    
    # MANDATORY: Ensure timezone-naive (UTC assumed)
    if hasattr(time_values[0], "tzinfo") and time_values[0].tzinfo is not None:
        raise ValueError(
            f"Time coordinate must be timezone-naive (UTC assumed). "
            f"Convert with .dt.tz_localize(None)"
        )
    
    if logger:
        logger.info(
            f"✓ Time axis validated: {len(time_coord)} steps, "
            f"range={time_coord.values[0]} to {time_coord.values[-1]}, "
            f"freq={freq or 'variable'}"
        )
    
    return da


def load_discharge_data(
    path: Path,
    basin_id: str,
    freq: Literal["D", "H"] = "D",
    logger: logging.Logger | None = None,
) -> xr.DataArray:
    """Load discharge data with full temporal and physical validation.
    
    Args:
        path: Path to NetCDF file
        basin_id: Basin identifier for subsetting
        freq: Expected temporal frequency
        logger: Logger for validation messages
        
    Returns:
        Validated discharge DataArray (m³/s)
        
    Example:
        >>> discharge = load_discharge_data(
        ...     Path("../data/discharge.nc"),
        ...     basin_id="12345",
        ...     freq="D"
        ... )
    """
    da = xr.open_dataarray(path)
    
    # Subset by basin
    if "basin_id" in da.dims:
        da = da.sel(basin_id=basin_id)
    
    # MANDATORY: Temporal validation
    da = validate_temporal_axis(da, freq=freq, allow_gaps=False, logger=logger)
    
    # MANDATORY: Physical bounds validation
    if (da < 0).any():
        n_negative = (da < 0).sum().item()
        raise ValueError(
            f"Negative discharge values detected: {n_negative} timesteps. "
            f"Discharge must be ≥ 0 m³/s. Clean data or mask invalid values."
        )
    
    # MANDATORY: Units validation
    units = da.attrs.get("units", "UNDEFINED")
    if units not in {"m³/s", "m3/s", "m^3/s"}:
        raise ValueError(
            f"Invalid discharge units: {units}. Expected 'm³/s'. "
            f"Set da.attrs['units'] = 'm³/s'"
        )
    
    if logger:
        logger.info(f"✓ Discharge data validated: {basin_id}, {len(da)} timesteps")
    
    return da
```

**Validation in Notebooks:**
```python
# Cell 4: Load and validate discharge data
discharge_da = load_discharge_data(
    path=DATA_DIR / "discharge.nc",
    basin_id="12345",
    freq="D",
    logger=logger
)

# ASSERT: Verify data quality
assert discharge_da.notnull().all(), "Missing discharge values detected"
assert (discharge_da >= 0).all(), "Negative discharge detected"
assert discharge_da.attrs.get("units") in {"m³/s", "m3/s"}, "Invalid units"
logger.info(f"✓ Discharge data quality validated: {len(discharge_da)} days")
```

---

### **CF Conventions & Metadata (Enforcement)**

- **NetCDF files**: Follow CF-1.8 conventions (MANDATORY for all saved outputs)
  - **Required attributes**: `units`, `long_name`, `standard_name` (where applicable)
  - **Coordinate variables**: `time` (CF datetime), `lat`, `lon` (or `x`, `y` for projected)
  - **CRS**: Store in `crs` variable or coordinate attribute (`spatial_ref`)
  
- **Units Documentation**:
  - **Docstrings**: MUST specify units for all physical quantities (e.g., mm/day, m³/s, °C)
  - **xarray attrs**: MUST include `units` attribute on all DataArrays
  
- **Physical Bounds Validation** (MANDATORY assertions in loaders):
  - Precipitation: `≥ 0 mm/day`
  - Temperature: `-100°C ≤ T ≤ +100°C` (raise on physical impossibility)
  - Discharge: `≥ 0 m³/s`
  - Relative humidity: `0% ≤ RH ≤ 100%`
  
- **Missing Data**: Use `NaN` ONLY. No sentinel values (-9999, 999). Raise if detected.

**Hydrological metrics:**
- Report standard metrics: **NSE, KGE, RMSE, MAE, bias %**.
- Regime-specific: low-flow bias, high-flow bias, seasonal decomposition.
- Stabilize divisions: add epsilon (1e-8) to denominators to prevent div-by-zero.
- Provide skill scores with uncertainty bounds when possible.

**File formats (priority order):**
1. **Zarr** (cloud-optimized, multi-dimensional, chunked).
2. **NetCDF4** (self-describing, CF-compliant, HDF5 backend).
3. **Parquet** (tabular; columnar compression; better than CSV).
4. **GeoPackage** (vectors; single-file; better than Shapefiles).
5. Avoid: CSV for large arrays, pickles (non-portable), Shapefiles (legacy).

**NetCDF Chunking Strategy (MANDATORY for large datasets):**
- **Time-series operations** (`.sel()`, `.isel()` along time): Chunk time axis (`chunks={"time": 365}`)
- **Spatial operations** (`.mean(dim=["lat", "lon"])`): Chunk spatial axes (`chunks={"lat": 100, "lon": 100}`)
- **Groupby operations** (`.groupby("time.month")`): Rechunk BEFORE groupby to align with operation
- **Memory estimation**: `da.nbytes / (1024**3)` for GB; ensure chunks fit in RAM
- **Dask dashboard**: Use `from dask.diagnostics import ProgressBar; ProgressBar().register()` for monitoring

---

### **Earth Science Edge Cases (MANDATORY Handling)**

#### **1. CRS Mismatches (Runtime Detection)**
**Problem**: Watershed geometries (UTM), point data (WGS84), rasters (varied) cause silent failures.

**Solution Pattern:**
```python
# MANDATORY: CRS validation in spatial operations
def spatial_join_validated(
    left_gdf: gpd.GeoDataFrame,
    right_gdf: gpd.GeoDataFrame,
    target_crs: str = "EPSG:4326",
    logger: logging.Logger | None = None,
) -> gpd.GeoDataFrame:
    """Spatial join with mandatory CRS validation."""
    if left_gdf.crs is None or right_gdf.crs is None:
        raise ValueError("Both GeoDataFrames must have defined CRS")
    
    if str(left_gdf.crs) != target_crs:
        if logger:
            logger.info(f"Reprojecting left: {left_gdf.crs} → {target_crs}")
        left_gdf = left_gdf.to_crs(target_crs)
    
    if str(right_gdf.crs) != target_crs:
        if logger:
            logger.info(f"Reprojecting right: {right_gdf.crs} → {target_crs}")
        right_gdf = right_gdf.to_crs(target_crs)
    
    return gpd.sjoin(left_gdf, right_gdf, how="inner", predicate="intersects")
```

#### **2. Non-Standard Time Axes**
**Problems**:
- Non-standard calendars (`360_day`, `noleap`) break `.resample()`
- Missing timezone info (naive vs. UTC-aware)
- Irregular time steps in observational data

**Solution**: Use `validate_temporal_axis()` function (defined above) in ALL loaders.

#### **3. Discharge Data Quality Issues**
**Problems**:
- Negative discharge (sensor errors)
- Rating curve discontinuities (station moves)
- Ice-affected periods (Russian basins—common in winter)
- Dam operations (non-stationary regimes)

**Solution Pattern:**
```python
def clean_discharge_data(
    da: xr.DataArray,
    remove_negative: bool = True,
    flag_ice_affected: bool = True,
    temp_threshold: float = 0.0,  # °C
    temp_da: xr.DataArray | None = None,
    logger: logging.Logger | None = None,
) -> xr.DataArray:
    """Clean discharge data with optional ice-period flagging.
    
    Args:
        da: Discharge DataArray (m³/s)
        remove_negative: Mask negative values as NaN
        flag_ice_affected: Add 'ice_affected' attribute for sub-zero periods
        temp_threshold: Temperature below which ice is assumed (°C)
        temp_da: Temperature DataArray for ice detection
        logger: Logger for cleaning operations
        
    Returns:
        Cleaned discharge DataArray with quality flags
    """
    cleaned = da.copy()
    
    if remove_negative:
        n_negative = (cleaned < 0).sum().item()
        if n_negative > 0:
            if logger:
                logger.warning(f"Masking {n_negative} negative discharge values")
            cleaned = cleaned.where(cleaned >= 0)
    
    # Flag ice-affected periods
    if flag_ice_affected and temp_da is not None:
        ice_mask = temp_da < temp_threshold
        cleaned.attrs["ice_affected_days"] = ice_mask.sum().item()
        if logger:
            logger.info(
                f"Flagged {ice_mask.sum().item()} ice-affected days "
                f"(T < {temp_threshold}°C)"
            )
    
    return cleaned
```

#### **4. Rating Curve Discontinuities**
**Detection**: Sudden discharge jumps not explained by precipitation.

**Solution**:
```python
def detect_rating_curve_shifts(
    discharge: xr.DataArray,
    window: int = 30,  # days
    threshold: float = 3.0,  # standard deviations
) -> xr.DataArray:
    """Detect abrupt discharge shifts indicating rating curve changes."""
    # Rolling z-score of discharge changes
    discharge_diff = discharge.diff("time")
    rolling_mean = discharge_diff.rolling(time=window, center=True).mean()
    rolling_std = discharge_diff.rolling(time=window, center=True).std()
    
    z_scores = (discharge_diff - rolling_mean) / (rolling_std + 1e-8)
    outliers = np.abs(z_scores) > threshold
    
    return outliers  # Boolean mask of suspected rating curve shifts
```

---

## Reproducibility & validation
- **Seed all RNGs**: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`.
- Document residual stochasticity (e.g., GPU non-determinism, parallel reduction order).
- **Input validation**: Raise `ValueError`/`TypeError` early with actionable messages.
- Exception chaining: `raise NewError("context") from original_error`.
- **Unit tests**: Fast, deterministic; avoid network/disk I/O unless via `tmp_path` fixtures.
- Integration tests: Use small synthetic datasets in `test/data/`.

## Code style best practices
**Imports (Ruff-isort compliant):**
```python
# Standard library
import logging
import sys
from pathlib import Path
from typing import Literal

# Third-party
import numpy as np
import pandas as pd
import xarray as xr

# First-party (project-specific)
from floodforecast.models import gr4j  # Update to actual package name
from floodforecast.utils.metrics import nse, kge
```

**Typing examples:**
```python
from typing import Literal
import numpy as np
import xarray as xr

def calculate_pet(
    temp: xr.DataArray,  # °C
    latitude: float,  # degrees
    method: Literal["oudin", "hargreaves"] = "oudin",
) -> xr.DataArray:  # mm/day
    """Compute potential evapotranspiration."""
    ...
```

**Logging patterns (MANDATORY enforcement):**

**In `src/` modules** (use `setup_logger` with function context):
```python
from src.utils.logger import setup_logger

def calibrate_model(params: dict[str, float]) -> float:
    """Calibrate hydrological model parameters.
    
    Args:
        params: Model parameters to calibrate
        
    Returns:
        NSE score for calibrated model
    """
    # MANDATORY: Use setup_logger with function name for context tracing
    logger = setup_logger("calibrate_model")
    
    logger.info("Starting calibration with params: %s", params)
    try:
        result = optimize(params)
        logger.info("Calibration complete: NSE=%.3f", result.nse)
        return result.nse
    except ValueError as e:
        logger.error("Calibration failed: %s", e)
        raise
```

**In notebooks** (use `setup_logger` with notebook name in Cell 2):
```python
# Cell 2: Configuration
from src.utils.logger import setup_logger

logger = setup_logger(
    "chapter_one_clustering",
    log_file="../logs/chapter_one.log",
    level="INFO"
)
logger.info("Notebook initialized")
```

**FORBIDDEN patterns** (will fail code review):
```python
# ❌ NEVER use logging.getLogger() directly
import logging
logger = logging.getLogger(__name__)  # WRONG

# ❌ NEVER use print() in src/**
print("Processing data...")  # LINT VIOLATION (T20)

# ❌ NEVER use basicConfig
logging.basicConfig(level=logging.INFO)  # Breaks centralized config
```

**Rationale**: `setup_logger` provides:
- Emoji-enhanced console output for rapid triage
- Automatic rotating file handlers (10MB rotation)
- Function context injection (`func_ctx` field) for traceability
- Jupyter detection with auto-coloring
- Safe argument handling (prevents `%`-format crashes)
- Idempotent initialization (no duplicate handlers)