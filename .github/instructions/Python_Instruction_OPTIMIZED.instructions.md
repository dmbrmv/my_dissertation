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
In order to run python use conda env: conda activate geo

## Notebook philosophy: marimo-style architecture
**CRITICAL: Migrating from Jupyter (.ipynb) to marimo-style reactive notebooks.**

- **Prefer marimo notebooks** (`.py` format with `@app.cell` decorators) over `.ipynb`.
- **Notebooks are for visualization and evaluation ONLY** — not computation.
- **No heavy calculations in notebook cells**: All data processing, model training, 
  feature engineering, and complex computations belong in dedicated modules under `src/`.
- Notebook cells should **only**:
  1. Import pre-computed results from `src/` functions.
  2. Call lightweight visualization/plotting functions.
  3. Display metrics, tables, or summary statistics.
- **Organizational pattern**:
  ```
  src/
    └── analytics/        # Computation-heavy analysis functions
        └── chapter_one/  # e.g., basin clustering, regime analysis
        └── chapter_two/  # e.g., model calibration runners
        └── chapter_three/# e.g., forecast evaluation
  notebooks/
    └── ChapterOne.py     # marimo notebook: imports from src/analytics/chapter_one
  ```
- When creating new analysis code:
  1. Write computational logic in `src/analytics/<topic>/` with full type hints.
  2. Create marimo notebook that imports and visualizes results.
  3. Never inline multi-line loops, model training, or file I/O in notebook cells.

## marimo Reactive Notebook Rules (STRICTLY ENFORCED)
**marimo models notebooks as Directed Acyclic Graphs (DAGs) with automatic reactivity.**

### Core Constraint: One Cell, One Variable
**🚨 CRITICAL: Every global variable MUST be defined in exactly ONE cell.**

- ❌ **NEVER define the same global variable in multiple cells**
- ❌ **NEVER use the same loop iterator (`i`, `j`, `idx`) in multiple cells**
- ❌ **NEVER use the same temporary variable (`ax`, `fig`, `df`) in multiple cells**
- ✅ **Each cell exports ONLY its essential results as global variables**
- ✅ **All temporary/intermediate variables MUST use underscore prefix (`_`)**

### Variable Scoping Pattern
```python
# Cell A: Exports global variables for downstream cells
cluster_centroids = geo_scaled.groupby("cluster")[features].mean()
cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")

# Use underscore prefix for ALL temporary variables
for _idx, _cluster_id in enumerate(range(1, n_clusters + 1)):
    _centroid_values = cluster_centroids.loc[_cluster_id, :]
    _normalized = (_centroid_values - _centroid_values.min()) / _centroid_values.max()
    # Process data...

# Cell B: Can use cluster_centroids and cluster_labels (defined in Cell A)
# But MUST use different loop variable names
for _i in range(len(cluster_centroids)):
    _data = cluster_centroids.iloc[_i]
    # NEVER reuse _idx from Cell A
```

### Underscore Prefix Rules
**Use `_` prefix for ANY variable that:**
1. **Loop iterators**: `_i`, `_idx`, `_cid`, `_cluster_id`, `_row`, `_col`
2. **Temporary computation results**: `_temp`, `_result`, `_intermediate`
3. **Plotting objects**: `_fig`, `_ax`, `_axes`, `_lines`
4. **Function returns within cell**: `_short_name`, `_description`, `_values`
5. **Intermediate DataFrames/Series**: `_df_temp`, `_series_subset`
6. **Any variable used ONLY within the current cell**

### Export Pattern (What NOT to Prefix)
**Only these should be global (no underscore):**
```python
# ✅ Final results needed by downstream cells
cluster_analysis_df = pd.DataFrame(...)
cluster_names = pd.Series(...)
cluster_display_names = pd.Series(...)
fig_final = plt.figure(...)

# ❌ Don't export temporary computation variables
# Use _temp_df, _intermediate_result, etc.
```

### Common Pitfalls & Solutions
```python
# ❌ BAD: Multiple cells define 'ax'
# Cell 1
ax = plt.subplot()

# Cell 2  
ax = plt.subplot()  # ERROR: 'ax' redefined

# ✅ GOOD: Use underscore prefix for plotting objects
# Cell 1
_ax1 = plt.subplot()
fig_cluster_map = plt.gcf()  # Export only the figure

# Cell 2
_ax2 = plt.subplot()
fig_comparison = plt.gcf()
```

```python
# ❌ BAD: Same loop variable across cells
# Cell 1
for cid in range(n_clusters):
    process(cid)

# Cell 2
for cid in range(n_clusters):  # ERROR: 'cid' redefined
    visualize(cid)

# ✅ GOOD: Use underscore prefix for ALL loop variables
# Cell 1
for _cid in range(n_clusters):
    process(_cid)

# Cell 2
for _cid in range(n_clusters):  # OK: both use underscore
    visualize(_cid)
```

### Reactivity Flow
```python
# Cell A defines: data_scaled, cluster_labels
data_scaled = (data - data.min()) / (data.max() - data.min())
cluster_labels = kmeans.fit_predict(data_scaled)

# Cell B references data_scaled → Cell B automatically re-runs when Cell A changes
cluster_centers = data_scaled.groupby(cluster_labels).mean()

# Cell C references cluster_centers → Cell C re-runs when Cell B changes
fig_viz = plot_clusters(cluster_centers)
```

### Functions in Cells
```python
# ✅ Define helper functions but keep loop variables local
def classify_cluster(centroid: pd.Series) -> str:
    """Helper function for classification."""
    # Internal variables can be named normally
    dominant_feature = centroid.idxmax()
    return f"Cluster dominated by {dominant_feature}"

# When using the function, still use underscore for loop vars
cluster_classifications = {}
for _cid, _centroid in cluster_centroids.iterrows():
    cluster_classifications[_cid] = classify_cluster(_centroid)

# Export only the result
cluster_names = pd.Series(cluster_classifications)
```

### Quick Reference Checklist
Before writing marimo cell code:
1. ✅ What global variables does this cell EXPORT? (no underscore)
2. ✅ All loop iterators use underscore prefix? (`_i`, `_idx`, `_cid`)
3. ✅ All plotting objects use underscore prefix? (`_fig`, `_ax`, `_axes`)
4. ✅ All temporary variables use underscore prefix? (`_temp`, `_result`)
5. ✅ No variable name conflicts with other cells?
6. ✅ Cell exports only ESSENTIAL results needed downstream?

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
```
src/
  ├── geometry/      # Watershed delineation, spatial ops
  ├── grids/         # Meteorological grid processing (NetCDF/GDAL)
  ├── loaders/       # Data ingestion (discharge, static attributes)
  ├── models/        # HBV, GR4J, LSTM implementations
  ├── readers/       # Format-specific readers (HydroATLAS, AIS, etc.)
  └── utils/         # Metrics, logging, validation
archive/             # Legacy experiments (lenient linting)
scripts/             # CLI tools (prints allowed)
test/                # Unit tests (assert allowed)
notebooks/           # Analysis notebooks (prints + minimal docs OK)
```

**Code organization:**
- Functions ≤40 LOC; single responsibility; isolate I/O from logic.
- Layered workflow: **ingestion → preprocessing → modeling → evaluation**.
- Config via `TypedDict`/`dataclasses`; avoid hidden globals or module-level state.
- CLI scripts: use `if __name__ == "__main__":` guard; leverage `argparse`/`click`.
- Never log secrets; sanitize paths/env vars in logs.

## Hydrology & Earth science specifics
**Data structures:**
- **Multi-dimensional gridded data**: `xarray.Dataset`/`DataArray` with labeled dims 
  (time, lat, lon, basin_id).
- **Tabular data**: Prefer `polars.DataFrame` for large CSVs; pandas for small/legacy data.
- **Large/OOM datasets**: Use Dask-backed xarray (`chunks`) or Zarr storage.
- **Geospatial**: `geopandas.GeoDataFrame` + `shapely` for vectors; `rasterio`/`rioxarray` 
  for rasters.

**CF conventions & metadata:**
- NetCDF files: follow CF-1.8 conventions.
  - Required attributes: `units`, `long_name`, `standard_name` (where applicable).
  - Coordinate variables: `time` (CF datetime), `lat`, `lon` (or `x`, `y` for projected CRS).
  - CRS: store in `crs` variable or coordinate attribute (`spatial_ref`).
- **Always document units** in docstrings and xarray attributes (e.g., mm/day, m³/s, °C).
- Validate physical bounds: precip ≥ 0, temperature reasonable, discharge ≥ 0.
- Use `NaN` for missing data; avoid sentinel values (-9999, 999).

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

**Logging (not print):**
```python
import logging

logger = logging.getLogger(__name__)

def calibrate_model(params: dict[str, float]) -> float:
    logger.info("Starting calibration with params: %s", params)
    try:
        result = optimize(params)
        logger.info("Calibration complete: NSE=%.3f", result.nse)
        return result.nse
    except ValueError as e:
        logger.error("Calibration failed: %s", e)
        raise
```

## Operational checklist (pre-commit mental model)
1. ✅ Imports sorted (Ruff-isort; first-party correct).
2. ✅ No `print()` in `src/**` (T20); logging used.
3. ✅ No shadowed builtins (A).
4. ✅ Cyclomatic complexity ≤ 10 (C90).
5. ✅ Public functions type-hinted with return types.
6. ✅ Docstrings present for public API (Google style).
7. ✅ Physical units documented in docstrings + xarray attrs.
8. ✅ Input validation with specific exceptions.
9. ✅ Pyright "basic" passes (no unresolved imports).
10. ✅ Tests added/updated for new logic.

## Absolute guardrails
- **No implicit relative paths** — always use `Path(__file__).resolve()` or config.
- **No hidden singletons** or module-level mutable state.
- **No deep nested try/except** (max 2 levels); prefer early validation.
- **No ad-hoc `time.sleep()`** — use proper async or exponential backoff.
- **No hardcoded credentials** — use environment variables or secrets management.
- **No CSV for large datasets** — use NetCDF/Zarr/Parquet.
- **No unsolicited documentation files** — no summary.md, report.md, or analysis docs 
  unless user explicitly requests them.

## When uncertain
State assumptions concisely before code, but **never ship lint violations**. If a 
rule conflict arises, consult pyproject.toml and prioritize correctness > brevity.
