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
In order to run python use conda env: conda activate camels_ru

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
