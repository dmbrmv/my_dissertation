# Conceptual Hydrological Models

This document provides detailed information about the conceptual rainfall-runoff models implemented in this project: **HBV** and **GR4J + CemaNeige**.

## Table of Contents

- [Why Conceptual Models?](#why-conceptual-models)
- [HBV Model](#hbv-model)
- [GR4J + CemaNeige Model](#gr4j--cemaneige-model)
- [Calibration Best Practices](#calibration-best-practices)
- [Comparison and Selection Guide](#comparison-and-selection-guide)

---

## Why Conceptual Models?

Conceptual hydrological models offer several advantages for this research:

### ✅ Advantages

1. **Physical Interpretability**: Parameters have physical meaning (field capacity, recession coefficients, etc.)
2. **Data Efficiency**: Require only meteorological inputs (precipitation, temperature)
3. **Computational Speed**: Fast simulation enables extensive calibration trials
4. **Process Representation**: Explicitly model snow, soil moisture, and groundwater
5. **Climate Change Transferability**: Physical parameters may be more robust under non-stationary conditions

### ⚠️ Limitations

1. **Equifinality**: Multiple parameter sets can produce similar results
2. **Calibration Dependent**: Performance highly dependent on optimization
3. **Structural Uncertainty**: Fixed process representation may not suit all basins
4. **Lumped Approach**: Spatial heterogeneity averaged within catchment

### 🎯 Best Use Cases

- Snow-dominated catchments (HBV)
- Data-limited scenarios (GR4J)
- Need for physical interpretation
- Climate change impact assessments
- Benchmarking against machine learning models

---

## HBV Model

### Model History and Development

**Developed**: 1976 by Sten Bergström at SMHI (Swedish Meteorological and Hydrological Institute)

**Original Purpose**: Operational flood forecasting in Swedish catchments

**Key Publications**:
- Bergström, S. (1976). *Development and application of a conceptual runoff model for Scandinavian catchments*. SMHI Reports RHO 7.
- Seibert, J., & Vis, M. J. P. (2012). *Teaching hydrological modeling with a user-friendly catchment-runoff-model software package*. Hydrology and Earth System Sciences, 16(9), 3315-3325.

### Model Structure

HBV consists of four main routines operating in sequence:

#### 1. Snow Routine

Separates precipitation into snow and rain based on threshold temperature:

$$
\text{Snow} = 
\begin{cases}
P & \text{if } T < T_t \\
0 & \text{otherwise}
\end{cases}
$$

$$
\text{Rain} = 
\begin{cases}
0 & \text{if } T < T_t \\
P & \text{otherwise}
\end{cases}
$$

**Snowmelt** (degree-day approach):

$$
M = \text{CFMAX} \cdot (T - T_t) \quad \text{if } T > T_t
$$

**Refreezing**:

$$
R = \text{CFR} \cdot \text{CFMAX} \cdot (T_t - T) \quad \text{if } T < T_t
$$

**Water holding capacity**:

$$
\text{Meltwater}_{\text{release}} = \max(0, \text{Meltwater} - \text{CWH} \cdot \text{Snowpack})
$$

#### 2. Soil Moisture Routine

**Soil moisture accounting**:

$$
\frac{dSM}{dt} = P_{\text{eff}} - EA - R_{\text{gen}}
$$

Where:
- $SM$ = soil moisture storage (mm)
- $P_{\text{eff}}$ = effective precipitation (rain + snowmelt)
- $EA$ = actual evapotranspiration
- $R_{\text{gen}}$ = runoff generation

**Actual evapotranspiration**:

$$
EA = EP \cdot \min\left(1, \frac{SM}{LP \cdot FC}\right)
$$

**Runoff generation** (beta function):

$$
R_{\text{gen}} = P_{\text{eff}} \cdot \left(\frac{SM}{FC}\right)^{\beta}
$$

#### 3. Response (Groundwater) Routine

Runoff is routed through two zones:

**Upper zone** (fast response):

$$
Q_0 = K_0 \cdot \max(0, SUZ - UZL)
$$

$$
Q_1 = K_1 \cdot SUZ
$$

**Lower zone** (baseflow):

$$
Q_2 = K_2 \cdot SLZ
$$

**Percolation** from upper to lower zone:

$$
\text{Perc} = \min(\text{PERC}, SUZ)
$$

**Total discharge**:

$$
Q_{\text{total}} = Q_0 + Q_1 + Q_2
$$

#### 4. Routing Routine (Optional)

Optional Butterworth filter for channel routing:

$$
Q_{\text{routed}} = \text{Butterworth}(Q_{\text{total}}, \text{MAXBAS})
$$

### Parameters

HBV has **14 parameters** (12 hydrological + 2 routing):

| Parameter | Description | Unit | Typical Range | Physical Meaning |
|-----------|-------------|------|---------------|------------------|
| **Snow Routine** |||||
| `TT` | Threshold temperature | °C | -2 to 2 | Snow/rain transition |
| `CFMAX` | Degree-day factor | mm/°C/day | 1 to 10 | Snowmelt rate |
| `CFR` | Refreezing coefficient | - | 0 to 0.1 | Meltwater refreezing |
| `CWH` | Water holding capacity | - | 0 to 0.2 | Liquid water in snowpack |
| **Soil Routine** |||||
| `BETA` | Shape coefficient | - | 1 to 6 | Runoff generation non-linearity |
| `FC` | Field capacity | mm | 50 to 500 | Maximum soil moisture |
| `LP` | Limit for PET | - | 0.3 to 1.0 | ET reduction threshold |
| **Response Routine** |||||
| `K0` | Recession coefficient 0 | 1/day | 0.05 to 0.5 | Fast runoff |
| `K1` | Recession coefficient 1 | 1/day | 0.01 to 0.3 | Medium runoff |
| `K2` | Recession coefficient 2 | 1/day | 0.001 to 0.1 | Baseflow |
| `UZL` | Threshold for fast runoff | mm | 0 to 100 | Upper zone limit |
| `PERC` | Percolation rate | mm/day | 0 to 5 | Recharge to groundwater |
| **Routing** |||||
| `MAXBAS` | Routing length | days | 1 to 7 | Channel routing time |

### Implementation Details

**Location**: `src/models/hbv/`

**Core files**:
- `hbv.py` - Main simulation function
- `hbv_optuna.py` - Optuna-based calibration
- `hbv_calibrator.py` - Multi-objective framework
- `parallel.py` - Parallel processing utilities
- `pareto.py` - Pareto front analysis

**Training script**: `scripts/hbv_train.py`

### Code Structure

```python
def simulation(
    data: pd.DataFrame,
    params: tuple[float, ...],
    return_components: bool = False,
) -> np.ndarray:
    """Run HBV rainfall-runoff simulation.
    
    Args:
        data: DataFrame with columns [prec, temp, evap]
        params: Tuple of 14 parameters (or 12 without routing)
        return_components: If True, return all state variables
    
    Returns:
        Simulated discharge time series (or dict of components)
    """
    # Extract parameters
    tt, cfmax, cfr, cwh = params[:4]  # Snow
    beta, fc, lp = params[4:7]        # Soil
    k0, k1, k2, uzl, perc = params[7:12]  # Response
    
    # Initialize state variables
    snowpack = meltwater = 0.0
    soil_moisture = fc / 2  # Start at half capacity
    upper_zone = lower_zone = 0.0
    
    # Main simulation loop
    for i in range(len(data)):
        # 1. Snow routine
        snowpack, meltwater = _snow_routine(...)
        
        # 2. Soil routine
        soil_moisture, runoff_gen = _soil_routine(...)
        
        # 3. Response routine
        upper_zone, lower_zone, q = _groundwater_routine(...)
        
        discharge[i] = q
    
    # 4. Optional routing
    if maxbas > 1:
        discharge = _apply_routing(discharge, maxbas)
    
    return discharge
```

### Calibration Strategy

**Multi-objective optimization** using Optuna:

```python
def objective(trial):
    # Sample parameters
    params = (
        trial.suggest_float("TT", -2.5, 2.5),
        trial.suggest_float("CFMAX", 0.5, 10.0),
        # ... all 14 parameters
    )
    
    # Run simulation
    sim = simulation(train_data, params)
    
    # Calculate metrics
    kge = kling_gupta_efficiency(obs, sim)
    nse = nash_sutcliffe(obs, sim)
    log_nse = nash_sutcliffe(np.log(obs + 0.01), np.log(sim + 0.01))
    pbias = percent_bias(obs, sim)
    
    # Multi-objective composite
    score = 0.5 * kge + 0.3 * nse + 0.15 * log_nse - 0.05 * abs(pbias)
    
    return score
```

**Warm-up period**: 

⚠️ **Critical**: Use 2-3 years warm-up for:
- Snowpack initialization (especially in Russia)
- Soil moisture equilibration
- Groundwater zone stabilization

### Performance Characteristics

**Median performance** (996 Russian basins):
- NSE: 0.65
- KGE: 0.68
- PBIAS: -3.5%
- RMSE: 1.45 mm/day

**Best performing regions**:
- Snow-dominated catchments (Arctic, mountain)
- Medium-sized basins (500-10,000 km²)
- Humid climates with regular snowfall

**Challenges**:
- Semi-arid regions (limited soil moisture dynamics)
- Very small basins (<100 km²) - spatial heterogeneity
- Highly regulated rivers (no infrastructure representation)

### Example Usage

```python
from pathlib import Path
from src.models.hbv.hbv_optuna import run_optimization

# Run HBV calibration
results = run_optimization(
    gauge_id="70158",
    gauge_data_dir=Path("data/ws_related_meteo"),
    static_attributes_path=Path("data/attributes/hydro_atlas_cis_camels.csv"),
    output_dir=Path("data/optimization/hbv_results"),
    n_trials=200,
    timeout=3600,
    n_jobs=4
)

# Results saved to:
# - best_parameters.json (optimized params)
# - test_metrics.json (performance on test period)
# - optimization_study.pkl (Optuna study for analysis)
```

### Tips for Successful Calibration

1. **Start with reasonable ranges**: Use typical values from literature
2. **Use warm-up period**: Don't evaluate first 2-3 years
3. **Check snow balance**: Ensure realistic snowpack accumulation/melt
4. **Validate components**: Plot snowpack, soil moisture, baseflow
5. **Multi-objective**: Balance high flows, low flows, and volume

---

## GR4J + CemaNeige Model

### Model History and Development

**GR4J Developed**: 1989-2003 by Claude Michel, Charles Perrin, and colleagues at INRAE (formerly IRSTEA/Cemagref), France

**CemaNeige Developed**: 2014 by Audrey Valéry at INRAE

**Philosophy**: Parsimonious model with only 4 parameters, balancing simplicity and performance

**Key Publications**:
- Perrin, C., Michel, C., & Andréassian, V. (2003). *Improvement of a parsimonious model for streamflow simulation*. Journal of Hydrology, 279(1-4), 275-289.
- Valéry, A., Andréassian, V., & Perrin, C. (2014). *"As simple as possible but not simpler": What is useful in a temperature-based snow-accounting routine?*. Journal of Hydrology, 517, 1288-1299.

### Model Structure

#### CemaNeige Snow Module

**Elevation band approach**: Catchment divided into 10 equal-area elevation bands

**For each elevation band**:

1. **Temperature adjustment**:

$$
T_{\text{band}} = T_{\text{obs}} - 0.0065 \cdot (Z_{\text{band}} - Z_{\text{gauge}})
$$

2. **Snow accumulation**:

$$
\Delta S_{\text{snow}} = 
\begin{cases}
P & \text{if } T_{\text{band}} < 0°C \\
0 & \text{otherwise}
\end{cases}
$$

3. **Snowmelt**:

$$
M = \text{CTG} \cdot T_{\text{band}} \quad \text{if } T_{\text{band}} > 0°C
$$

4. **Thermal state** (tracks cold content):

$$
G = (1 - K_f) \cdot G + K_f \cdot \min(T_{\text{band}}, 0)
$$

5. **Liquid water fraction**:

Meltwater must first warm the snowpack before contributing to runoff.

**Aggregated output**: Weighted average of melt across all elevation bands

#### GR4J Core Model

**1. Production Store** (soil moisture accounting):

Net rainfall/snowmelt:

$$
P_n = P - E
$$

Production store ($S$) evolution:

$$
\frac{dS}{dt} = P_s - P_r
$$

Where:
- $P_s$ = fraction stored in production
- $P_r$ = fraction routed to runoff

If $P_n \geq 0$ (wetting):

$$
P_s = \frac{X_1 \cdot (1 - (S/X_1)^2) \cdot \tanh(P_n/X_1)}{1 + (S/X_1) \cdot \tanh(P_n/X_1)}
$$

If $P_n < 0$ (drying):

$$
P_s = \frac{S \cdot (2 - S/X_1) \cdot \tanh(-P_n/X_1)}{1 + (1 - S/X_1) \cdot \tanh(-P_n/X_1)}
$$

**2. Percolation**:

$$
\text{Perc} = S \cdot \left(1 - \left[1 + \left(\frac{4S}{9X_1}\right)^4\right]^{-1/4}\right)
$$

**3. Routing**:

Effective rainfall split into 90% slow flow (UH1) and 10% fast flow (UH2):

$$
Q_9 = 0.9 \cdot (P_n - P_s + \text{Perc})
$$

$$
Q_1 = 0.1 \cdot (P_n - P_s + \text{Perc})
$$

**Unit Hydrographs** (SH1 and SH2 derived from $X_4$):

$$
\text{UH1} = \text{function of } X_4
$$

$$
\text{UH2} = \text{function of } 2 \cdot X_4
$$

**4. Routing Store** ($R$):

$$
\frac{dR}{dt} = Q_9(t) + F - Q_r
$$

Groundwater exchange $F$:

$$
F = X_2 \cdot \left(\frac{R}{X_3}\right)^{3.5}
$$

Outflow $Q_r$:

$$
Q_r = R \cdot \left[1 - \left(1 + \left(\frac{R}{X_3}\right)^4\right)^{-1/4}\right]
$$

**5. Total discharge**:

$$
Q_{\text{total}} = Q_r + \max(0, Q_1 + F)
$$

### Parameters

**GR4J (4 parameters)**:

| Parameter | Description | Unit | Typical Range | Physical Meaning |
|-----------|-------------|------|---------------|------------------|
| `X1` | Production store capacity | mm | 100-1200 | Maximum soil moisture |
| `X2` | Groundwater exchange | mm/day | -5 to 3 | Imports (+) / exports (-) |
| `X3` | Routing store capacity | mm | 20-300 | Groundwater storage |
| `X4` | Unit hydrograph time base | days | 1.1-2.9 | Catchment response time |

**CemaNeige (2 parameters)**:

| Parameter | Description | Unit | Typical Range | Physical Meaning |
|-----------|-------------|------|---------------|------------------|
| `CTG` | Degree-day melt factor | mm/°C/day | 0-1000 | Snowmelt rate (scaled internally) |
| `Kf` | Thermal exchange coefficient | - | 0-10 | Snowpack thermal inertia |

**Total**: 6 parameters (4 + 2)

### Implementation Details

**Location**: `src/models/gr4j/`

**Core files**:
- `model.py` - GR4J simulation
- `cema_neige.py` - Snow module
- `pet.py` - Oudin PET calculation
- `gr4j_optuna.py` - Calibration framework
- `parallel.py` - Parallel processing
- `pareto.py` - Multi-objective analysis

**Training script**: `scripts/gr4j_train.py`

### Code Structure

```python
def run_gr4j_cema(
    precip: np.ndarray,
    temp: np.ndarray,
    pet: np.ndarray,
    params: tuple[float, ...],
    elevation_bands: np.ndarray,
) -> np.ndarray:
    """Run GR4J with CemaNeige snow module.
    
    Args:
        precip: Daily precipitation (mm)
        temp: Daily temperature (°C)
        pet: Daily PET (mm)
        params: (X1, X2, X3, X4, CTG, Kf)
        elevation_bands: 10 elevation values (m)
    
    Returns:
        Simulated discharge (mm/day)
    """
    # Extract parameters
    x1, x2, x3, x4, ctg, kf = params
    
    # Initialize stores
    S = x1 * 0.5  # Production store
    R = x3 * 0.5  # Routing store
    
    # Snow state for each band
    snowpack = np.zeros(10)
    thermal_state = np.zeros(10)
    
    for t in range(len(precip)):
        # 1. CemaNeige snow module
        liquid_water = cema_neige_step(
            precip[t], temp[t], elevation_bands,
            snowpack, thermal_state, ctg, kf
        )
        
        # 2. GR4J production
        S, P_r = production_store(S, liquid_water - pet[t], x1)
        
        # 3. GR4J routing
        R, Q = routing_store(R, P_r, x2, x3, x4)
        
        discharge[t] = Q
    
    return discharge
```

### Calibration Strategy

**Multi-objective optimization**:

```python
def objective(trial):
    # GR4J parameters
    X1 = trial.suggest_float("X1", 100, 1200)
    X2 = trial.suggest_float("X2", -5, 3)
    X3 = trial.suggest_float("X3", 20, 300)
    X4 = trial.suggest_float("X4", 1.1, 2.9)
    
    # CemaNeige parameters
    CTG = trial.suggest_float("CTG", 0, 1000)
    Kf = trial.suggest_float("Kf", 0, 10)
    
    # Run simulation
    sim = run_gr4j_cema(precip, temp, pet, 
                         (X1, X2, X3, X4, CTG, Kf),
                         elevation_bands)
    
    # Multi-objective score
    kge = kling_gupta_efficiency(obs, sim)
    nse = nash_sutcliffe(obs, sim)
    log_nse = nash_sutcliffe(np.log(obs + 0.01), np.log(sim + 0.01))
    pbias = abs(percent_bias(obs, sim))
    
    return 0.5 * kge + 0.3 * nse + 0.15 * log_nse - 0.05 * pbias
```

**Critical requirements**:

⚠️ **Warm-up period**: 2-3 years for snowpack and soil moisture initialization

⚠️ **PET calculation**: Use Oudin formula for consistency

⚠️ **Elevation data**: Requires mean elevation and hypsometric curve

### Performance Characteristics

**Median performance** (996 Russian basins):
- NSE: 0.61
- KGE: 0.64
- PBIAS: -4.2%
- RMSE: 1.58 mm/day

**Best performing regions**:
- Temperate climates
- Medium-elevation catchments
- Basins with moderate snow influence

**Strengths**:
- Simple structure (only 4 core parameters)
- Fast computation
- Robust performance across diverse conditions
- Well-documented and widely used

**Limitations**:
- Lower performance than HBV in extreme snow conditions
- Fixed structure may not capture all processes
- No explicit baseflow separation

### Example Usage

```python
from pathlib import Path
from src.models.gr4j.gr4j_optuna import run_optimization

# Run GR4J calibration
results = run_optimization(
    gauge_id="70158",
    gauge_data_dir=Path("data/ws_related_meteo"),
    static_attributes_path=Path("data/attributes/hydro_atlas_cis_camels.csv"),
    output_dir=Path("data/optimization/gr4j_results"),
    n_trials=200,
    n_jobs=4
)

# Results include:
# - Optimized parameters (X1, X2, X3, X4, CTG, Kf)
# - Performance metrics (NSE, KGE, PBIAS, RMSE)
# - Time series of simulated discharge
```

---

## Calibration Best Practices

### 1. Warm-up Period

**Why it matters**: Initial state variables (snowpack, soil moisture, groundwater) affect simulation

**Recommendations**:
- **Minimum**: 1 year
- **Recommended**: 2-3 years for Russian basins (substantial snowpack)
- **Ideal**: 3-5 years if data available

**Implementation**:
```python
# Start calibration/validation after warm-up
warmup_end = pd.Timestamp("2010-12-31")  # 2008-2010 warm-up
calibration_start = pd.Timestamp("2011-01-01")
```

### 2. Multi-objective Optimization

**Why**: Single metrics can miss important aspects (e.g., NSE favors high flows)

**Recommended objectives**:

| Metric | Weight | Purpose |
|--------|--------|---------|
| KGE | 0.4-0.5 | Overall performance (correlation, bias, variability) |
| NSE | 0.2-0.3 | High flow accuracy |
| log-NSE | 0.15-0.2 | Low flow accuracy |
| PBIAS | 0.05-0.1 | Volume conservation |
| RMSE | 0.05 | Magnitude errors |

**Implementation**:
```python
score = (0.5 * kge + 0.3 * nse + 0.15 * log_nse 
         - 0.05 * abs(pbias) / 100)  # Normalize PBIAS
```

### 3. Parameter Constraints

**Use physically reasonable bounds**:

```python
# HBV example - tighter ranges often better
params = {
    "TT": (-1.5, 1.5),      # Narrower than (-2.5, 2.5)
    "CFMAX": (1, 8),        # Typical range for Russia
    "FC": (100, 400),       # Avoid extreme values
    "BETA": (1, 4),         # Most basins in 1-3 range
}
```

### 4. Validation Strategy

**Time-based split** (preferred):
- Calibration: 2008-2015 (8 years)
- Validation: 2016-2018 (3 years)
- Test: 2019-2020 (2 years)

**Why not random split**: Hydrological memory (snowpack, groundwater) makes random splits invalid

### 5. Diagnostic Plots

**Always check**:
1. **Hydrograph**: Observed vs. simulated discharge
2. **Flow duration curve**: Model performance across flow regime
3. **Residuals**: Temporal patterns in errors
4. **Component time series**: Snowpack, soil moisture (if available)

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Hydrograph
axes[0].plot(obs.index, obs.values, label="Observed", alpha=0.7)
axes[0].plot(sim.index, sim.values, label="Simulated", alpha=0.7)
axes[0].set_ylabel("Discharge (mm/day)")
axes[0].legend()

# Flow duration curve
obs_sorted = np.sort(obs.values)[::-1]
sim_sorted = np.sort(sim.values)[::-1]
exceedance = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted) * 100

axes[1].plot(exceedance, obs_sorted, label="Observed")
axes[1].plot(exceedance, sim_sorted, label="Simulated")
axes[1].set_xlabel("Exceedance Probability (%)")
axes[1].set_ylabel("Discharge (mm/day)")
axes[1].set_yscale("log")
axes[1].legend()

# Residuals
residuals = obs.values - sim.values
axes[2].scatter(obs.index, residuals, alpha=0.5, s=10)
axes[2].axhline(0, color='r', linestyle='--')
axes[2].set_xlabel("Date")
axes[2].set_ylabel("Residuals (mm/day)")

plt.tight_layout()
plt.savefig("diagnostics.png", dpi=300)
```

### 6. Optuna Configuration

**Recommended settings**:

```python
import optuna

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=20
    )
)

study.optimize(
    objective,
    n_trials=200,      # 100-500 depending on time budget
    timeout=3600,      # 1 hour per basin
    n_jobs=4,          # Parallel trials (if deterministic)
    show_progress_bar=True
)
```

**Tips**:
- **TPE sampler**: Better than random search for continuous parameters
- **Pruning**: Stop poor trials early (saves ~30% computation)
- **n_trials**: 100-200 usually sufficient for 4-14 parameters
- **n_jobs**: Use with caution (ensure thread safety)

---

## Comparison and Selection Guide

### When to Use HBV

✅ **Choose HBV if**:
- Basin has significant snow processes
- Need explicit snow state representation
- Working in Arctic, subarctic, or mountain regions
- Require detailed runoff component separation (baseflow, fast flow)
- Have time for 14-parameter optimization

❌ **Avoid HBV if**:
- Very limited calibration data (<5 years)
- Semi-arid basin with minimal snowfall
- Need fastest possible computation
- Working with many basins (prefer simpler GR4J)

### When to Use GR4J

✅ **Choose GR4J if**:
- Need simplest reliable model (only 4 core parameters)
- Limited data or computational resources
- Benchmarking against complex models
- Working with many basins in batch mode
- Basin has moderate snow influence (CemaNeige handles it)

❌ **Avoid GR4J if**:
- Need detailed process representation
- Basin has complex snow dynamics requiring detailed tracking
- Require explicit baseflow component

### Head-to-Head Comparison

| Aspect | HBV | GR4J + CemaNeige |
|--------|-----|-------------------|
| **Parameters** | 14 | 6 |
| **Complexity** | Medium | Low |
| **Snow representation** | Detailed (state variables) | Simplified (elevation bands) |
| **Baseflow** | Explicit (lower zone) | Implicit (routing store) |
| **Calibration time** | Higher | Lower |
| **Median NSE (Russia)** | 0.65 | 0.61 |
| **Best regions** | Snow-dominated | Temperate |
| **Interpretability** | High | Medium |
| **Computational speed** | Fast | Very fast |

### Performance by Basin Characteristics

| Basin Type | HBV Performance | GR4J Performance | Recommendation |
|------------|-----------------|------------------|----------------|
| **Arctic** | Excellent (NSE~0.70) | Good (NSE~0.63) | **HBV** |
| **Temperate** | Good (NSE~0.64) | Good (NSE~0.61) | **Either** (preference: GR4J for speed) |
| **Mountain** | Excellent (NSE~0.68) | Good (NSE~0.60) | **HBV** |
| **Small (<500 km²)** | Good (NSE~0.60) | Fair (NSE~0.55) | **HBV** |
| **Large (>5000 km²)** | Good (NSE~0.66) | Good (NSE~0.64) | **Either** |

---

## References

### HBV

1. Bergström, S. (1976). Development and application of a conceptual runoff model for Scandinavian catchments. SMHI Reports RHO 7, Norrköping.

2. Seibert, J., & Vis, M. J. P. (2012). Teaching hydrological modeling with a user-friendly catchment-runoff-model software package. Hydrology and Earth System Sciences, 16(9), 3315-3325.

3. Lindström, G., Johansson, B., Persson, M., Gardelin, M., & Bergström, S. (1997). Development and test of the distributed HBV-96 hydrological model. Journal of Hydrology, 201(1-4), 272-288.

### GR4J

1. Perrin, C., Michel, C., & Andréassian, V. (2003). Improvement of a parsimonious model for streamflow simulation. Journal of Hydrology, 279(1-4), 275-289.

2. Valéry, A., Andréassian, V., & Perrin, C. (2014). "As simple as possible but not simpler": What is useful in a temperature-based snow-accounting routine? Part 1 - Comparison of six snow accounting routines on 380 catchments. Journal of Hydrology, 517, 1166-1175.

3. Oudin, L., Hervieu, F., Michel, C., Perrin, C., Andréassian, V., Anctil, F., & Loumagne, C. (2005). Which potential evapotranspiration input for a lumped rainfall–runoff model? Part 2 - Towards a simple and efficient potential evapotranspiration model for rainfall–runoff modelling. Journal of Hydrology, 303(1-4), 290-306.

### Multi-objective Calibration

1. Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling. Journal of Hydrology, 377(1-2), 80-91.

2. Efstratiadis, A., & Koutsoyiannis, D. (2010). One decade of multi-objective calibration approaches in hydrological modelling: a review. Hydrological Sciences Journal, 55(1), 58-78.

---

## Quick Reference

### HBV Training

```bash
# Activate environment
conda activate Geo

# Run training for all gauges
python scripts/hbv_train.py

# Results in: data/optimization/hbv_results/
```

### GR4J Training

```bash
# Activate environment
conda activate Geo

# Run training
python scripts/gr4j_train.py

# Results in: data/optimization/gr4j_results/
```

### Load and Use Calibrated Model

```python
import json
import pandas as pd
from src.models.hbv.hbv import simulation

# Load parameters
with open("data/optimization/hbv_results/70158/best_parameters.json") as f:
    params = json.load(f)["parameters"]

# Load data
data = pd.read_csv("my_basin_data.csv")  # Must have: prec, temp, evap

# Run simulation
discharge = simulation(data, params)
```
