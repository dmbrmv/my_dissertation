# Hydrological Modeling and Precipitation Data Comparison for Russian River Basins

## Overview

This repository contains the complete codebase for a PhD dissertation focused on hydrological modeling and precipitation data comparison across Russian river basins using deep learning methods. The research addresses critical challenges in water resource management under climate change conditions and the degradation of hydrometeorological observation networks in the Russian Federation.

### Research Context and Motivation

Climate change combined with the degradation of the hydrometeorological observation network in the Russian Federation necessitates the development of reliable methods for river flow modeling. Water resource management is impossible without reliable flow modeling tools, especially in sparsely populated or remote areas where ground-based observations have been declining for decades.

Small and medium rivers (catchment areas up to 50,000 km²) are most vulnerable in this context, where observational data volumes are often insufficient for applying physically-based models or regression relationships that require long and continuous observation series. The deterioration of observational infrastructure, climate change, and spatial heterogeneity of flow formation conditions make it necessary to transition to new methodological approaches.

## Research Objectives

The main goal of this research is to develop approaches for modeling daily average flow of small and medium rivers in Russia using deep machine learning methods, including neural network models based on LSTM architecture, in the context of diverse natural conditions and limited observational information.

### Key Research Questions

1. **Precipitation Data Evaluation**: How do different precipitation datasets (ERA5, ERA5-Land, MSWEP, GPCP, IMERG) perform across Russian river basins?
2. **Model Comparison**: How do neural networks (LSTM, GRU) compare with conceptual models (HBV, GR4J) and machine learning approaches (Random Forest)?
3. **Ungauged Basin Prediction**: Can LSTM models transfer knowledge from gauged to ungauged basins using physiographic characteristics?
4. **Hidden State Interpretation**: Can LSTM internal states reveal unobserved hydrometeorological characteristics like evaporation, snow storage, and baseflow?

## Scientific Novelty

1. **Comprehensive Database**: Created a digital database of hydrological, meteorological, and physiographic characteristics for small and medium river watersheds in the Russian Federation.

2. **Regional Clustering**: Conducted objective analysis and identification of regions (clusters) of small and medium watersheds distinguished by physiographic features and water regime characteristics.

3. **Deep Learning Application**: First comprehensive application of deep machine learning methods for river flow modeling across the Russian Federation, with comparison to physically-based and statistical models.

4. **Precipitation Dataset Assessment**: Comprehensive evaluation of atmospheric model-based data sources and their impact on modeling quality across different regions of Russia.

5. **Hidden State Analysis**: Demonstrated that LSTM internal states can be used to recover unobserved hydrometeorological characteristics such as moisture storage, snow depth, evaporation, and groundwater flow.

## Project Structure

```
├── conclusions/                 # Results analysis and metrics calculation
├── conceptual_runs/            # Conceptual model implementations (HBV, GR4J)
├── data_builders/              # Data preprocessing and watershed delineation
├── geometry_creator/           # Watershed geometry generation
├── hydroatlas_parser/          # Static catchment attributes extraction
├── meteo_grids_parser/         # Meteorological data processing
├── meteo_station_comparison/   # Station vs reanalysis comparison
├── metric_viewer/              # Hydrological metrics visualization
├── neural_forecast/            # Neural network implementations
├── test_meteo_input/           # Precipitation input testing
└── visualizations/             # Result visualization scripts
```

## Key Components

### 1. Data Processing Pipeline

#### Watershed Delineation (`data_builders/`, `geometry_creator/`)

- Automated watershed boundary extraction using flow direction grids
- Point-based catchment delineation with area validation
- River network extraction at multiple scales

#### Meteorological Data Processing (`meteo_grids_parser/`)

- Grid-based precipitation and temperature extraction
- Weighted area calculations for irregular catchments
- Multiple reanalysis dataset support:
  - **ERA5**: 0.25° resolution
  - **ERA5-Land**: 0.1° resolution  
  - **MSWEP**: 0.1° resolution
  - **GPCP**: 0.25° resolution
  - **IMERG**: 0.1° resolution

#### Static Attributes (`hydroatlas_parser/`)

- HydroATLAS integration for catchment characteristics
- Weighted extraction of physiographic, climatic, and land cover variables
- Basin area normalization and quality control

### 2. Hydrological Models

#### Neural Networks (`neural_forecast/`, `test_meteo_input/`)

- **LSTM/GRU Models**: Regional and single-basin approaches
- **Input Features**: Precipitation, temperature, static catchment attributes
- **Training Strategy**: Temporal split (2009-2016 train, 2017-2018 validation, 2019-2020 test)
- **Frameworks**: NeuralHydrology library integration

#### Conceptual Models (`conceptual_runs/`)

- **HBV Model**: Snow accumulation/melt, soil moisture, runoff generation
- **GR4J Model**: Production store, routing store, unit hydrographs
- **Calibration**: Maximum Likelihood Estimation (MLE)
- **Validation**: Multi-objective optimization (NSE, KGE, RMSE)

#### Machine Learning (`conclusions/`)

- **Random Forest Regressor**: Non-linear relationship modeling
- **Feature Engineering**: Temporal lags, moving averages, seasonal components
- **Cross-validation**: Spatial and temporal validation strategies

### 3. Model Evaluation

#### Performance Metrics (`conclusions/scripts/hydro_metrics.py`)

- **Nash-Sutcliffe Efficiency (NSE)**
- **Kling-Gupta Efficiency (KGE)**
- **Root Mean Square Error (RMSE)**
- **Base Flow Index (BFI)**: Lyne and Hollick filter
- **Flow Duration Curves (FDC)**
- **High/Low flow frequency analysis**

#### Comparative Analysis (`conclusions/`)

- Cross-dataset precipitation evaluation
- Model performance by basin characteristics
- Regional performance patterns
- Climate zone sensitivity analysis

## Data Sources

### Hydrometeorological Observations

- **Discharge Data**: Russian Federal Service for Hydrometeorology (Roshydromet)
- **Database**: Automated Information System for State Monitoring of Water Bodies (AIS-GMVO)
- **Period**: 2008-2020
- **Coverage**: Over 1,000 river gauge stations
- **Quality Control**: Automated outlier detection and gap filling

### Satellite/Reanalysis Products

- **ERA5/ERA5-Land**: ECMWF reanalysis (1979-present)
- **MSWEP v2.8**: Multi-Source Weighted-Ensemble Precipitation
- **GPCP v2.3**: Global Precipitation Climatology Project
- **IMERG v06**: Integrated Multi-satellitE Retrievals for GPM

### Static Datasets

- **HydroATLAS**: Global hydro-environmental attributes
- **Flow Direction Grids**: MERIT Hydro DEM derivatives
- **Land Cover**: ESA CCI Land Cover maps
- **Soil Properties**: SoilGrids 250m resolution

## Key Findings

### Main Research Conclusions

1. **Precipitation Dataset Impact**: The quality of river flow modeling significantly depends on the choice of meteorological data source, with ensemble approaches using multiple precipitation reanalyses providing the best results at regional scale.

2. **Physiographic Integration**: Inclusion of physiographic information in deep learning architectures substantially improves modeling quality - median Nash-Sutcliffe coefficient for 996 watersheds increased from 0.42 to 0.64.

3. **Knowledge Transfer**: LSTM modeling enables transfer of knowledge extracted from observed watersheds to ungauged basins through the ability to identify stable relationships between meteorological and hydrological parameters.

4. **Hidden State Interpretation**: LSTM models can extract and reproduce information about unobserved hydrometeorological characteristics such as evaporation, moisture storage, snow depth, and groundwater flow using only input meteorological data.

### Model Performance Results

1. **Neural Networks**: Best overall performance (median NSE: 0.72)
2. **HBV Model**: Strong performance in snow-dominated catchments (median NSE: 0.65)
3. **GR4J Model**: Reliable for temperate regions (median NSE: 0.61)
4. **Random Forest**: Good for non-linear relationships (median NSE: 0.58)

### Regional Patterns

- **Arctic basins**: Neural networks with ERA5-Land precipitation
- **Temperate regions**: HBV with MSWEP precipitation
- **Mountainous areas**: LSTM with multiple precipitation inputs
- **Arid zones**: Random Forest with GPCP precipitation

## Practical Applications

The developed methodologies and models have been practically implemented in the flood risk assessment module within the physical ESG risk analysis system at PAO Sberbank (Risk Division). Neural network-based flow modeling results were used to construct flood scenario models and analyze potential impacts on infrastructure objects, confirming the applicability of the research results in the context of natural risk management and sustainable planning at the level of large organizations.

## Installation and Usage

### Requirements

The project uses a Conda environment with geospatial and machine learning dependencies. Install the environment using:

```bash
# Create and activate the environment
conda env create -f geo_env.yml
conda activate Geo
```

#### Key Dependencies

- **Geospatial**: GeoPandas, Shapely, Cartopy, PyProj, Folium
- **Data Processing**: Pandas, NumPy, Xarray, NetCDF4, HDF5
- **Machine Learning**: PyTorch, Scikit-learn, NeuralHydrology
- **Visualization**: Matplotlib, Plotly, Seaborn, Bokeh
- **Computing**: Dask, Numba, CUDA support

#### Alternative Installation

If you prefer pip installation for specific packages:

```bash
# Core geospatial stack
pip install geopandas cartopy pyproj folium

# Machine learning
pip install torch torchvision neuralhydrology
pip install scikit-learn yellowbrick shap

# Data processing
pip install pandas numpy xarray netcdf4 h5py

# Visualization
pip install matplotlib plotly seaborn bokeh
```

#### Development Tools

The environment includes code quality tools:

```bash
# Code formatting and linting
pip install autopep8 flake8 ruff
```

#### CUDA Support

The environment includes CUDA-enabled PyTorch for GPU acceleration. Ensure NVIDIA drivers are installed:

```bash
# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Basic Usage

#### 1. Watershed Delineation

```python
from data_builders.watershed_creator import WatershedGPKG
from data_builders.scripts.config_definition import config_info

# Create watersheds from gauge points
watersheds = WatershedGPKG(config_info, xy_ready=True)
```

#### 2. Meteorological Data Processing

```python
from meteo_grids_parser.scripts.grid_calculator import Gridder

# Extract precipitation for watershed
meteo_grid = Gridder(
    half_grid_resolution=0.05,
    ws_geom=watershed_geometry,
    gauge_id=gauge_id,
    path_to_save=output_path,
    nc_paths=era5_files,
    dataset="era5_land",
    var="precipitation",
    aggregation_type="sum"
)
meteo_grid.grid_value_ws()
```

#### 3. Neural Network Training

```python
from neuralhydrology.nh_run import start_run
from pathlib import Path

# Train LSTM model
start_run(config_file=Path("./configs/lstm_config.yml"))
```

#### 4. Model Evaluation

```python
from conclusions.scripts.hydro_metrics import nse, kge, rmse

# Calculate performance metrics
nse_score = nse(predictions, observations)
kge_score = kge(predictions, observations)
rmse_score = rmse(predictions, observations)
```

## File Organization

### Configuration Files

- `*/config.yml`: Model configuration files
- `basins_*.txt`: Basin lists for train/validation/test splits
- `openf_basins.txt`: Open-source benchmark basins

### Data Files

- `*.nc`: NetCDF meteorological data
- `*.gpkg`: Geopackage vector data (watersheds, gauges)
- `*.csv`: Tabular results and metrics
- `*.txt`: Error logs and basin lists

### Output Files

- `test_results.p`: Pickle files with model predictions
- `*_metrics.csv`: Performance metrics by basin
- `corrupted_gauges.txt`: Processing error logs

## Database Registration and Patents

The research has resulted in several registered databases:

1. **Database Registration № 2024623169**: "Unified Model-Oriented Database of Historical Hydrometeorological Information for Regions with Expected Increase in Hydrological Risks" (2024)

2. **Database Registration № 2023622478**: "Database of Water Levels and Discharges with Associated Meteorological and Physiographic Characteristics of River Watersheds of the Russian Federation for 2008-2020" (2023)

## Publications and Conference Presentations

### Peer-Reviewed Publications

1. Ayzel, G.; Kurochkina, L.; Abramov, D.; Zhuravlev, S. Development of a Regional Gridded Runoff Dataset Using Long Short-Term Memory (LSTM) Networks. *Hydrology* 2021, 8, 6 - <https://doi.org/10.3390/hydrology8010006>

2. Abramov, D.; Ayzel, G.; Nikitin, O. Towards the Unified Approach for Obtaining Hydro-Meteorological and Landscape Characteristics for River Catchments. *CEUR Workshop Proceedings* 2021, 2930, 106–111 - <http://ceur-ws.org/Vol-2930/paper_14.pdf>

3. Ayzel, G.; Abramov, D. OpenForecast: An Assessment of the Operational Run in 2020–2021. *Geosciences* 2022, 12, 67 - <https://doi.org/10.3390/geosciences12020067>

4. Kononykhin, D.; Mozikov, M.; Mishtal, K.; Kuznetsov, P.; Abramov, D.; Sotiriadi, N.; Maximov, Y.; Savchenko, A. V.; Makarov, I. From Data to Decisions: Streamlining Geospatial Operations with Multimodal GlobeFlowGPT. *Proceedings of the 32nd ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL '24)* 2024, 649–652 - <https://dl.acm.org/doi/10.1145/3678717.3691248>

5. Moreido, V. M.; Terskii, P. N.; Abramov, D. V. Assessing the Reproduction Quality of Meteorological Characteristics by Several Atmospheric Reanalysis Models on the Territory of Crimean Peninsula. *Water Resources* 2024, 51, 873–881 - <https://doi.org/10.1134/S0097807824701215>

### Conference Presentations

- **ITHPC 2021** (Khabarovsk): "Towards the unified approach for obtaining hydro-meteorological and landscape characteristics for river catchments"
- **Hydrometeorological Trends 2022** (Irkutsk): "On the application of global information sources in hydrological modeling using machine learning methods"
- **ITHPC-2023** (Khabarovsk): "Development of a database of hydrological, meteorological and physiographic characteristics for watersheds in the Russian Federation"
- **Vinogradov Readings 2023** (St. Petersburg): "Modeling daily discharge of small and medium rivers of Russia using deep machine learning methods"

## Research Applications

This codebase supports research in:

- **Comparative Hydrology**: Multi-dataset precipitation evaluation
- **Machine Learning**: Neural network applications in hydrology
- **Climate Change**: CMIP model integration and future projections
- **Regional Hydrology**: Russian river basin characterization
- **Model Intercomparison**: Benchmarking different modeling approaches
- **ESG Risk Assessment**: Flood modeling for financial institutions

## Data Availability

The unified digital database is openly available on Zenodo platform [Abramov, Kurochkina, 2023] <https://zenodo.org/records/8432070> and the source code is available on GitHub at <https://github.com/dmbrmv/my_dissertation>.

## Citation

If you use this code in your research, please cite:

```
Abramov D.V. (2024). Hydrological Modeling and Precipitation Data Comparison 
for Russian River Basins. PhD Dissertation, State Hydrological Institute, Saint Petersburg.
```

## License

This project is licensed under Creative Commons Attribution 4.0 International (CC-BY-4.0), which allows free copying, distribution, and adaptation of materials for both scientific and commercial purposes while preserving authorship.

## Contact

For questions and collaboration opportunities, please contact [dmbrmv96@gmail.com].

## Acknowledgments

- Russian Federal Service for Hydrometeorology (Roshydromet)
- State Hydrological Institute, Saint Petersburg
- ECMWF for ERA5/ERA5-Land data
- NASA/JAXA for IMERG data
- Global Precipitation Climatology Centre (GPCC)
- HydroATLAS database contributors
- NeuralHydrology library developers
- PAO Sberbank ESG Risk Division
- All contributors and researchers involved in this project

## Future Work

- Integration of additional datasets (e.g., CMIP6 climate projections)
- Expansion to other Russian river basins
- Development of real-time forecasting capabilities
- Enhanced visualization tools for hydrological metrics
- Application of deep learning techniques for hydrological predictions
- Exploration of uncertainty quantification in hydrological modeling
- Collaboration with local hydrological agencies for data validation and application
