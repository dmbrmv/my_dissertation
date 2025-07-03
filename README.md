# TFT Predictions - Hydrological Forecasting with Darts

A clean, modular implementation for hydrological time series forecasting using Temporal Fusion Transformer (TFT) models via the Darts framework.

## Project Structure

```text
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Configuration management
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py           # Data loading utilities
│   │   └── preprocessors.py     # Data preprocessing and transformation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tft_model.py         # TFT model wrapper and training logic
│   │   └── callbacks.py         # PyTorch Lightning callbacks
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py           # Hydrological evaluation metrics
│   └── utils/
│       ├── __init__.py
│       └── helpers.py           # Utility functions
├── scripts/
│   ├── train_single_gauge.py    # Single gauge training script
│   ├── train_multi_gauge.py     # Multi-gauge training script
│   ├── predict.py               # Prediction script
│   ├── evaluate.py              # Model evaluation script
│   └── workflow.py              # Complete workflow example
├── notebooks/
│   └── exploratory_analysis.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_data/
│   ├── test_models/
│   └── test_evaluation/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Features

- **Modular Design**: Clean separation of concerns with dedicated modules for data, models, and evaluation
- **Type Safety**: Full type hints throughout the codebase
- **Hydrological Focus**: Specialized metrics (NSE, KGE) and preprocessing for hydrological data
- **GPU Support**: Automatic GPU detection and utilization
- **Flexible Configuration**: YAML-based configuration management
- **Comprehensive Testing**: Unit tests for all core functionality

## Quick Start

1. Install dependencies:

   ```bash
   pip install -e .
   ```

2. Configure your data paths in `src/config/settings.py`

3. Train a single gauge model:

   ```bash
   python scripts/train_single_gauge.py --config configs/single_gauge.yaml
   ```

4. Evaluate results:

   ```bash
   python scripts/evaluate.py --model_path models/tft_model.pkl
   ```

## Dependencies

- Python >= 3.12
- PyTorch >= 2.0
- Darts >= 0.27
- xarray >= 2023.1
- polars >= 0.20
- numpy >= 1.24
- pandas >= 2.0 (for legacy compatibility)
- geopandas >= 0.14
- pytorch-lightning >= 2.0

## Configuration

All model and data configurations are managed through YAML files in the `configs/` directory. See example configurations for details.
