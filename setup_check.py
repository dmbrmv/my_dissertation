#!/usr/bin/env python3
"""Setup and installation verification script for TFT Predictions."""

import sys
from pathlib import Path
import subprocess
import importlib.util


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    if sys.version_info < (3, 12):
        print(f"❌ Python 3.12+ required, but you have {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True


def check_package_installed(package_name: str) -> bool:
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None


def install_package():
    """Install the package in development mode."""
    print("\nInstalling TFT Predictions package...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], 
                      check=True, capture_output=True, text=True)
        print("✅ Package installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_core_dependencies():
    """Check if core dependencies are available."""
    print("\nChecking core dependencies...")
    
    core_deps = [
        "numpy", "pandas", "torch", "pytorch_lightning", 
        "darts", "xarray", "geopandas", "pydantic", "yaml"
    ]
    
    missing_deps = []
    for dep in core_deps:
        if check_package_installed(dep):
            print(f"✅ {dep}")
        else:
            print(f"❌ {dep}")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All core dependencies available")
    return True


def check_project_structure():
    """Check if project structure is correct."""
    print("\nChecking project structure...")
    
    required_dirs = [
        "src", "src/config", "src/data", "src/models", 
        "src/evaluation", "src/utils", "scripts", "configs", "tests"
    ]
    
    required_files = [
        "src/__init__.py", "src/config/settings.py", "src/models/tft_model.py",
        "scripts/train_single_gauge.py", "configs/single_gauge.yaml",
        "requirements.txt", "pyproject.toml"
    ]
    
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ (missing)")
            return False
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            return False
    
    print("✅ Project structure is correct")
    return True


def run_basic_imports():
    """Test basic imports."""
    print("\nTesting basic imports...")
    
    try:
        from src.config.settings import Settings
        print("✅ Config module")
        
        from src.models.tft_model import TFTModelWrapper
        print("✅ Models module")
        
        from src.evaluation.metrics import nse, kge
        print("✅ Evaluation module")
        
        from src.data.loaders import prepare_model_data
        print("✅ Data module")
        
        from src.utils.helpers import set_random_seed
        print("✅ Utils module")
        
        print("✅ All imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def show_usage_examples():
    """Show usage examples."""
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    print("\n1. Quick Start with Workflow Script:")
    print("   python scripts/workflow.py --gauge-ids GAUGE_001 --quick-mode")
    
    print("\n2. Train Single Gauge Model:")
    print("   python scripts/train_single_gauge.py \\")
    print("     --gauge-id GAUGE_001 \\")
    print("     --config configs/single_gauge.yaml")
    
    print("\n3. Train Multi-Gauge Model:")
    print("   python scripts/train_multi_gauge.py \\")
    print("     --gauge-ids GAUGE_001 GAUGE_002 GAUGE_003 \\")
    print("     --config configs/multi_gauge.yaml")
    
    print("\n4. Make Predictions:")
    print("   python scripts/predict.py \\")
    print("     --model-path models/tft_model.pkl \\")
    print("     --gauge-ids GAUGE_001 GAUGE_002")
    
    print("\n5. Evaluate Model:")
    print("   python scripts/evaluate.py \\")
    print("     --model-path models/tft_model.pkl \\")
    print("     --create-plots")
    
    print("\n6. Interactive Analysis:")
    print("   jupyter notebook notebooks/exploratory_analysis.ipynb")
    
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    
    print("\nBefore running, update data paths in:")
    print("  - configs/single_gauge.yaml")
    print("  - configs/multi_gauge.yaml") 
    print("  - src/config/settings.py")
    
    print("\nRequired data structure:")
    print("  geo_data/")
    print("  ├── geometry/russia_ws.gpkg")
    print("  ├── attributes/static_with_height.csv")
    print("  └── ws_related_meteo/{nc_variable}/*.nc")


def main():
    """Main setup verification function."""
    print("TFT Predictions - Setup Verification")
    print("="*50)
    
    all_checks_passed = True
    
    # Run all checks
    checks = [
        check_python_version,
        check_project_structure,
        check_core_dependencies,
        run_basic_imports
    ]
    
    for check in checks:
        if not check():
            all_checks_passed = False
            break
    
    print("\n" + "="*50)
    if all_checks_passed:
        print("🎉 SETUP VERIFICATION PASSED!")
        print("The TFT Predictions framework is ready to use.")
        show_usage_examples()
    else:
        print("❌ SETUP VERIFICATION FAILED!")
        print("Please fix the issues above before proceeding.")
        
        print("\nCommon solutions:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Install package: pip install -e .")
        print("3. Check Python version: python --version")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    main()
