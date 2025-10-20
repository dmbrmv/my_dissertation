"""Quick test script to verify RFR enhanced features implementation.

This script tests key components without running full optimization:
1. PET calculation
2. Cyclic temporal encoding
3. Static attribute loading
4. Feature engineering pipeline
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append("./")
from src.models.gr4j.pet import pet_oudin
from src.models.rfr.rfr_optuna import create_temporal_features
from src.readers.geom_reader import load_geodata


def test_pet_calculation():
    """Test PET calculation with sample data."""
    print("Testing PET calculation...")
    temp = [10.0, 15.0, 20.0, 25.0]
    doy = [1, 100, 200, 300]
    lat = 55.0

    pet = pet_oudin(temp, doy, lat)
    print(f"  ✓ PET values: {pet}")
    assert len(pet) == len(temp), "PET length mismatch"
    assert np.all(pet >= 0), "Negative PET values"
    print("  ✓ PET test passed\n")


def test_cyclic_encoding():
    """Test cyclic temporal encoding."""
    print("Testing cyclic temporal encoding...")

    # Create sample date range
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    df = pd.DataFrame(
        {"value": np.random.randn(len(dates)), "q_mm_day": np.random.rand(len(dates))},
        index=dates,
    )

    # Test feature creation
    df_features = create_temporal_features(
        df, rolling_windows=[1, 2], base_features=["value"], pet_values=None
    )

    assert "doy_sin" in df_features.columns, "doy_sin missing"
    assert "doy_cos" in df_features.columns, "doy_cos missing"

    # Check cyclic property
    assert -1 <= df_features["doy_sin"].min() <= 1, "doy_sin out of range"
    assert -1 <= df_features["doy_cos"].min() <= 1, "doy_cos out of range"

    print(f"  ✓ Cyclic features shape: {df_features.shape}")
    print(f"  ✓ Features: {list(df_features.columns)}")
    print("  ✓ Cyclic encoding test passed\n")


def test_static_attributes():
    """Test static attribute loading."""
    print("Testing static attributes loading...")

    static_file = Path("data/attributes/hydro_atlas_cis_camels.csv")
    if not static_file.exists():
        print(f"  ⚠ Static file not found: {static_file}")
        print("  Skipping test\n")
        return

    static_df = pd.read_csv(static_file, dtype={"gauge_id": str})
    static_df.set_index("gauge_id", inplace=True)

    static_columns = [
        "for_pc_sse",
        "crp_pc_sse",
        "ws_area",
        "ele_mt_sav",
    ]

    # Get first gauge
    gauge_id = static_df.index[0]
    static_values = static_df.loc[gauge_id][static_columns].values.astype(float)

    print(f"  ✓ Test gauge: {gauge_id}")
    print(f"  ✓ Static values: {static_values}")
    print(f"  ✓ Number of features: {len(static_values)}")
    assert len(static_values) == len(static_columns), "Feature count mismatch"
    print("  ✓ Static attributes test passed\n")


def test_full_pipeline():
    """Test complete feature engineering pipeline."""
    print("Testing full feature engineering pipeline...")

    # Load gauge geometry
    _, gauges = load_geodata(folder_depth=".")
    gauge_id = gauges.index[0]
    latitude = gauges.loc[gauge_id, "geometry"].y

    print(f"  ✓ Test gauge: {gauge_id}, Lat: {latitude:.2f}")

    # Check if data file exists
    data_path = Path(f"data/nc_all_q/{gauge_id}.nc")
    if not data_path.exists():
        print(f"  ⚠ Data file not found: {data_path}")
        print("  Skipping test\n")
        return

    # Load data
    with xr.open_dataset(data_path) as ds:
        df = ds.to_dataframe()

    # Calculate mean temperature
    if "t_mean_e5l" not in df.columns:
        df["t_mean_e5l"] = (df["t_max_e5l"] + df["t_min_e5l"]) / 2

    # Select period
    df_subset = df.loc["2019-01-01":"2019-12-31", ["t_mean_e5l", "prcp_e5l"]].copy()

    # Calculate PET
    t_mean_list = df_subset["t_mean_e5l"].tolist()
    doy_list = df_subset.index.dayofyear.tolist()  # type: ignore[attr-defined]
    pet_values = pet_oudin(t_mean_list, doy_list, latitude)

    print(f"  ✓ PET range: {pet_values.min():.2f} - {pet_values.max():.2f} mm/day")

    # Create features
    df_features = create_temporal_features(
        df_subset,
        rolling_windows=[1, 2, 4],
        base_features=["prcp_e5l", "t_mean_e5l"],
        pet_values=pet_values,
    )

    print(f"  ✓ Input shape: {df_subset.shape}")
    print(f"  ✓ Output shape: {df_features.shape}")
    print(f"  ✓ Number of features: {len(df_features.columns)}")
    print(f"  ✓ Feature names: {list(df_features.columns)[:10]}...")

    # Verify expected features
    assert "doy_sin" in df_features.columns, "Missing doy_sin"
    assert "doy_cos" in df_features.columns, "Missing doy_cos"
    assert "pet_mm_day" in df_features.columns, "Missing pet_mm_day"
    assert "prcp_e5l_sum_2d" in df_features.columns, "Missing rolling feature"

    print("  ✓ Full pipeline test passed\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("RFR ENHANCED FEATURES - COMPONENT TESTS")
    print("=" * 60)
    print()

    try:
        test_pet_calculation()
        test_cyclic_encoding()
        test_static_attributes()
        test_full_pipeline()

        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Run: python scripts/rfr_train.py")
        print("  2. Monitor: logs/rfr_train.log")
        print("  3. Check results: data/optimization/rfr_results/")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
