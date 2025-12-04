import json
from pathlib import Path


def load_metrics_from_directory(base_path: Path) -> dict:
    """Load all metrics from model optimization directory.

    Args:
        base_path: Path to model optimization directory.

    Returns:
        Dictionary with structure {gauge_id: {meteo_source: metrics_dict}}.
    """
    results = {}

    for gauge_dir in base_path.iterdir():
        if not gauge_dir.is_dir():
            continue

        gauge_id = gauge_dir.name
        results[gauge_id] = {}

        # Find all metrics files
        for metrics_file in gauge_dir.glob("*_metrics.json"):
            # Extract meteo source from filename (e.g., "10042_e5_metrics.json")
            parts = metrics_file.stem.split("_")
            meteo_source = "_".join(parts[1:-1])  # Get middle part(s)

            try:
                with open(metrics_file, encoding="utf-8-sig") as f:
                    metrics = json.load(f)
                    results[gauge_id][meteo_source] = metrics
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Try with latin1 encoding if utf-8 fails
                try:
                    with open(metrics_file, encoding="latin1") as f:
                        metrics = json.load(f)
                        results[gauge_id][meteo_source] = metrics
                except (UnicodeDecodeError, json.JSONDecodeError, FileNotFoundError):
                    continue
            except FileNotFoundError:
                continue

    return results
