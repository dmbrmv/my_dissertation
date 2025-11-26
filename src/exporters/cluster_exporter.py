"""Cluster analysis export utilities.

This module provides functions for exporting cluster analysis results
to CSV and YAML formats with Russian naming conventions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

log = logging.getLogger(__name__)


def export_gauge_mapping_csv(
    gauge_mapping: pd.DataFrame,
    output_path: str | Path,
    columns: list[str] | None = None,
) -> None:
    """Export gauge-level cluster mapping to CSV.

    Args:
        gauge_mapping: DataFrame with gauge_id and cluster assignments.
        output_path: Path to output CSV file.
        columns: Column names to export. If None, exports all columns.

    Example:
        >>> export_gauge_mapping_csv(
        ...     gauge_mapping,
        ...     "res/chapter_one/gauge_hybrid_mapping.csv",
        ...     columns=[
        ...         "gauge_id",
        ...         "geo_cluster_ru",
        ...         "hydro_cluster_ru",
        ...         "hybrid_class",
        ...         "hex_id",
        ...     ],
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if columns is not None:
        export_df = gauge_mapping[columns].copy()
    else:
        export_df = gauge_mapping.copy()

    export_df.to_csv(output_path, index=False, encoding="utf-8")

    log.info("Exported gauge mapping to %s (%d gauges)", output_path, len(export_df))


def export_geo_clusters_yaml(
    cluster_info: pd.DataFrame,
    feature_descriptions: dict[str, dict[str, Any]],
    output_path: str | Path,
    n_clusters: int,
) -> None:
    """Export geographical cluster metadata to YAML.

    Args:
        cluster_info: DataFrame with cluster centroids and metadata.
        feature_descriptions: Feature metadata dictionary from config.
        output_path: Path to output YAML file.
        n_clusters: Total number of geographical clusters.

    Example:
        >>> export_geo_clusters_yaml(
        ...     cluster_summary,
        ...     feature_descriptions,
        ...     "docs/cluster_references/geo_clusters_ru.yaml",
        ...     n_clusters=9,
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_data = {
        "geographical_clustering": {
            "description": "Физико-географические кластеры водосборов России",
            "n_clusters": n_clusters,
            "method": "Ward hierarchical clustering",
            "features_used": list(feature_descriptions.keys()),
            "clusters": {},
        }
    }

    for _, row in cluster_info.iterrows():
        cluster_id = int(row["cluster_id"])
        cluster_ru = f"Ф{cluster_id}"

        yaml_data["geographical_clustering"]["clusters"][cluster_ru] = {
            "name": row.get("cluster_name", f"Cluster {cluster_id}"),
            "n_gauges": int(row["n_gauges"]),
            "mean_silhouette": float(row["mean_silhouette"]),
            "location": {
                "lat_mean": float(row["lat_mean"]),
                "lon_mean": float(row["lon_mean"]),
            },
            "watershed_area_km2": {
                "mean": float(row["mean_area_km2"]),
                "median": float(row["median_area_km2"]),
            },
        }

    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(
            yaml_data, f, allow_unicode=True, sort_keys=False, default_flow_style=False
        )

    log.info("Exported geo clusters to %s", output_path)


def export_hydro_clusters_yaml(
    regime_df: pd.DataFrame,
    output_path: str | Path,
    n_clusters: int,
) -> None:
    """Export hydrological cluster metadata with regime characteristics to YAML.

    Args:
        regime_df: DataFrame with regime info (from build_regime_dataframe).
        output_path: Path to output YAML file.
        n_clusters: Total number of hydrological clusters.

    Example:
        >>> export_hydro_clusters_yaml(
        ...     regime_df,
        ...     "docs/cluster_references/hydro_clusters_ru.yaml",
        ...     n_clusters=10,
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_data = {
        "hydrological_clustering": {
            "description": "Гидрологические режимные кластеры водосборов России",
            "n_clusters": n_clusters,
            "method": "Ward hierarchical clustering on normalized seasonal patterns",
            "clusters": {},
        }
    }

    for _, row in regime_df.iterrows():
        cluster_name = row["cluster_name"]
        # Extract cluster number and format as Г#
        cluster_num = cluster_name.split()[-1]
        cluster_ru = f"Г{cluster_num}"

        yaml_data["hydrological_clustering"]["clusters"][cluster_ru] = {
            "regime_type": row["regime_type"],
            "n_gauges": int(row["n_gauges"]),
            "peak_doy": int(row["peak_doy"]),
            "peak_value": float(row["peak_value"]),
            "seasonal_ratios": {
                "winter": float(row["winter_ratio"]),
                "spring": float(row["spring_ratio"]),
                "summer": float(row["summer_ratio"]),
                "autumn": float(row["autumn_ratio"]),
            },
            "cv": float(row["cv"]),
        }

    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(
            yaml_data, f, allow_unicode=True, sort_keys=False, default_flow_style=False
        )

    log.info("Exported hydro clusters to %s", output_path)


def export_hybrid_clusters_yaml(
    gauge_mapping: pd.DataFrame,
    geo_yaml_path: str | Path,
    hydro_yaml_path: str | Path,
    output_path: str | Path,
) -> None:
    """Export hybrid cluster combinations with cross-references to YAML.

    Args:
        gauge_mapping: DataFrame with gauge-level hybrid class assignments.
        geo_yaml_path: Path to geographical clusters YAML (for reference).
        hydro_yaml_path: Path to hydrological clusters YAML (for reference).
        output_path: Path to output YAML file.

    Example:
        >>> export_hybrid_clusters_yaml(
        ...     gauge_mapping,
        ...     "docs/cluster_references/geo_clusters_ru.yaml",
        ...     "docs/cluster_references/hydro_clusters_ru.yaml",
        ...     "docs/cluster_references/hybrid_clusters_ru.yaml",
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count gauges per hybrid class
    class_counts = gauge_mapping["hybrid_class"].value_counts().to_dict()

    yaml_data = {
        "hybrid_classification": {
            "description": "Гибридная физико-гидрологическая классификация",
            "n_classes": len(class_counts),
            "naming_convention": "Ф#-Г# (физико-географический × гидрологический)",
            "references": {
                "geo_clusters": str(geo_yaml_path),
                "hydro_clusters": str(hydro_yaml_path),
            },
            "classes": {},
        }
    }

    for hybrid_class, count in sorted(class_counts.items()):
        # Parse geo and hydro components
        parts = hybrid_class.split("-")
        geo_part = parts[0] if len(parts) > 0 else "?"
        hydro_part = parts[1] if len(parts) > 1 else "?"

        yaml_data["hybrid_classification"]["classes"][hybrid_class] = {
            "n_gauges": int(count),
            "geo_cluster": geo_part,
            "hydro_cluster": hydro_part,
        }

    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(
            yaml_data, f, allow_unicode=True, sort_keys=False, default_flow_style=False
        )

    log.info("Exported hybrid clusters to %s", output_path)


def generate_cluster_readme(
    output_dir: str | Path,
    n_geo_clusters: int,
    n_hydro_clusters: int,
    n_hybrid_classes: int,
    geo_method: str = "Ward hierarchical clustering",
    hydro_method: str = "Ward hierarchical clustering on seasonal patterns",
) -> None:
    """Generate README.md with cluster methodology summary.

    Args:
        output_dir: Directory to write README.md.
        n_geo_clusters: Number of geographical clusters.
        n_hydro_clusters: Number of hydrological clusters.
        n_hybrid_classes: Number of consolidated hybrid classes.
        geo_method: Description of geographical clustering method.
        hydro_method: Description of hydrological clustering method.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    readme_path = output_dir / "README.md"

    content = f"""# Cluster Classification Reference

## Overview

This directory contains YAML reference files for the hybrid geo-hydrological
classification of Russian watersheds.

## Clustering Summary

- **Geographical clusters:** {n_geo_clusters} (Ф1-Ф{n_geo_clusters})
- **Hydrological clusters:** {n_hydro_clusters} (Г1-Г{n_hydro_clusters})
- **Hybrid classes:** {n_hybrid_classes} consolidated combinations

## Methodology

### Geographical Clustering

{geo_method}

Based on 19 HydroATLAS physiographic features including land cover,
soil properties, topography, and hydrogeology.

### Hydrological Clustering

{hydro_method}

Based on normalized seasonal discharge patterns (366-day median cycles)
with spike removal and Savitzky-Golay smoothing.

### Hybrid Classification

Combines geographical and hydrological cluster assignments at gauge level.
Rare combinations are consolidated using hexagonal spatial aggregation
(80 km radius) to create meaningful classes with sufficient sample sizes.

## Files

- `geo_clusters_ru.yaml` — Geographical cluster metadata
- `hydro_clusters_ru.yaml` — Hydrological regime characteristics
- `hybrid_clusters_ru.yaml` — Hybrid class combinations
- `gauge_cluster_mapping.csv` — Gauge-level assignments

## Naming Convention

- **Ф#** — Физико-географический кластер (Geographical)
- **Г#** — Гидрологический кластер (Hydrological)
- **Ф#-Г#** — Гибридная комбинация (Hybrid)

## Usage

```python
import yaml
from pathlib import Path

# Load cluster metadata
with open("geo_clusters_ru.yaml", "r", encoding="utf-8") as f:
    geo_clusters = yaml.safe_load(f)

# Load gauge mapping
import pandas as pd
gauge_mapping = pd.read_csv("gauge_cluster_mapping.csv")
```

## Citation

Generated by HybridClusterAnalysis workflow.
See `notebooks/ChapterOne.ipynb` for implementation details.
"""

    readme_path.write_text(content, encoding="utf-8")

    log.info("Generated README at %s", readme_path)
