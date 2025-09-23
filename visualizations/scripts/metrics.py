"""Helpers for merging metric tables with gauge geometries."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from . import _style  # noqa: F401  # ensures font configuration is applied


def metric_viewer(gauges_file: gpd.GeoDataFrame, metric_col: str, metric_csv: str):
    model_metric = pd.read_csv(metric_csv)
    model_metric = model_metric.rename(columns={"basin": "gauge_id", "gauge": "gauge_id"})
    model_metric["gauge_id"] = model_metric["gauge_id"].astype("str")
    model_metric = model_metric.set_index("gauge_id")
    if "gauge_id" not in gauges_file.columns:
        res_file = gauges_file.join(model_metric).dropna()
    else:
        res_file = gauges_file.set_index("gauge_id").join(model_metric).dropna()
    nse_median = res_file[metric_col].median()
    return res_file, nse_median


__all__ = ["metric_viewer"]
