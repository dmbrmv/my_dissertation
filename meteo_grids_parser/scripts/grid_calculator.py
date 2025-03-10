import gc
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from dask import config as dask_cfg
from shapely.geometry import Polygon

from .geom_proc import create_gdf, getSquareVertices, poly_from_multipoly
from .nc_proc import nc_by_extent


class Gridder:
    """_summary_"""

    def __init__(
        self,
        half_grid_resolution: float,
        ws_geom: Polygon,
        gauge_id: str,
        path_to_save: Path,
        nc_paths: list,
        var: str,
        dataset: str,
        aggregation_type: str = "sum",
        force_weights: bool = False,
        weight_mark: str = "",
        prcp_coef: float = 1e2,
        extend_data: bool = False,
        merge_data: bool = False,
    ) -> None:
        """_summary_.

        Args.
        ----
            half_grid_resolution (float): _description_
            ws_geom (Polygon): _description_
            gauge_id (str): _description_
            path_to_save (Path): _description_
            nc_paths (list): _description_
            var (str): _description_
            dataset (str): _description_
            aggregation_type (str, optional): _description_. Defaults to "sum".
            force_weights (bool, optional): _description_. Defaults to False.

        Raises.
        ------
            Exception: _description_

        """
        self.grid_res = half_grid_resolution
        self.ws_geom = ws_geom
        self.gauge_id = gauge_id
        self.path_to_save = path_to_save
        self.nc_paths = nc_paths
        self.dataset = dataset
        self.var = var
        self.force_weights = force_weights
        self.wm = weight_mark
        self.prcp_coef = prcp_coef
        self.extend_data = extend_data
        self.merge_data = merge_data

        self.aggregation_type = aggregation_type
        if self.aggregation_type not in ["sum", "mean"]:
            raise Exception(
                f"Sorry, only sum and mean aggregations are allowed!\
                You insert {aggregation_type}"
            )
        with dask_cfg.set(**{"array.slicing.split_large_chunks": True}):
            self.nc_data = xr.open_mfdataset(nc_paths)
        # weights
        if self.force_weights:
            self.weight_folder = Path(
                f"{self.path_to_save}/weights/{self.wm}_{self.grid_res}"
            )
            self.weight_folder.mkdir(exist_ok=True, parents=True)
        else:
            self.weight_folder = Path(f"{self.path_to_save}/weights/{self.grid_res}")
            self.weight_folder.mkdir(exist_ok=True, parents=True)

        self.test_weight = Path(f"{self.weight_folder}/{self.gauge_id}.nc")
        self.weight_condition = self.test_weight.is_file()
        if self.weight_condition:
            self.weights = xr.open_dataarray(self.test_weight)
        else:
            self.weights = Gridder.grid_weights(self)

    def grid_weights(self):
        """_summary_

        Args:
            nc_data (xr.Dataset): _description_
            ws_geom (Polygon): _description_
            gauge_id (str): _description_
            half_grid_resolution (float): _description_
            path_to_save (Path): _description_
        """
        # watershed boundaries geometry as geodataframe
        ws_gdf = create_gdf(self.ws_geom)
        with dask_cfg.set(**{"array.slicing.split_large_chunks": True}):
            mask_nc = nc_by_extent(
                nc=self.nc_data,
                shape=self.ws_geom,
                grid_res=self.grid_res,
                dataset=self.dataset,
            )
        # get lat, lon which help define area for intersection
        nc_lat, nc_lon = mask_nc.lat.values, mask_nc.lon.values
        # estimate metric projection
        tiff_epsg = ws_gdf.estimate_utm_crs().to_epsg()
        # emulate polygons for grid
        polygons = list()
        for lat in nc_lat:
            for lon in nc_lon:
                # h = 0.125 as a half of ERA5 resolution
                # phi rotation angle
                polygons.append(
                    Polygon(getSquareVertices(mm=(lon, lat), h=self.grid_res, phi=0))
                )
        # create geodataframe from each polygon from emulation
        polygons = [create_gdf(poly) for poly in polygons]

        # find intersection beetween grid cell and actual watershed
        intersected = list()
        for polygon in polygons:
            try:
                intersected.append(
                    gpd.overlay(df1=ws_gdf, df2=polygon, how="intersection")
                )
            except KeyError:
                intersected.append(gpd.GeoDataFrame())
        # find biggest intersection if it returns MultiPolygon instance
        # select biggest Polygon in MultiPolygon
        intersected = [
            create_gdf(poly_from_multipoly(section.loc[0, "geometry"]))  # type: ignore
            if len(section) != 0
            else gpd.GeoDataFrame()
            for section in intersected
        ]
        # create mask for intersection with net_cdf
        inter_mask = np.array(
            [False if section.empty is True else True for section in intersected]
        )
        # create GeoDataFrame for polygons from GRID
        poly_gdf = pd.DataFrame(columns=["geometry"])
        poly_gdf["geometry"] = polygons
        poly_gdf = gpd.GeoDataFrame(data=poly_gdf, geometry="geometry").set_crs(
            epsg=4326
        )
        # get indexes with data in initial file
        sjoin_idx = ws_gdf.sjoin(poly_gdf).index_right.sort_values().values
        # get fraction of overlay
        try:
            overlay_area = (
                ws_gdf.overlay(poly_gdf, keep_geom_type=False)
                .to_crs(epsg=tiff_epsg)
                .area.values
            )
            # calculate area of each polygon
            initial_area = poly_gdf.to_crs(epsg=tiff_epsg).area.values[sjoin_idx]
            # get weights of intersection
            poly_gdf.loc[sjoin_idx, "weights"] = overlay_area / initial_area
        except ValueError as e:
            with open("corrupted_gauges.txt", "a", encoding="utf-8") as f:
                f.write(f"{self.gauge_id}-{self.dataset}-weight receiver - {e}" + "\n")
        weights = poly_gdf["weights"].values
        # shape of initial coordindate size
        grid_shape = (len(nc_lat), len(nc_lon))
        # transform weight according to initial grid shape
        weights = weights.reshape(grid_shape)
        # transform to DataArray for calculations
        weights = xr.DataArray(data=weights, dims=["lat", "lon"])
        weights.name = "weights"
        weights = weights.fillna(0)
        weights.to_netcdf(f"{self.weight_folder}/{self.gauge_id}.nc")
        gc.collect()

        return weights

    def grid_value_ws(self):
        """_summary_

        Args:
            nc_pathes (list): _description_
            weights (xr.DataArray): _description_
            grid_res (float): _description_
            ws_geom (Polygon): _description_
            gauge_id (str): _description_
            var_folder (str): _description_
            path_to_save (str): _description_
            aggregation_type (str, optional): _description_. Defaults to "sum".

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """

        # select variable from netcdf
        var = list(self.nc_data.data_vars)[0]

        # use mask on net_cdf
        with dask_cfg.set(**{"array.slicing.split_large_chunks": True}):
            mask_nc = nc_by_extent(
                nc=self.nc_data,
                shape=self.ws_geom,
                grid_res=self.grid_res,
                dataset=self.dataset,
            )

        inter_mask = self.weights.astype(bool)
        # create final instersection
        try:
            ws_nc = mask_nc.where(inter_mask, drop=True)
        except ValueError as e:
            with open("corrupted_gauges.txt", "a", encoding="utf-8") as f:
                f.write(f"{self.gauge_id}-{self.dataset}-weighted value - {e}" + "\n")

        final_save = Path(f"{self.path_to_save}/{self.dataset}/{self.var}")
        final_save.mkdir(exist_ok=True, parents=True)

        res_df = pd.DataFrame()
        try:
            # if self.aggregation_type == "sum":
            #     res_df["date"] = ws_nc.time.values
            #     res_df[var] = (
            #         ws_nc.weighted(weights=self.weights)
            #         .sum(dim=["lat", "lon"])[var]
            #         .values
            #     )
            #     res_df = res_df.set_index("date")
            #     res_df.to_csv(f"{final_save}/{self.gauge_id}.csv")

            if self.aggregation_type == "sum":
                res_df["date"] = ws_nc.time.values
                res_df[var] = (
                    ws_nc.weighted(weights=self.weights)
                    .sum(dim=["lat", "lon"])[var]
                    .values
                )
                res_df = res_df.set_index("date")
                res_df *= self.prcp_coef
                self.res_df = res_df

                if self.extend_data:
                    try:
                        # extend existed forecast, update from day to future
                        old_data = pd.read_csv(f"{final_save}/{self.gauge_id}.csv")
                        old_data["date"] = pd.to_datetime(old_data["date"])
                        old_data = old_data.set_index("date")
                        new_data = res_df.combine_first(old_data)
                        new_data.to_csv(f"{final_save}/{self.gauge_id}.csv")
                    except FileNotFoundError:
                        res_df.to_csv(f"{final_save}/{self.gauge_id}.csv")
                elif self.merge_data:
                    try:
                        # create long list with different forecast horizons
                        old_data = pd.read_csv(f"{final_save}/{self.gauge_id}.csv")
                        old_data["date"] = pd.to_datetime(old_data["date"])
                        res_df = res_df.reset_index()
                        res_df["forecast_horizon"] = [i for i in range(len(res_df))]
                        new_data = (
                            pd.concat([old_data, res_df])
                            .sort_values(by="date")
                            .reset_index(drop=True)
                        )
                        new_data.to_csv(
                            f"{final_save}/{self.gauge_id}.csv", index=False
                        )
                    except FileNotFoundError:
                        res_df = res_df.reset_index()
                        res_df["forecast_horizon"] = [i for i in range(len(res_df))]
                        res_df.to_csv(f"{final_save}/{self.gauge_id}.csv", index=False)
                else:
                    res_df.to_csv(f"{final_save}/{self.gauge_id}.csv")

            else:
                res_df["date"] = ws_nc.time.values
                res_df[var] = (
                    ws_nc.weighted(weights=self.weights)
                    .mean(dim=["lat", "lon"])[var]
                    .values
                )
                res_df = res_df.set_index("date")
            if self.extend_data:
                try:
                    old_data = pd.read_csv(f"{final_save}/{self.gauge_id}.csv")
                    old_data["date"] = pd.to_datetime(old_data["date"])
                    old_data = old_data.set_index("date")
                    new_data = res_df.combine_first(old_data)
                    new_data.to_csv(f"{final_save}/{self.gauge_id}.csv")
                except FileNotFoundError:
                    res_df.to_csv(f"{final_save}/{self.gauge_id}.csv")
            elif self.merge_data:
                try:
                    # create long list with different forecast horizons
                    old_data = pd.read_csv(f"{final_save}/{self.gauge_id}.csv")
                    old_data["date"] = pd.to_datetime(old_data["date"])
                    res_df = res_df.reset_index()
                    res_df["forecast_horizon"] = [i for i in range(len(res_df))]
                    new_data = (
                        pd.concat([old_data, res_df])
                        .sort_values(by="date")
                        .reset_index(drop=True)
                    )
                    new_data.to_csv(f"{final_save}/{self.gauge_id}.csv", index=False)
                except FileNotFoundError:
                    res_df = res_df.reset_index()
                    res_df["forecast_horizon"] = [i for i in range(len(res_df))]
                    res_df.to_csv(f"{final_save}/{self.gauge_id}.csv", index=False)
            else:
                res_df.to_csv(f"{final_save}/{self.gauge_id}.csv")
        except ValueError as e:
            with open("corrupted_gauges.txt", "a", encoding="utf-8") as f:
                f.write(f"{self.gauge_id}-{self.dataset}-weighted value - {e}" + "\n")
        except UnboundLocalError as e:
            with open("corrupted_gauges.txt", "a", encoding="utf-8") as f:
                f.write(f"{self.gauge_id}-{self.dataset}-weighted value - {e}" + "\n")
        gc.collect()

        return res_df
