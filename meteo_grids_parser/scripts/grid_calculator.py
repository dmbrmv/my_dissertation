from .geom_proc import (
    create_gdf, getSquareVertices, polygon_area, poly_from_multipoly)
from .nc_proc import nc_by_extent

from dask import config as dask_cfg
from shapely.geometry import Polygon
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import gc


class Gridder:
    """_summary_
    """

    def __init__(self,
                 half_grid_resolution: float, ws_geom: Polygon,
                 gauge_id: str, path_to_save: Path,
                 nc_pathes: list, var: str, dataset: str,
                 aggregation_type: str = 'sum',
                 force_weights: bool = False,
                 weight_mark: str = '') -> None:
        """_summary_

        Args:
            half_grid_resolution (float): _description_
            ws_geom (Polygon): _description_
            gauge_id (str): _description_
            path_to_save (Path): _description_
            nc_pathes (list): _description_
            var (str): _description_
            dataset (str): _description_
            aggregation_type (str, optional): _description_. Defaults to 'sum'.
            force_weights (bool, optional): _description_. Defaults to False.

        Raises:
            Exception: _description_
        """

        self.grid_res = half_grid_resolution
        self.ws_geom = ws_geom
        self.gauge_id = gauge_id
        self.path_to_save = path_to_save
        self.nc_pathes = nc_pathes
        self.dataset = dataset
        self.var = var
        self.force_weights = force_weights
        self.wm = weight_mark

        self.aggregation_type = aggregation_type
        if self.aggregation_type not in ['sum', 'mean']:
            raise Exception(
                f"Sorry, only sum and mean aggregations are allowed!\
                You insert {aggregation_type}")
        with dask_cfg.set(**{'array.slicing.split_large_chunks': True}):
            self.nc_data = xr.open_mfdataset(nc_pathes)
        # weights
        if self.force_weights:
            self.weight_folder = Path(
                f'{self.path_to_save}/weights/{self.wm}_{self.grid_res}')
            self.weight_folder.mkdir(exist_ok=True, parents=True)
        else:
            self.weight_folder = Path(
                f'{self.path_to_save}/weights/{self.grid_res}')
            self.weight_folder.mkdir(exist_ok=True, parents=True)

        self.test_weight = Path(f'{self.weight_folder}/{self.gauge_id}.nc')
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
        with dask_cfg.set(**{'array.slicing.split_large_chunks': True}):
            mask_nc = nc_by_extent(nc=self.nc_data,
                                   shape=self.ws_geom,
                                   grid_res=self.grid_res,
                                   dataset=self.dataset)
        # calculate area of watershed to latter comparisons
        ws_area = polygon_area(geo_shape=ws_gdf.loc[0, 'geometry'])
        # get lat, lon which help define area for intersection
        nc_lat, nc_lon = mask_nc.lat.values, mask_nc.lon.values

        # emulate polygons for grid
        polygons = list()
        for lat in nc_lat:
            for lon in nc_lon:
                # h = 0.125 as a half of ERA5 resolution
                # phi rotation angle
                polygons.append(Polygon(
                    getSquareVertices(mm=(lon, lat),
                                      h=self.grid_res,
                                      phi=0)))
        # create geodataframe from each polygon from emulation
        polygons = [create_gdf(poly) for poly in polygons]
        # calculate area of watershed to latter comparisons
        ws_area = polygon_area(geo_shape=ws_gdf.loc[0, 'geometry'])
        # find intersection beetween grid cell and actual watershed
        intersected = list()
        for polygon in polygons:
            try:
                intersected.append(gpd.overlay(df1=ws_gdf,
                                               df2=polygon,
                                               how='intersection'))
            except KeyError:
                intersected.append(gpd.GeoDataFrame())
        # find biggest intersection if it returns MultiPolygon instance
        # select biggest Polygon in MultiPolygon
        intersected = [create_gdf(
            poly_from_multipoly(
                section.loc[0, 'geometry']))  # type: ignore
            if len(section) != 0
            else gpd.GeoDataFrame()
            for section in intersected]
        # create mask for intersection with net_cdf
        inter_mask = np.array([False if section.empty is True
                               else True
                               for section in intersected])
        # shape of initial coordindate size
        grid_shape = (len(nc_lat), len(nc_lon))

        inter_mask = inter_mask.reshape(grid_shape)
        inter_mask = xr.DataArray(data=inter_mask,
                                  dims=['lat', 'lon'],
                                  coords=[nc_lat, nc_lon])
        # calculate weights of each intersection correspond to net cdf grid
        weights = np.array(
            [0 if section.empty else
                polygon_area(geo_shape=section.loc[0, 'geometry']) / ws_area
                for i, section in enumerate(intersected)])
        weights = weights.reshape(grid_shape)
        # transform to DataArray for calculations
        weights = xr.DataArray(data=weights,
                               dims=['lat', 'lon'])
        weights.name = 'weights'
        weights = weights.where(inter_mask, drop=True)
        weights = weights.fillna(0)
        weights.to_netcdf(f'{self.weight_folder}/{self.gauge_id}.nc')
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
            aggregation_type (str, optional): _description_. Defaults to 'sum'.

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """

        # select variable from netcdf
        var = list(self.nc_data.data_vars)[0]

        # use mask on net_cdf
        with dask_cfg.set(**{'array.slicing.split_large_chunks': True}):
            mask_nc = nc_by_extent(nc=self.nc_data,
                                   shape=self.ws_geom,
                                   grid_res=self.grid_res,
                                   dataset=self.dataset)

        inter_mask = self.weights.astype(bool)

        # create final instersection
        ws_nc = mask_nc.where(inter_mask, drop=True)

        final_save = Path(f'{self.path_to_save}/{self.dataset}/{self.var}')
        final_save.mkdir(exist_ok=True, parents=True)

        res_df = pd.DataFrame()

        if self.aggregation_type == 'sum':
            res_df['date'] = ws_nc.time.values
            res_df[var] = ws_nc.weighted(weights=self.weights).sum(
                dim=['lat', 'lon'])[var].values
            res_df = res_df.set_index('date')
            res_df.to_csv(f'{final_save}/{self.gauge_id}.csv')

        else:
            res_df['date'] = ws_nc.time.values
            res_df[var] = ws_nc.weighted(weights=self.weights).mean(
                dim=['lat', 'lon'])[var].values
            res_df = res_df.set_index('date')
            res_df.to_csv(f'{final_save}/{self.gauge_id}.csv')

        gc.collect()

        return res_df
