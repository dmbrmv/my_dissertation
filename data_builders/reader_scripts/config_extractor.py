from pathlib import Path
import pandas as pd


class ConfigExtractor:
    """
    brief shell to hide unnecessary paths from forecaster
        Args:
            config_path: path to configuration file
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._cmip_path = self.cfg["cmip_path"]
        self._meteo_path = self.cfg["meteo_path"]
        self._layers_path = self.cfg["layers_path"]
        self._ws_path = self.cfg["ws_path"]
        self._id_label_path = self.cfg["id_label_path"]
        self._forecast_predictors = self.cfg["forecast_predictors"]
        self._ais_dates = pd.date_range(
            start=self.cfg["ais_first_date"],
            end=self.cfg["ais_last_date"])
        self._forecast_dates = pd.date_range(
            start=self.cfg["forecast_start_date"],
            end=self.cfg["forecast_end_date"])
        # geometry builder settings

        self._save_storage = self.cfg['save_storage']
        self._watershed_storage = self.cfg['watershed_storage']
        self._raster_storage = self.cfg['raster_storage']
        self._initial_fdir = self.cfg['initial_fdir']
        self._accum_masks = self.cfg['accum_masks']
        
        self._point_geometry = self.cfg['point_geometry']
        self._watershed_name = self.cfg['watershed_name']

    @property
    def save_storage(self):
        return Path(self._save_storage)

    @property
    def point_geometry(self):
        return Path(self._point_geometry)

    @property
    def raster_storage(self):
        return Path(self._raster_storage)

    @property
    def accum_masks(self):
        return Path(self._accum_masks)

    @property
    def initial_fdir(self):
        return Path(self._initial_fdir)

    @property
    def watershed_storage(self):
        return Path(self._watershed_storage)

    @property
    def watershed_name(self):
        return Path(self._watershed_name)







    @property
    def cmip_path(self):
        return Path(self._cmip_path)

    @property
    def meteo_path(self):
        return Path(self._meteo_path)

    @property
    def layers_path(self):
        return Path(self._layers_path)

    @property
    def ws_path(self):
        return Path(self._ws_path)

    @property
    def id_label_path(self):
        return Path(self._id_label_path)

    @property
    def forecast_predictors(self):
        return self._forecast_predictors

    @property
    def ais_dates(self):
        return self._ais_dates

    @property
    def forecast_dates(self):
        return self._forecast_dates
