from pathlib import Path
import logging
import os
import shutil
from datetime import datetime
import bz2
from urllib import request
from urllib.error import HTTPError
import glob

from .editors import triple_position_number
from .nc_proc import from_icon_to_ru_nc


class Icon_loader:

    def __init__(self,
                 icon_storage: str, icon_times: list,
                 icon_variables: list, i_t: str,
                 res_storage: Path) -> None:

        # initial values
        self.icon_storage = icon_storage
        self.res_storage = res_storage
        self.icon_times = icon_times
        self.icon_variables = icon_variables
        self.i_t = i_t

        # constants for one cycle
        self.today_date = datetime.today().strftime('%Y%m%d')
        self.desc_f = './data/static_icon/grid_world_0125.txt'
        self.weight_f = './data/static_icon/weights_icon_world.nc'

        self.coord_limits = {'max_lon': 179., 'min_lon': 19.,
                             'max_lat': 72., 'min_lat': 40.}

        # logger
        self.icon_dw_logger = logging.getLogger(__name__)
        self.icon_dw_logger.setLevel(20)

        icon_dw_handler = logging.FileHandler("icon_dw_logger.log", mode="a")
        working_filename = Path(__name__).name
        forecaster_formatter = logging.Formatter(
            f"{working_filename} %(asctime)s %(levelname)s %(message)s")

        icon_dw_handler.setFormatter(forecaster_formatter)
        self.icon_dw_logger.addHandler(icon_dw_handler)

    def download_bz2(self):
        icon_model_time = [triple_position_number(val)
                           for val in range(0, 181)]
        # download every evailable value for modeled time
        for time in self.icon_times:
            for var in self.icon_variables:
                self.icon_dw_logger.info(f"Variable {var} is downloading ..")
                for m_time in icon_model_time:
                    self.bz2_path = Path('./data/icon_bz2')
                    self.bz2_path.mkdir(exist_ok=True, parents=True)

                    bz = Path(f'{self.bz2_path}/{time}/{var}')
                    bz.mkdir(exist_ok=True, parents=True)

                    i_tag = f'{self.today_date}{time}_{m_time}'
                    i_link = f'{self.icon_storage}/{time}/{var}/{self.i_t}'

                    dw_url = f'{i_link}_{i_tag}_{var.upper()}.grib2.bz2'
                    try:
                        request.urlretrieve(
                            dw_url,
                            f'{bz}/{self.i_t}_{i_tag}_{var.upper()}.grib2.bz2')
                    except HTTPError:
                        # self.icon_dw_logger.warning(
                        #     f"No file for at {dw_url}")
                        continue
                self.icon_dw_logger.info(
                    f"Variable {var} has been downloaded\n")

    def unarchive_bz2(self):
        # unarchive bz2 files in each folder with correspond variable
        for time in self.icon_times:
            for var in self.icon_variables:
                self.icon_dw_logger.info(f"Unarchive {var} ..")
                files = glob.glob(f'./data/icon_bz2/{time}/{var}/*.bz2')

                for filename in files:
                    f_name = filename.split('/')[-1][:-4]

                    self.grib2_folder = Path('./data/icon_grib2')
                    self.grib2_folder.mkdir(exist_ok=True, parents=True)

                    grib = Path(f'{self.grib2_folder}/{time}/{var}')
                    grib.mkdir(exist_ok=True, parents=True)

                    fin_file = f'{grib}/{f_name}'

                    with open(fin_file, 'wb') as nw_f, bz2.BZ2File(filename, 'rb') as f:
                        for data in iter(lambda: f.read(100 * 1024), b''):
                            nw_f.write(data)
                self.icon_dw_logger.info(
                    f"Unarchive for {var} has been finished\n")

    def bz2_nc_conversion(self):
        for time in self.icon_times:
            for var in self.icon_variables:
                self.icon_dw_logger.info(f"Nc coversion for {var} ..")
                files = glob.glob(f'./data/icon_grib2/{time}/{var}/*.grib2')
                self.nc_folder = Path('./data/icon_nc')
                self.nc_folder.mkdir(exist_ok=True, parents=True)

                for filename in files:
                    out_f = filename.split('/')[-1][:-6]
                    nc_temp = Path(f'{self.nc_folder}/{time}/{var}')
                    nc_temp.mkdir(exist_ok=True, parents=True)
                    fin_f = f'{nc_temp}/{out_f}'

                    os.system(
                        f'cdo -s -f nc remap,{self.desc_f},{self.weight_f} {filename} {fin_f}.nc')
                self.icon_dw_logger.info(
                    f"Nc coversion for {var} has been finished\n")

    def nc_aggregation(self):
        for time in self.icon_times:
            for var in self.icon_variables:
                self.icon_dw_logger.info(f"Nc aggregation for {var} ..")
                _ = from_icon_to_ru_nc(
                    path_to_nc=self.nc_folder,
                    path_to_result=self.res_storage,
                    var=var,
                    icon_time=time,
                    coord_limits=self.coord_limits)
                self.icon_dw_logger.info(
                    f"Nc aggregation for {var} has been finished\n")

    def clean_temp_files(self):

        shutil.rmtree(self.bz2_path)
        shutil.rmtree(self.grib2_folder)
        shutil.rmtree(self.nc_folder)
        self.icon_dw_logger.info(
            f"Data for {self.today_date} downloaded\n")
