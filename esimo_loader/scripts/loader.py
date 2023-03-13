from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import logging
from pathlib import Path
from datetime import datetime
import csv
import pandas as pd


class Esimo_loader:

    def __init__(self,
                 dw_f: str, res_f: str,
                 first_dw: bool = False) -> None:

        self.dw_f = dw_f
        self.dw_f = Path(dw_f)
        self.dw_f.mkdir(exist_ok=True, parents=True)
        self.first_dw = first_dw
        self.res_f = res_f
        self.res_f = Path(res_f)
        self.res_f.mkdir(exist_ok=True, parents=True)
        # constants for one cycle
        self.today_date = datetime.today().strftime('%Y-%m-%d')

        # logger
        self.esimo_dw_logger = logging.getLogger(__name__)
        self.esimo_dw_logger.setLevel(20)

        esimo_dw_handler = logging.FileHandler("esimo_dw_logger.log", mode="a")
        working_filename = Path(__name__).name
        esimo_formatter = logging.Formatter(
            f"{working_filename} %(asctime)s %(levelname)s %(message)s")

        esimo_dw_handler.setFormatter(esimo_formatter)
        self.esimo_dw_logger.addHandler(esimo_dw_handler)

        if first_dw:
            Esimo_loader.first_placement(self)
        else:
            Esimo_loader.data_extension(self)

    def enable_download_headless(self,
                                 browser,
                                 download_dir):
        browser.command_executor._commands["send_command"] = (
            "POST", '/session/$sessionId/chromium/send_command')

        params = {'cmd': 'Page.setDownloadBehavior',
                  'params': {'behavior': 'allow',
                             'downloadPath': download_dir}}
        browser.execute("send_command", params)

    def web_loader(self):
        options = Options()
        # set profile for chrome
        options = Options()
        options.add_argument("--disable-notifications")
        options.add_argument('--no-sandbox')
        options.add_argument('--verbose')
        options.add_experimental_option("prefs", {
            "download.default_directory": f'{self.dw_f}',
            "download.prompt_for_download": False,
            "download.directory_upgrade": True})

        options.add_argument('--disable-gpu')
        options.add_argument('--disable-software-rasterizer')
        options.add_argument('--headless')

        driver = webdriver.Chrome(service=Service("/usr/bin/chromedriver"),
                                  options=options)

        Esimo_loader.enable_download_headless(self,
                                              driver, f"{self.dw_f}")

        esimo = 'http://portal.esimo.ru'
        viewer = 'dataview/viewresource?resourceId=RU_RIHMI-WDC_1325_1'
        website = f'{esimo}/{viewer}'
        driver.get(website)
        time.sleep(240)
        driver.find_element(
            by=By.CSS_SELECTOR,
            value="div.portlet-form-button.portlet-icon.icon-tools").click()
        time.sleep(180)
        driver.find_element(
            by=By.CSS_SELECTOR,
            value="#display-analytics-export-csv").click()
        time.sleep(180)

        driver.quit()
        self.esimo_dw_logger.info(f"Data for {self.today_date} downloaded")

    def first_placement(self):
        # saver
        with open('./data.csv', newline='') as csvfile:
            data = csv.reader(csvfile, delimiter=',', quotechar='"')
            res_str = next(data)

        res_str = [word.replace(',', '') for word in res_str]

        file = pd.read_csv('./data.csv',
                           sep=',', names=res_str,
                           skiprows=1, skipfooter=1,
                           on_bad_lines='skip',
                           quotechar='"',
                           engine='python', encoding='utf-8')

        group_gauge = file.groupby(
            by='Платформа: идентификатор локальный').groups

        for gauge_id, loc_index in group_gauge.items():

            gauge = file.loc[loc_index][['Платформа: идентификатор локальный',
                                        'Дата и время',
                                         'Уровень воды над нулем поста']]
            gauge = gauge.rename(
                columns={'Платформа: идентификатор локальный': 'gauge_id',
                         'Дата и время': 'date',
                         'Уровень воды над нулем поста': 'level'})
            gauge['date'] = pd.to_datetime(gauge['date'])
            gauge = gauge.set_index('date')
            res = gauge[['level']].groupby(by=pd.Grouper(freq='1d')).mean()
            res.to_csv(f'{self.res_f}/{gauge_id}.csv')
        gauge_len = len(group_gauge.keys())
        self.esimo_dw_logger.info(
            f"Initial data was stored in {self.res_f} for {gauge_len} gauges")

    def data_extension(self):
        with open(f'{self.dw_f}/data.csv', newline='') as csv_new_file:
            data = csv.reader(csv_new_file, delimiter=',', quotechar='"')
            res_str = next(data)

        res_str = [word.replace(',', '') for word in res_str]

        new_file = pd.read_csv(f'{self.dw_f}/data.csv',
                               sep=',', names=res_str,
                               skiprows=1, skipfooter=1,
                               on_bad_lines='skip',
                               quotechar='"',
                               engine='python', encoding='utf-8')
        group_gauge = new_file.groupby(
            by='Платформа: идентификатор локальный').groups
        new_g = 0
        exist_g = 0
        for gauge_id, loc_index in group_gauge.items():

            new_gauge = new_file.loc[loc_index][[
                'Платформа: идентификатор локальный',
                'Дата и время',
                'Уровень воды над нулем поста']]
            new_gauge = new_gauge.rename(
                columns={'Платформа: идентификатор локальный': 'gauge_id',
                         'Дата и время': 'date',
                         'Уровень воды над нулем поста': 'level'})
            new_gauge['date'] = pd.to_datetime(new_gauge['date'])
            new_gauge = new_gauge.set_index('date')
            new_res = new_gauge[['level']].groupby(
                by=pd.Grouper(freq='1d')).mean()
            try:
                old_res = pd.read_csv(f'{self.res_f}/{gauge_id}.csv')
                old_res['date'] = pd.to_datetime(old_res['date'])
                old_res = old_res.set_index('date')

                res = old_res.combine_first(new_res)
                res.to_csv(f'{self.res_f}/{gauge_id}.csv')
                exist_g += 1
            except FileNotFoundError:
                new_res.to_csv(f'{self.res_f}/{gauge_id}.csv')
                new_g += 1
        self.esimo_dw_logger.info(
            f"""
\n###########################################################

On {self.today_date} extended data for {exist_g} gauges
and stored data for new {new_g} gauges
###########################################################\n""")
