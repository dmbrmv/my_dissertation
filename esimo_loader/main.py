from scripts.loader import Esimo_loader
from schedule import repeat, every, run_pending
from datetime import datetime
import gc
import glob

hdd = '/home/anton/dima_experiments'


@repeat(every().day.at("04:20"))
def downloader():

    day = datetime.today().strftime('%Y-%m-%d')
    h_m = datetime.today().strftime('%H-%M')
    print(f'It is time to download esimo for {day} at {h_m}')

    if not bool(glob.glob(f'{hdd}/geo_data/esimo_data/*.csv')):
        loader = Esimo_loader(
            dw_f=f'{hdd}/my_dissertation/esimo_loader/dw_folder',
            res_f=f'{hdd}/geo_data/esimo_data',
            first_dw=True)
        loader.web_loader()
        loader.first_placement()
    else:
        loader = Esimo_loader(
            dw_f=f'{hdd}/my_dissertation/esimo_loader/dw_folder',
            res_f=f'{hdd}/geo_data/esimo_data')
        loader.web_loader()
        loader.data_extension()
    # clear garbage
    gc.collect()


while True:
    run_pending()
