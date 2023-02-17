from scripts.loaders import Icon_loader
from pathlib import Path
from schedule import repeat, every, run_pending
from datetime import datetime
import gc

day = datetime.today().strftime('%Y-%m-%d')
h_m = datetime.today().strftime('%H-%M')


@repeat(every().day.at("04:20"))
def downloader():
    print(f'It is time to download icon for {day} at {h_m}')
    loader = Icon_loader(
        icon_storage='https://opendata.dwd.de/weather/nwp/icon/grib',
        icon_times=['00'],
        icon_variables=['tmax_2m', 'tmin_2m',
                        'tot_prec', 'alb_rad'],
        i_t='icon_global_icosahedral_single-level',
        res_storage=Path('/home/anton/dima_experiments/geo_data/icon_data'))

    loader.download_bz2()
    loader.unarchive_bz2()
    loader.bz2_nc_conversion()
    loader.nc_aggregation()
    loader.clean_temp_files()
    # clear garbage
    gc.collect()


while True:
    run_pending()
