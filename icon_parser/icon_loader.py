from scripts.loaders import Icon_loader
from pathlib import Path
from schedule import repeat, every, run_pending
from datetime import datetime
import gc
import subprocess


@repeat(every().day.at("04:20"))
def downloader():

    day = datetime.today().strftime('%Y-%m-%d')
    h_m = datetime.today().strftime('%H-%M')
    print(f'It is time to download icon for {day} at {h_m}')
    loader = Icon_loader(
        icon_description='./data/static_icon/grid_world_0125.txt',
        icon_weights='./data/static_icon/icon_weights.nc',
        icon_storage='https://opendata.dwd.de/weather/nwp/icon/grib',
        icon_times=['00'],
        icon_variables=['t_2m', 'tmax_2m', 'tmin_2m',
                        'tot_prec', 'alb_rad'],
        i_t='icon_global_icosahedral_single-level',
        coord_limits={'max_lon': 179., 'min_lon': 19.,
                      'max_lat': 72., 'min_lat': 40.},
        res_storage=Path('/home/anton/dima_experiments/geo_data/icon_oct_23'))

    loader.download_bz2()
    loader.unarchive_bz2()
    loader.bz2_nc_conversion()
    loader.nc_aggregation()
    loader.clean_temp_files()
    # clear garbage
    gc.collect()
    subprocess.run(['python', './icon_merger.py'])


while True:
    run_pending()
