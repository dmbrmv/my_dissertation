import gc
import glob
from datetime import datetime

from schedule import every, repeat, run_pending
from scripts.loader import Esimo_loader

hdd = "/home/anton/dima_experiments"


@repeat(every().day.at("14:00"))
def downloader():
    """Download ESIMO data daily at 14:00.

    This function is scheduled to run daily at 14:00 and downloads ESIMO data.
    It checks if any CSV files exist in the data directory and either performs
    a first-time download with placement or extends existing data.

    The function handles two scenarios:
    - First download: When no CSV files exist, performs initial data download
      and placement using the Esimo_loader with first_dw=True
    - Data extension: When CSV files exist, downloads new data and extends
      the existing dataset

    After processing, garbage collection is performed to free memory.

    Note:
        This function uses a hardcoded path and is designed to run as a
        scheduled background task.

    """
    day = datetime.today().strftime("%Y-%m-%d")
    h_m = datetime.today().strftime("%H-%M")
    print(f"It is time to download esimo for {day} at {h_m}")

    if not bool(glob.glob(f"{hdd}/data/esimo_data/*.csv")):
        loader = Esimo_loader(
            dw_f=f"{hdd}/my_dissertation/esimo_loader/dw_folder",
            res_f=f"{hdd}/data/esimo_data",
            first_dw=True,
        )
        loader.web_loader()
        loader.first_placement()
    else:
        loader = Esimo_loader(
            dw_f=f"{hdd}/my_dissertation/esimo_loader/dw_folder", res_f=f"{hdd}/data/esimo_data"
        )
        loader.web_loader()
        loader.data_extension()
    # clear garbage
    gc.collect()


while True:
    run_pending()
