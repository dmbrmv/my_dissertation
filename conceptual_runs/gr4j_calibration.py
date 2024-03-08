import glob
import pandas as pd
from scripts.data_readers import read_gauge, get_params
from pathlib import Path

calibration_place = Path('./cal_res')
gr4j_calibration = Path(f'{calibration_place}/gr4j_full')
gr4j_calibration.mkdir(exist_ok=True, parents=True)

ready_gr4j = [i.split('/')[-1][:-4]
              for i in glob.glob('./cal_res/gr4j_full/*.csv')]

gr4j_df = pd.DataFrame()

for i, gauge in enumerate([i.split('/')[-1][:-3]
                           for i in
                           glob.glob('../geo_data/great_db/nc_all_q/*.nc')]):

    if gauge not in ready_gr4j:
        try:
            train, test = read_gauge(gauge_id=gauge)
            gr4j_df.loc[i, 'gauge_id'] = gauge
            gr4j_df.loc[i, 'NSE'] = get_params(
                gauge_id=gauge,
                model_name='gr4j',
                iter_number=24000,
                params_path=gr4j_calibration,
                train=train, test=test,
                calibrate=True,
                with_plot=False)
        except Exception:
            with open('./gr4j_error_full.txt', 'a') as f:
                f.write(''.join(f'{gauge}\n'))
            continue
    else:
        print(f'{gauge} got an gr4j parameters !\n')
        continue

gr4j_df.to_csv('./res_hbv_full.csv', index=False)
