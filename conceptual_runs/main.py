import glob
import pandas as pd
from scripts.data_readers import read_gauge, get_params
from pathlib import Path

calibration_place = Path('./cal_res')
hbv_calibration = Path(f'{calibration_place}/hbv_full')
hbv_calibration.mkdir(exist_ok=True, parents=True)
gr4j_calibration = Path(f'{calibration_place}/hbv_full')
gr4j_calibration.mkdir(exist_ok=True, parents=True)


ready_hbv = [i.split('/')[-1][:-4]
             for i in glob.glob('./cal_res/hbv_full/*.csv')]
ready_gr4j = [i.split('/')[-1][:-4]
              for i in glob.glob('./cal_res/gr4j_full/*.csv')]
hbv_df = pd.DataFrame()
gr4j_df = pd.DataFrame()
for i, gauge in enumerate([i.split('/')[-1][:-3]
                           for i in
                           glob.glob('../geo_data/great_db/nc_all_q/*.nc')]):

    for model in ['hbv', 'gr4j']:

        if model == 'hbv':
            if gauge not in ready_hbv:
                try:
                    train, test = read_gauge(gauge_id=gauge)
                    hbv_df.loc[i, 'gauge_id'] = gauge
                    hbv_df.loc[i, 'NSE'] = get_params(
                        gauge_id=gauge,
                        model_name=model,
                        iter_number=60000,
                        params_path=hbv_calibration,
                        train=train, test=test,
                        calibrate=True,
                        with_plot=False)
                except Exception as e:
                    with open('./hbv_error_full.txt', 'w') as f:
                        f.write(''.join(f'{gauge}\n'))
                    print(e)
                    continue
            else:
                print(f'{gauge} got an hbv parameters !\n')
                continue

        elif model == 'gr4j':
            if gauge not in ready_gr4j:
                try:
                    train, test = read_gauge(gauge_id=gauge)
                    gr4j_df.loc[i, 'gauge_id'] = gauge
                    gr4j_df.loc[i, 'NSE'] = get_params(
                        gauge_id=gauge,
                        model_name=model,
                        iter_number=60000,
                        params_path=gr4j_calibration,
                        train=train, test=test,
                        calibrate=True,
                        with_plot=False)
                except Exception:
                    with open('./gr4j_error_full.txt', 'w') as f:
                        f.write(''.join(f'{gauge}\n'))
                    continue
            else:
                print(f'{gauge} got an GR4J parameters !\n')
                continue

hbv_df.to_csv('./res_hbv_full.csv', index=False)
gr4j_df.to_csv('./res_gr4j_full.csv', index=False)
