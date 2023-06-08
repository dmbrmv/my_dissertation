import glob
import pandas as pd
from scripts.data_readers import read_gauge, get_params
from pathlib import Path

calibration_place = Path('./cal_res')
hbv_calibration = Path(f'{calibration_place}/hbv_sceua')
hbv_calibration.mkdir(exist_ok=True, parents=True)
gr4j_calibration = Path(f'{calibration_place}/gr4j_sceua')
gr4j_calibration.mkdir(exist_ok=True, parents=True)


hbv_df = pd.DataFrame()
ready_hbv = [i.split('/')[-1][:-4] for i in glob.glob('./cal_res/hbv/*.csv')]
ready_gr4j = [i.split('/')[-1][:-4] for i in glob.glob('./cal_res/gr4j/*.csv')]
gr4j_df = pd.DataFrame()
for i, gauge in enumerate([i.split('/')[-1][:-3]
                           for i in
                           glob.glob('../geo_data/great_db/nc_all_q/*.nc')]):

    for model in ['hbv', 'gr4j']:

        if model == 'hbv':
            if gauge in ready_hbv:
                print(f'{gauge} got an hbv parameters !\n')
                continue
            else:
                try:
                    train, test = read_gauge(gauge_id=gauge)
                    hbv_df.loc[i, 'gauge_id'] = gauge
                    hbv_df.loc[i, 'NSE'] = get_params(
                        gauge_id=gauge,
                        model_name=model,
                        params_path=hbv_calibration,
                        train=train, test=test,
                        with_plot=False)
                except Exception:
                    with open('./hbv_error.txt', 'w') as f:
                        f.write(''.join(f'{gauge}\n'))
                    continue
        elif model == 'gr4j':
            if gauge in ready_gr4j:
                print(f'{gauge} got an GR4J parameters !\n')
                continue
            else:
                try:
                    train, test = read_gauge(gauge_id=gauge)
                    gr4j_df.loc[i, 'gauge_id'] = gauge
                    gr4j_df.loc[i, 'NSE'] = get_params(
                        gauge_id=gauge,
                        model_name=model,
                        params_path=gr4j_calibration,
                        train=train, test=test,
                        with_plot=False)
                except Exception:
                    with open('./gr4j_error.txt', 'w') as f:
                        f.write(''.join(f'{gauge}\n'))
                    continue
hbv_df.to_csv('./res_hbv_2.csv', index=False)
gr4j_df.to_csv('./res_gr4j_2.csv', index=False)
