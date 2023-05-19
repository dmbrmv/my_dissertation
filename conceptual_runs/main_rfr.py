import glob
import pandas as pd
import joblib
from copy import deepcopy
from scripts.data_readers import read_gauge, nse, feature_target
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
pd.options.mode.chained_assignment = None

calibration_place = Path('./cal_res')
rfr_calibration = Path(f'{calibration_place}/rfr')
rfr_calibration.mkdir(exist_ok=True, parents=True)

rfr_df = pd.DataFrame()
ready_rfr = [i.split('/')[-1][:-4]
             for i in glob.glob('./cal_res/rfr/*.joblib')]

for i, gauge in enumerate(tqdm(
    [i.split('/')[-1][:-3] for i in
     glob.glob('../geo_data/great_db/nc_all_q/*.nc')])):
    try:
        train, test = read_gauge(gauge_id=gauge, simple=True)
        # set data
        X, Y = feature_target(deepcopy(train))
        x_test, y_test = feature_target(deepcopy(test))
        # setup model
        rfr_model = RandomForestRegressor(random_state=42)
        rfr_model.fit(X, Y)
        # store hyperparameters
        joblib.dump(rfr_model, f"{rfr_calibration}/{gauge}.joblib", compress=2)
        # get prediction
        fin_df = test.iloc[255:, :]
        fin_df['q_mm_rfr'] = rfr_model.predict(x_test)

        res_nse = nse(predictions=fin_df['q_mm_rfr'],
                      targets=fin_df['q_mm_day'])

        rfr_df.loc[i, 'gauge_id'] = gauge
        rfr_df.loc[i, 'NSE'] = res_nse
    except Exception:
        with open('./rfr_error.txt', 'w') as f:
            f.write(''.join(f'{gauge}\n'))
            continue

rfr_df.to_csv('./res_rfr.csv', index=False)
