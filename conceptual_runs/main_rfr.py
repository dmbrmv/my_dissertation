import glob
import pandas as pd
import joblib
from copy import deepcopy
from scripts.data_readers import read_gauge, nse, feature_target
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from tqdm import tqdm
pd.options.mode.chained_assignment = None

calibration_place = Path('./cal_res')
rfr_calibration = Path(f'{calibration_place}/rfr_cv')
rfr_calibration.mkdir(exist_ok=True, parents=True)

rfr_df = pd.DataFrame()
ready_rfr = [i.split('/')[-1][:-4]
             for i in glob.glob('./cal_res/rfr/*.joblib')]

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['log2', 'sqrt', 1.0]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]  # Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

for i, gauge in enumerate(tqdm(
    [i.split('/')[-1][:-3] for i in
     glob.glob('../geo_data/great_db/nc_all_q/*.nc')])):
    try:
        train, test = read_gauge(gauge_id=gauge, simple=True)
        # set data
        X, Y = feature_target(deepcopy(train))
        x_test, y_test = feature_target(deepcopy(test))
        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rfr = RandomForestRegressor()
        # Random search of parameters, using 3 fold cross validation,
        # # search across 100 different combinations,
        # and use all available cores
        rfr_model = RandomizedSearchCV(estimator=rfr,
                                       param_distributions=random_grid,
                                       n_iter=100,
                                       cv=3, verbose=2,
                                       random_state=42, n_jobs=-1)

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
        with open('./rfr_error_cv.txt', 'w') as f:
            f.write(''.join(f'{gauge}\n'))
            continue

rfr_df.to_csv('./res_rfr_cv.csv', index=False)
