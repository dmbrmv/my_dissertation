from pathlib import Path

import joblib
import numpy as np
from scripts.data_readers import read_gauge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm


calibration_place = Path("/app/geo_data/conceptual_models")
rfr_calibration = Path(f"{calibration_place}/rfr_cv")
rfr_calibration.mkdir(exist_ok=True, parents=True)
rfr_parameters = Path("/app/geo_data/conceptual_models/rfr_params")
rfr_parameters.mkdir(exist_ok=True, parents=True)
gauges = list(i.stem for i in Path("/app/geo_data/time_series").glob("*.nc"))

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ["log2", "sqrt", 1.0]
# Maximum number of levels in tree
max_depth = [None, 10, 20, 30, 40, 50]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]  # Create the random grid

random_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "bootstrap": bootstrap,
}


def kge(y_true, y_pred):
    sim_mean = np.mean(y_true, axis=0)
    obs_mean = np.mean(y_pred)

    r_num = np.sum((y_true - sim_mean) * (y_pred - obs_mean), axis=0)
    r_den = np.sqrt(
        np.sum((y_true - sim_mean) ** 2, axis=0)
        * np.sum(
            (y_pred - obs_mean) ** 2,
        )
    )
    if r_den == 0:
        r = np.NaN
    else:
        r = r_num / r_den
    # calculate error in spread of flow alpha
    if np.std(y_pred) == 0:
        alpha = np.NaN
    else:
        alpha = np.std(y_true, axis=0) / np.std(y_pred)
    # calculate error in volume beta (bias of mean discharge)
    if np.sum(y_pred) == 0:
        beta = np.NaN
    else:
        beta = np.sum(y_true, axis=0) / np.sum(y_pred)
    # calculate the Kling-Gupta Efficiency KGE
    kge_ = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return kge_


kge_score = make_scorer(kge, greater_is_better=True)

for gauge_id in tqdm(gauges):
    if gauge_id in list(i.stem for i in Path("/app/geo_data/conceptual_models/rfr_cv").glob("*joblib")):
        continue
    train, _ = read_gauge(gauge_id=gauge_id, simple=True)
    # set data

    for days in (2**n for n in range(6)):
        train[[f"prcp_{days}"]] = train[["prcp_e5l"]].rolling(window=days).sum()
        train[[f"t_min_mean_{days}"]] = train[["t_min_e5l"]].rolling(window=days).mean()
        train[[f"t_max_mean_{days}"]] = train[["t_min_e5l"]].rolling(window=days).mean()
    train = train.dropna()

    feature_cols = [
        item
        for sublist in [
            [f"{var}_{day}" for day in (2**n for n in range(6))]
            for var in ["prcp", "t_min_mean", "t_max_mean"]  # "t_min_sum", "t_max_sum"]
        ]
        for item in sublist
    ]
    X_train = train[feature_cols]

    # feature_train_mean = X_train.mean()
    # feature_train_std = X_train.std()
    # coef_df.loc[gauge_id, "feature_mean"] = feature_train_mean.values[0]
    # coef_df.loc[gauge_id, "feature_std"] = feature_train_std.values[0]
    # normalize features
    # X_train = (X_train - feature_train_mean) / feature_train_std

    X_train = X_train.to_numpy()
    # create target
    Y_train = train[["q_mm_day"]]

    # target_train_mean = Y_train.mean()
    # target_train_std = Y_train.std()
    # coef_df.loc[gauge_id, "target_mean"] = target_train_mean.values[0]
    # coef_df.loc[gauge_id, "target_std"] = target_train_std.values[0]
    # Y_train = (Y_train - target_train_mean) / target_train_std

    Y_train = Y_train.to_numpy().ravel()

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rfr = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # # search across 100 different combinations,
    # and use all available cores
    rfr_model = RandomizedSearchCV(
        estimator=rfr,
        param_distributions=random_grid,
        n_iter=14,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        scoring=kge_score,
    )

    rfr_model.fit(X_train, Y_train)
    # Оценка на тестовых данных
    best_model = rfr_model.best_estimator_
    # store hyperparameters
    joblib.dump(best_model, f"{rfr_calibration}/{gauge_id}.joblib", compress=2)
    # coef_df.index.name = "gauge_id"
    # coef_df.to_csv(f"{rfr_parameters}/{gauge_id}.csv")
