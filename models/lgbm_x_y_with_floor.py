#!/usr/bin/env python
# coding: utf-8
##########################################
# Copied from: https://www.kaggle.com/ghaiyur/lightgbm-regressor
##########################################

import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path
import glob

from pandas import DataFrame
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb

import psutil
import random
import os
import time
import sys
import math
from contextlib import contextmanager

##################
# Variables
##################

N_SPLITS = 10
SEED = 1234

base_path = '.'


# base_path = '../.'

##################
# Utilities
##################

@contextmanager
def timer(name: str):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 30
    try:
        yield
    finally:
        m1 = p.memory_info()[0] / 2. ** 30
        delta = m1 - m0
        sign = '+' if delta >= 0 else '-'
        delta = math.fabs(delta)
        print(f"[{m1:.1f}GB({sign}{delta:.1f}GB): {time.time() - t0:.3f}sec] {name}", file=sys.stderr)


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def comp_metric(xhat, yhat, fhat, x, y, f):
    intermediate = np.sqrt(np.power(xhat - x, 2) + np.power(yhat - y, 2)) + 15 * np.abs(fhat - f)
    return intermediate.sum() / xhat.shape[0]


def comp_metric_position_only(xhat, yhat, x, y):
    intermediate = np.sqrt(np.power(xhat - x, 2) + np.power(yhat - y, 2))
    return intermediate.sum() / xhat.shape[0]


##################
# Load files
##################

set_seed(SEED)
feature_dir = f"{base_path}/input/indoor-navigation-and-location-wifi-features"
train_files = sorted(glob.glob(os.path.join(feature_dir, '*_train.csv')))
test_files = sorted(glob.glob(os.path.join(feature_dir, '*_test.csv')))
subm = pd.read_csv(f'{base_path}/input/indoor-location-navigation/sample_submission.csv', index_col=0)
floor_predictions = pd.read_csv(f'{base_path}/input/simple-99-accurate-floor-model/submission.csv', index_col=0)

##################
# LGBM parameters
##################

lgb_params = {'objective': 'root_mean_squared_error',
              'boosting_type': 'gbdt',
              'n_estimators': 50000,
              'learning_rate': 0.1,
              'num_leaves': 90,
              'colsample_bytree': 0.4,
              'subsample': 0.6,
              'subsample_freq': 2,
              'bagging_seed': SEED,
              'reg_alpha': 8,
              'reg_lambda': 2,
              'random_state': SEED,
              'n_jobs': -1
              }

##################
# Train the model
##################

predictions = list()
total_xy_rmse_error = 0
for n_files, file in enumerate(train_files):
    data = pd.read_csv(file, index_col=0)
    test_data = pd.read_csv(test_files[n_files], index_col=0)

    # Add floor predictions to test data
    floor_predictions_series = []
    for i in range(test_data.shape[0]):
        site_path_timestamp = test_data.iloc[i, -1]
        splitted = site_path_timestamp.split('_')
        site_path = "_".join(splitted[:-1])
        # Get prediction for the specified site path combination
        pred = floor_predictions.loc[floor_predictions.index.astype(str).str.startswith(site_path)]
        floor = pred.iloc[0, 0]
        floor_predictions_series.append(floor)
    test_data.insert(test_data.shape[1] - 1, 'f', floor_predictions_series)

    print("*+" * 40)
    print(f"building {n_files + 1}")
    print("*+" * 40)

    oof_x, oof_y = np.zeros(data.shape[0]), np.zeros(data.shape[0])
    preds_x, preds_y = 0, 0
    preds_f_arr = np.zeros((test_data.shape[0], N_SPLITS))

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (trn_idx, val_idx) in enumerate(kf.split(data.loc[:, 'path'], data.loc[:, 'path'])):
        # Train set
        x_train_set = data.iloc[trn_idx, :-4]
        x_train_set['f'] = data.iloc[trn_idx, -2]
        y_train_set_x = data.iloc[trn_idx, -4]
        y_train_set_y = data.iloc[trn_idx, -3]
        # Validation set
        x_val_set = data.iloc[val_idx, :-4]
        x_val_set['f'] = data.iloc[val_idx, -2]
        y_val_set_x = data.iloc[val_idx, -4]
        y_val_set_y = data.iloc[val_idx, -3]

        # Train X coordinate
        modelx = lgb.LGBMRegressor(**lgb_params)
        with timer("fit X"):
            modelx.fit(x_train_set, y_train_set_x,
                       eval_set=[(x_val_set, y_val_set_x)],
                       eval_metric='rmse',
                       verbose=True,
                       early_stopping_rounds=5
                       )
        # Train Y coordinate
        modely = lgb.LGBMRegressor(**lgb_params)
        with timer("fit Y"):
            modely.fit(x_train_set, y_train_set_y,
                       eval_set=[(x_val_set, y_val_set_y)],
                       eval_metric='rmse',
                       verbose=True,
                       early_stopping_rounds=5
                       )
        # Make validation
        oof_x[val_idx] = modelx.predict(x_val_set)
        oof_y[val_idx] = modely.predict(x_val_set)
        # Make predictions
        preds_x += modelx.predict(test_data.iloc[:, :-1]) / N_SPLITS
        preds_y += modely.predict(test_data.iloc[:, :-1]) / N_SPLITS
        # Calculate XY RMSE error for the validation set
        xy_rmse_error = comp_metric_position_only(oof_x[val_idx], oof_y[val_idx], y_val_set_x.to_numpy(),
                                                  y_val_set_y.to_numpy())
        print("*+" * 40)
        print(f"fold {fold + 1} - XY_RMSE: {xy_rmse_error}")
        print("*+" * 40)

    # Calculate XY RMSE error for the validation sets
    xy_rmse_error = comp_metric_position_only(oof_x, oof_y, data.iloc[:, -4].to_numpy(), data.iloc[:, -3].to_numpy())
    print("*+" * 40)
    print(f"total folds: {fold + 1} - XY_RMSE: {xy_rmse_error}")
    print("*+" * 40)
    # Add to total XY RMSE error
    total_xy_rmse_error += xy_rmse_error
    # Add predictions to list
    preds_f_mode = stats.mode(preds_f_arr, axis=1)
    preds_f = preds_f_mode[0].astype(int).reshape(-1)
    test_preds = pd.DataFrame(np.stack((preds_f, preds_x, preds_y))).T
    test_preds.columns = subm.columns
    test_preds.index = test_data["site_path_timestamp"]
    test_preds["floor"] = test_preds["floor"].astype(int)
    predictions.append(test_preds)

print("*+" * 40)
print(f"total XY_RMSE: {total_xy_rmse_error / (n_files + 1)}")
print("*+" * 40)

##################
# Make submission
##################

# Load predictions
all_preds = pd.concat(predictions)
all_preds = all_preds.reindex(subm.index)
# Override floor values with the floor predictions
all_preds['floor'] = floor_predictions['floor'].values
# Save submission
all_preds.to_csv('submission_lgbm_with_floor.csv')
