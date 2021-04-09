#!/usr/bin/env python
# coding: utf-8

###################
# Import
###################

import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path
import glob

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb

import pickle
import psutil
import random
import os
import time
import sys
import math
from contextlib import contextmanager

###################
# Options
###################

# BASE_PATH = "../."
BASE_PATH = "."

# Number of k-folds
N_SPLITS = 2
SEED = 1234

# Number of features that we use. There are 100 features in all but we do not use them all
NUM_FEATS = 20

# Create log path
LOG_PATH = Path(f"{BASE_PATH}/log/")
LOG_PATH.mkdir(parents=True, exist_ok=True)


###################
# Utilities
###################

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

def score_log(df: pd.DataFrame, num_files: int, nam_file: str, data_shape: tuple, n_fold: int, seed: int, mpe: float):
    score_dict = {'n_files': num_files, 'file_name': nam_file, 'shape': data_shape, 'fold': n_fold, 'seed': seed,
                  'score': mpe}
    # noinspection PyTypeChecker
    df = pd.concat([df, pd.DataFrame.from_dict([score_dict])])
    df.to_csv(LOG_PATH / f"log_score.csv", index=False)
    return df


######################################
# Model
######################################

###################
# Initial setup
###################

subm = pd.read_csv(f'{BASE_PATH}/input/indoor-location-navigation/sample_submission.csv', index_col=0)
floor_submission = pd.read_csv(f'{BASE_PATH}/input/simple-99-accurate-floor-model/submission.csv')
feature_dir = f"{BASE_PATH}/input/indoor-unified-wifi-ds"
# feature_dir = f"{BASE_PATH}/input/indoor-navigation-and-location-wifi-features"

train_files = sorted(glob.glob(os.path.join(feature_dir, '*_train.csv')))
test_files = sorted(glob.glob(os.path.join(feature_dir, '*_test.csv')))

###################
# Preprocess
###################

# Training target features

BSSID_FEATS = [f'bssid_{i}' for i in range(NUM_FEATS)]
RSSI_FEATS = [f'rssi_{i}' for i in range(NUM_FEATS)]

# Get data
with open(f'{feature_dir}/train_all.pkl', 'rb') as f:
    data = pickle.load(f)

with open(f'{feature_dir}/test_all.pkl', 'rb') as f:
    test_data = pickle.load(f)

# Get numbers of bssids to embed them in a layer

wifi_bssids = []
for i in range(100):
    wifi_bssids.extend(data.iloc[:, i].values.tolist())
wifi_bssids = list(set(wifi_bssids))

wifi_bssids_size = len(wifi_bssids)
print(f'BSSID TYPES: {wifi_bssids_size}')

wifi_bssids_test = []
for i in range(100):
    wifi_bssids_test.extend(test_data.iloc[:, i].values.tolist())
wifi_bssids_test = list(set(wifi_bssids_test))

wifi_bssids_size = len(wifi_bssids_test)
print(f'BSSID TYPES: {wifi_bssids_size}')

wifi_bssids.extend(wifi_bssids_test)
wifi_bssids_size = len(wifi_bssids)

# Add floor predictions

le = LabelEncoder()
le.fit(wifi_bssids)
le_site = LabelEncoder()
le_site.fit(data['site_id'])

ss = StandardScaler()
ss.fit(data.loc[:, RSSI_FEATS + ['floor']])

data.loc[:, RSSI_FEATS + ['floor']] = ss.transform(data.loc[:, RSSI_FEATS + ['floor']])
for i in BSSID_FEATS:
    data.loc[:, i] = le.transform(data.loc[:, i])
    data.loc[:, i] = data.loc[:, i] + 1

data.loc[:, 'site_id'] = le_site.transform(data.loc[:, 'site_id'])

data.loc[:, RSSI_FEATS + ['floor']] = ss.transform(data.loc[:, RSSI_FEATS + ['floor']])

test_data['floor'] = floor_submission['floor'].values

test_data.loc[:, RSSI_FEATS + ['floor']] = ss.transform(test_data.loc[:, RSSI_FEATS + ['floor']])
for i in BSSID_FEATS:
    test_data.loc[:, i] = le.transform(test_data.loc[:, i])
    test_data.loc[:, i] = test_data.loc[:, i] + 1

test_data.loc[:, 'site_id'] = le_site.transform(test_data.loc[:, 'site_id'])

test_data.loc[:, RSSI_FEATS + ['floor']] = ss.transform(test_data.loc[:, RSSI_FEATS + ['floor']])

site_count = len(data['site_id'].unique())
data.reset_index(drop=True, inplace=True)

set_seed(SEED)

###################
# LGBM parameters
###################

# Parameters for X and Y position training
lgb_params = {'objective': 'root_mean_squared_error',
              'boosting_type': 'gbdt',
              'n_estimators': 150,
              'learning_rate': 0.1,
              'num_iterations': 5000,
              'num_leaves': 90,
              'min_data_in_leaf': 2,
              'colsample_bytree': 0.4,
              'subsample': 0.6,
              'subsample_freq': 2,
              'bagging_seed': SEED,
              'reg_alpha': 8,
              'reg_lambda': 2,
              'random_state': SEED,
              'n_jobs': -1,
              'max_bin': 255
              }

set_seed(SEED)

# Splitting the data into train and validation and passed them through LGBm and then saving the predictions into pred

score_df = pd.DataFrame()
predictions = list()

oof = list()
oof_x, oof_y, oof_f = np.zeros(data.shape[0]), np.zeros(data.shape[0]), np.zeros(data.shape[0])
preds_x, preds_y = 0, 0
preds_f_arr = np.zeros((test_data.shape[0], N_SPLITS))

score_total = 0

site_ids = data['site_id'].unique().tolist()

for site_id in site_ids:
    site_data = data.loc[data['site_id'] == site_id]
    test_site_data = test_data[test_data['site_id'] == site_id]
    test_pred = test_site_data.loc[:, BSSID_FEATS + RSSI_FEATS + ['floor']]
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(site_data.loc[:, 'path'],
                                                                         site_data.loc[:, 'path'])

    for fold, (trn_idx, val_idx) in enumerate(kf):
        X_train = site_data.loc[trn_idx, BSSID_FEATS + RSSI_FEATS + ['floor']]
        y_trainx = site_data.loc[trn_idx, 'x']
        y_trainy = site_data.loc[trn_idx, 'y']
        y_trainf = site_data.loc[trn_idx, 'floor']

        X_valid = site_data.loc[val_idx, BSSID_FEATS + RSSI_FEATS + ['floor']]
        y_validx = site_data.loc[val_idx, 'x']
        y_validy = site_data.loc[val_idx, 'y']
        y_validf = site_data.loc[val_idx, 'floor']

        tmp = pd.concat([y_validx, y_validy], axis=1)
        # y_valid = [tmp, y_validf]
        y_valid = tmp

        modelx = lgb.LGBMRegressor(**lgb_params)
        with timer("fit X"):
            modelx.fit(X_train, y_trainx,
                       eval_set=[(X_valid, y_validx)],
                       eval_metric='rmse',
                       verbose=True,
                       early_stopping_rounds=5
                       )

        modely = lgb.LGBMRegressor(**lgb_params)
        with timer("fit Y"):
            modely.fit(X_train, y_trainy,
                       eval_set=[(X_valid, y_validy)],
                       eval_metric='rmse',
                       verbose=True,
                       early_stopping_rounds=5
                       )

        oof_x[val_idx] = modelx.predict(X_valid)
        oof_y[val_idx] = modely.predict(X_valid)
        #oof_f[val_idx] = y_validf.to_numpy()

        preds_x += modelx.predict(test_pred)
        preds_y += modely.predict(test_pred)

        xy_score = comp_metric(oof_x[val_idx], oof_y[val_idx], y_validx.to_numpy(), y_validy.to_numpy())
        score_total += xy_score

        print("*+" * 40)
        print(f"fold {fold} - XY_RMSE: {xy_score}")
        print("*+" * 40)

    print("*+" * 40)
    print(f"average mean position error {score_total / fold + 1}")
    print("*+" * 40)

    preds_x /= (fold + 1)
    preds_y /= (fold + 1)

    preds_f_mode = stats.mode(preds_f_arr, axis=1)
    # preds_f = preds_f_mode[0].astype(int).reshape(-1)
    preds_f = test_site_data['floor']
    test_preds = pd.DataFrame(np.stack((preds_f, preds_x, preds_y))).T
    test_preds.columns = subm.columns
    test_preds.index = test_site_data["site_path_timestamp"]
    test_preds["floor"] = test_preds["floor"].astype(int)
    predictions.append(test_preds)

# Save the predictions into the same format as required

all_preds = pd.concat(predictions)
all_preds = all_preds.reindex(subm.index)

all_preds['floor'] = floor_submission['floor'].values

all_preds.to_csv('submission.csv')
all_preds.head(20)
