#!/usr/bin/env python
# coding: utf-8
##########################################
# Copied from https://www.kaggle.com/kokitanisaka/lstm-by-keras-with-unified-wi-fi-feats
##########################################

import glob
import os
import pickle
import random
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats as stats
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

##################
# Variables
##################

N_SPLITS = 10
SEED = 2021
NUM_FEATS = 100  # number of features that we use. there are 100 feats but we don't need to use all of them
base_path = '.'


##################
# Utilities
##################

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=0,
        inter_op_parallelism_threads=0
    )
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)


def comp_metric(xhat, yhat, fhat, x, y, f):
    intermediate = np.sqrt(np.power(xhat - x, 2) + np.power(yhat - y, 2)) + 15 * np.abs(fhat - f)
    return intermediate.sum() / xhat.shape[0]


def comp_metric_position_only(xhat, yhat, x, y):
    intermediate = np.sqrt(np.power(xhat - x, 2) + np.power(yhat - y, 2))
    return intermediate.sum() / xhat.shape[0]


##################
# Load files
##################

feature_dir = f"{base_path}/input/indoor-unified-wifi-ds"
train_files = sorted(glob.glob(os.path.join(feature_dir, '*_train.csv')))
test_files = sorted(glob.glob(os.path.join(feature_dir, '*_test.csv')))
subm = pd.read_csv(f'{base_path}/input/indoor-location-navigation/sample_submission.csv', index_col=0)
floor_predictions = pd.read_csv(f'{base_path}/input/simple-99-accurate-floor-model/submission.csv')

with open(f'{feature_dir}/train_all.pkl', 'rb') as f:
    data = pickle.load(f)

with open(f'{feature_dir}/test_all.pkl', 'rb') as f:
    test_data = pickle.load(f)

##################
# Load features
##################

BSSID_FEATS = [f'bssid_{i}' for i in range(NUM_FEATS)]
RSSI_FEATS = [f'rssi_{i}' for i in range(NUM_FEATS)]

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

##################
# Preprocess
##################

le = LabelEncoder()
le.fit(wifi_bssids)
le_site = LabelEncoder()
le_site.fit(data['site_id'])

ss = StandardScaler()
ss.fit(data.loc[:, RSSI_FEATS])

data.loc[:, RSSI_FEATS] = ss.transform(data.loc[:, RSSI_FEATS])
for i in BSSID_FEATS:
    data.loc[:, i] = le.transform(data.loc[:, i])
    data.loc[:, i] = data.loc[:, i] + 1

data.loc[:, 'site_id'] = le_site.transform(data.loc[:, 'site_id'])

data.loc[:, RSSI_FEATS] = ss.transform(data.loc[:, RSSI_FEATS])

test_data.loc[:, RSSI_FEATS] = ss.transform(test_data.loc[:, RSSI_FEATS])
for i in BSSID_FEATS:
    test_data.loc[:, i] = le.transform(test_data.loc[:, i])
    test_data.loc[:, i] = test_data.loc[:, i] + 1

test_data.loc[:, 'site_id'] = le_site.transform(test_data.loc[:, 'site_id'])

test_data.loc[:, RSSI_FEATS] = ss.transform(test_data.loc[:, RSSI_FEATS])

site_count = len(data['site_id'].unique())
print(f'SITE COUNT: {site_count}')
data.reset_index(drop=True, inplace=True)

set_seed(SEED)


##################
# The model
##################

def create_model(input_data):
    # BSSID feats
    input_dim = input_data[0].shape[1]
    input_embd_layer = L.Input(shape=(input_dim,))
    x1 = L.Embedding(wifi_bssids_size, 64)(input_embd_layer)
    x1 = L.Flatten()(x1)

    # RSSI feats
    input_dim = input_data[1].shape[1]
    input_layer = L.Input(input_dim, )
    x2 = L.BatchNormalization()(input_layer)
    x2 = L.Dense(NUM_FEATS * 64, activation='relu')(x2)

    # Site
    input_site_layer = L.Input(shape=(1,))
    x3 = L.Embedding(site_count, 1)(input_site_layer)
    x3 = L.Flatten()(x3)

    # Main stream
    x = L.Concatenate(axis=1)([x1, x3, x2])

    x = L.BatchNormalization()(x)
    x = L.Dropout(0.3)(x)
    x = L.Dense(256, activation='relu')(x)

    x = L.Reshape((1, -1))(x)
    x = L.BatchNormalization()(x)
    x = L.LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True, activation='relu')(x)
    x = L.LSTM(16, dropout=0.1, return_sequences=False, activation='relu')(x)

    output_layer_1 = L.Dense(2, name='xy')(x)

    model = M.Model([input_embd_layer, input_layer, input_site_layer],
                    [
                        output_layer_1
                    ])
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='mse',
                  metrics=['mse'])

    return model


##################
# Train the model
##################

score_df = pd.DataFrame()
oof = list()
predictions = list()

oof_x, oof_y, oof_f = np.zeros(data.shape[0]), np.zeros(data.shape[0]), np.zeros(data.shape[0])
preds_x, preds_y = 0, 0
preds_f_arr = np.zeros((test_data.shape[0], N_SPLITS))

for fold, (trn_idx, val_idx) in enumerate(
        KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(data.loc[:, 'path'], data.loc[:, 'path'])):
    # Train set
    x_train_set = data.loc[trn_idx, BSSID_FEATS + RSSI_FEATS + ['site_id']]
    y_train_set_x = data.loc[trn_idx, 'x']
    y_train_set_y = data.loc[trn_idx, 'y']
    y_train_set_f = data.loc[trn_idx, 'floor']
    tmp = pd.concat([y_train_set_x, y_train_set_y], axis=1)
    y_train = [tmp]

    # Validation set
    x_val_set = data.loc[val_idx, BSSID_FEATS + RSSI_FEATS + ['site_id']]
    y_val_set_x = data.loc[val_idx, 'x']
    y_val_set_y = data.loc[val_idx, 'y']
    y_val_set_f = data.loc[val_idx, 'floor']
    tmp = pd.concat([y_val_set_x, y_val_set_y], axis=1)
    y_valid = [tmp]

    # Create model
    model = create_model(
        [x_train_set.loc[:, BSSID_FEATS], x_train_set.loc[:, RSSI_FEATS], x_train_set.loc[:, 'site_id']])
    # Train X and Y coordinate
    history = model.fit(
        [x_train_set.loc[:, BSSID_FEATS], x_train_set.loc[:, RSSI_FEATS], x_train_set.loc[:, 'site_id']], y_train,
        validation_data=(
            [x_val_set.loc[:, BSSID_FEATS], x_val_set.loc[:, RSSI_FEATS], x_val_set.loc[:, 'site_id']],
            y_valid),
        batch_size=128,
        epochs=1000,
        callbacks=[
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4,
                              mode='min')
            , EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5, mode='min', baseline=None,
                            restore_best_weights=True)
        ])
    # Make validation
    val_pred = model.predict([x_val_set.loc[:, BSSID_FEATS], x_val_set.loc[:, RSSI_FEATS], x_val_set.loc[:, 'site_id']])
    oof_x[val_idx] = val_pred[:, 0]
    oof_y[val_idx] = val_pred[:, 1]
    # Make predictions
    pred = model.predict([test_data.loc[:, BSSID_FEATS], test_data.loc[:, RSSI_FEATS], test_data.loc[:, 'site_id']])
    preds_x += pred[:, 0]
    preds_y += pred[:, 1]
    # Calculate XY RMSE error for the validation set
    xy_score = comp_metric_position_only(oof_x[val_idx], oof_y[val_idx],
                                         y_val_set_x.to_numpy(), y_val_set_y.to_numpy())
    print("*+" * 40)
    print(f"fold {fold + 1} - XY_RMSE: {xy_score}")
    print("*+" * 40)

# Get the average of the predictions
preds_x /= (fold + 1)
preds_y /= (fold + 1)

# Calculate XY RMSE error for the validation sets
xy_score = comp_metric_position_only(oof_x, oof_y, data.iloc[:, -5].to_numpy(), data.iloc[:, -4].to_numpy())
print("*+" * 40)
print(f"total folds {fold + 1} - XY_RMSE: {xy_score}")
print("*+" * 40)
# Add predictions to list
preds_f_mode = stats.mode(preds_f_arr, axis=1)
preds_f = preds_f_mode[0].astype(int).reshape(-1)
test_preds = pd.DataFrame(np.stack((preds_f, preds_x, preds_y))).T
test_preds.columns = subm.columns
test_preds.index = test_data["site_path_timestamp"]
test_preds["floor"] = test_preds["floor"].astype(int)
predictions.append(test_preds)

##################
# Make submission
##################

# Create predictions
all_preds = pd.concat(predictions)
all_preds = all_preds.reindex(subm.index)
# Override floor predictions
all_preds['floor'] = floor_predictions['floor'].values
# Current date and time
now = datetime.now()
timestamp = datetime.timestamp(now)
# Save submission
all_preds.to_csv(f'submission_lstm_x_y_{timestamp}.csv')
