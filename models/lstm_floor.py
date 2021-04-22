#!/usr/bin/env python
# coding: utf-8
##########################################
# Original from https://www.kaggle.com/jwilliamhughdore/99-80-floor-accurate-model-blstm
##########################################

import glob
import os
import pickle
import random
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from sklearn.model_selection import StratifiedKFold
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


##################
# Load files
##################

feature_dir = f"{base_path}/input/indoor-unified-wifi-ds"
train_files = sorted(glob.glob(os.path.join(feature_dir, '*_train.csv')))
test_files = sorted(glob.glob(os.path.join(feature_dir, '*_test.csv')))
subm = pd.read_csv(f'{base_path}/input/indoor-location-navigation/sample_submission.csv', index_col=0)
actual_subm = pd.read_csv(f'{base_path}/input/submissions/submission_lstm_x_y_with_floor_cm_stw.csv', index_col=0)

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
# Pre-process
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
data.reset_index(drop=True, inplace=True)


##################
# The model
##################

def create_fmodel(input_data):
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

    x = L.Reshape((1, -1))(x)
    x = L.BatchNormalization()(x)
    x = L.Bidirectional(L.LSTM(256, dropout=0.4, recurrent_dropout=0.3, return_sequences=True, activation='tanh'))(x)
    x = L.Bidirectional(L.LSTM(32, dropout=0.4, return_sequences=False, activation='relu'))(x)
    x = L.BatchNormalization()(x)
    x = L.Dense(16, activation='tanh')(x)

    output_layer_1 = L.Dense(11, activation='softmax', name='floor')(x)

    model = M.Model([input_embd_layer, input_layer, input_site_layer],
                    [output_layer_1])

    model.compile(optimizer=tf.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    return model


##################
# Train the model
##################

oof_f = np.zeros(data.shape[0])
preds_f_arr = np.zeros((test_data.shape[0], N_SPLITS))

one_hot = pd.get_dummies(data['floor'])

for fold, (trn_idx, val_idx) in enumerate(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
        data.loc[:, 'path'], data.loc[:, 'path'])):
    # Train set
    x_train_set = data.loc[trn_idx, BSSID_FEATS + RSSI_FEATS + ['site_id']]
    y_train_set_f = one_hot.loc[trn_idx, :]
    # Validation set
    x_val_set = data.loc[val_idx, BSSID_FEATS + RSSI_FEATS + ['site_id']]
    y_val_set_f = one_hot.loc[val_idx, :]

    fmodel = create_fmodel(
        [x_train_set.loc[:, BSSID_FEATS], x_train_set.loc[:, RSSI_FEATS], x_train_set.loc[:, 'site_id']])
    fmodel.fit([x_train_set.loc[:, BSSID_FEATS], x_train_set.loc[:, RSSI_FEATS], x_train_set.loc[:, 'site_id']],
               y_train_set_f,
               validation_data=(
                   [x_val_set.loc[:, BSSID_FEATS], x_val_set.loc[:, RSSI_FEATS], x_val_set.loc[:, 'site_id']],
                   y_val_set_f),
               batch_size=128, epochs=100
               , shuffle=True
               , callbacks=[
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4, mode='min')
            , EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5, mode='min', baseline=None,
                            restore_best_weights=True)
        ]
               )
    # Make validation
    val_pred = fmodel.predict(
        [x_val_set.loc[:, BSSID_FEATS], x_val_set.loc[:, RSSI_FEATS], x_val_set.loc[:, 'site_id']])
    val_pred = np.argmax(val_pred, axis=1) - 2
    oof_f[val_idx] = val_pred
    # Make predictions
    pred = fmodel.predict([test_data.loc[:, BSSID_FEATS], test_data.loc[:, RSSI_FEATS], test_data.loc[:, 'site_id']])
    pred = np.argmax(pred, axis=1) - 2  # minus two is make the interval [-2:8] again
    preds_f_arr[:, fold] = pred

    val_floors = data.loc[val_idx, ['floor']]['floor'].tolist()
    wrongs = ((val_floors - val_pred) != 0).tolist()
    accuracy_score = (len(val_floors) - wrongs.count(True)) / len(val_floors)
    print("*+" * 40)
    print(f"fold {fold + 1} - Floor accuracy: {accuracy_score}")
    print("*+" * 40)

val_pred = oof_f
val_floors = data['floor'].tolist()
wrongs = ((val_floors - val_pred) != 0).tolist()
accuracy_score = (len(val_floors) - wrongs.count(True)) / len(val_floors)
print("*+" * 40)
print(f"total folds {fold + 1} - Floor accuracy: {accuracy_score}")
print("*+" * 40)

preds_f_arr = preds_f_arr.astype(int)
floor_predictions = list()
# Find the most frequent value for each prediction between all the folds
for row in preds_f_arr:
    counter = Counter(row)
    floor = counter.most_common()[0][0]
    floor_predictions.append(floor)

##################
# Post-process
##################

# Train & Validatio data
data['pred_floor'] = oof_f.tolist()


def train_filter(a):
    '''returns the mode of the group'''
    return (a['pred_floor'].value_counts().head(1).reset_index()['index'].values[0])


df = pd.DataFrame()
df['fixed_pred_floor'] = data.groupby('path').apply(train_filter)
data = data.merge(df, how='left', on='path')

val_pred = data['fixed_pred_floor'].to_numpy()
val_floors = data['floor'].tolist()
wrongs = ((val_floors - val_pred) != 0).tolist()
accuracy_score = (len(val_floors) - wrongs.count(True)) / len(val_floors)
print("*+" * 40)
print(f"Post process - Floor accuracy: {accuracy_score}")
print("*+" * 40)

# Test data
test_data['path'] = test_data['site_path_timestamp'].str.split(pat='_', n=- 1, expand=True)[1]
test_data['pred_floor'] = floor_predictions


def test_filter(a):
    return (a['pred_floor'].value_counts().head(1).reset_index()['index'].values[0])


dft = pd.DataFrame()
dft['fixed_pred_floor'] = test_data.groupby('path').apply(test_filter)
test_data = test_data.merge(dft, how='left', on='path')

##################
# Make submission
##################

# Create predictions
floor_df = pd.DataFrame(floor_predictions, columns=['floor'])
# Override floor predictions
test_data.index = actual_subm.index
actual_subm['floor'] = test_data['fixed_pred_floor']
# Current date and time
now = datetime.now()
timestamp = datetime.timestamp(now)
# Save submission
actual_subm.to_csv(f'submission_lstm_floor_{timestamp}.csv')
