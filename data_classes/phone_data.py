from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    train_data: list
    test_data: list


@dataclass
class Site:
    def __init__(self, site_id):
        self.site_id = site_id
        self.paths = []

    site_id: int
    paths: list


@dataclass
class BuildingPath:
    def __init__(self, path_id):
        self.path_id = path_id
        self.data = []

    path_id: int
    data: list
    floor: int


@dataclass
class ReadData:
    def __init__(self):
        pass

    timestamp = 0
    acce_x = 0.0
    acce_y = 0.0
    acce_z = 0.0
    acce_accuracy = 0.0
    acce_uncali_x = 0.0
    acce_uncali_y = 0.0
    acce_uncali_z = 0.0
    acce_bias_x = 0.0
    acce_bias_y = 0.0
    acce_bias_z = 0.0
    acce_uncali_accuracy = 0.0
    magn_x = 0.0
    magn_y = 0.0
    magn_z = 0.0
    magn_accuracy = 0.0
    magn_uncali_x = 0.0
    magn_uncali_y = 0.0
    magn_uncali_z = 0.0
    magn_bias_x = 0.0
    magn_bias_y = 0.0
    magn_bias_z = 0.0
    magn_uncali_accuracy = 0.0
    gyro_x = 0.0
    gyro_y = 0.0
    gyro_z = 0.0
    gyro_accuracy = 0.0
    gyro_uncali_x = 0.0
    gyro_uncali_y = 0.0
    gyro_uncali_z = 0.0
    gyro_bias_x = 0.0
    gyro_bias_y = 0.0
    gyro_bias_z = 0.0
    gyro_uncali_accuracy = 0.0
    ahrs_x = 0.0
    ahrs_y = 0.0
    ahrs_z = 0.0
    ahrs_accuracy = 0.0
    wifi_ssid = ""
    wifi_bssid = ""
    wifi_rssi = 0
    wifi_freq = 0
    wifi_ls_ts = 0.0
    uuid = ""
    majorid = ""
    minorid = ""
    txpow = ""
    beacon_rssi = 0
    distance = 0.0
    macaddr = ""
    unix_time = 0.0
    way_x = 0.0
    way_y = 0.0
