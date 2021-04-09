import glob
import numpy as np

from dataclasses import dataclass

BASE_PATH = "."

test_paths = glob.glob(f'{BASE_PATH}/input/indoor-location-navigation/test/*')

@dataclass
class Site:
    site_id = ""
    paths = []
    count = 0

    def __init__(self, site_id):
        self.site_id = site_id


site_list = []

site = Site("SiteID:5a0546857ecc773753327266")
site_list.append(site)
site = Site("SiteID:5c3c44b80379370013e0fd2b")
site_list.append(site)
site = Site("SiteID:5d27075f03f801723c2e360f")
site_list.append(site)
site = Site("SiteID:5d27096c03f801723c31e5e0")
site_list.append(site)
site = Site("SiteID:5d27097f03f801723c320d97")
site_list.append(site)
site = Site("SiteID:5d27099f03f801723c32511d")
site_list.append(site)
site = Site("SiteID:5d2709a003f801723c3251bf")
site_list.append(site)
site = Site("SiteID:5d2709b303f801723c327472")
site_list.append(site)
site = Site("SiteID:5d2709bb03f801723c32852c")
site_list.append(site)
site = Site("SiteID:5d2709c303f801723c3299ee")
site_list.append(site)
site = Site("SiteID:5d2709d403f801723c32bd39")
site_list.append(site)
site = Site("SiteID:5d2709e003f801723c32d896")
site_list.append(site)
site = Site("SiteID:5da138274db8ce0c98bbd3d2")
site_list.append(site)
site = Site("SiteID:5da1382d4db8ce0c98bbe92e")
site_list.append(site)
site = Site("SiteID:5da138314db8ce0c98bbf3a0")
site_list.append(site)
site = Site("SiteID:5da138364db8ce0c98bc00f1")
site_list.append(site)
site = Site("SiteID:5da1383b4db8ce0c98bc11ab")
site_list.append(site)
site = Site("SiteID:5da138754db8ce0c98bca82f")
site_list.append(site)
site = Site("SiteID:5da138764db8ce0c98bcaa46")
site_list.append(site)
site = Site("SiteID:5da1389e4db8ce0c98bd0547")
site_list.append(site)
site = Site("SiteID:5da138b74db8ce0c98bd4774")
site_list.append(site)
site = Site("SiteID:5da958dd46f8266d0737457b")
site_list.append(site)
site = Site("SiteID:5dbc1d84c1eb61796cf7c010")
site_list.append(site)
site = Site("SiteID:5dc8cea7659e181adb076a3f")
site_list.append(site)

@dataclass
class ReadData:
    acce: np.ndarray
    acce_uncali: np.ndarray
    gyro: np.ndarray
    gyro_uncali: np.ndarray
    magn: np.ndarray
    magn_uncali: np.ndarray
    ahrs: np.ndarray
    wifi: np.ndarray
    ibeacon: np.ndarray
    waypoint: np.ndarray


def read_data_file(data_filename):
    acce = []
    acce_uncali = []
    gyro = []
    gyro_uncali = []
    magn = []
    magn_uncali = []
    ahrs = []
    wifi = []
    ibeacon = []
    waypoint = []

    with open(data_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line_data in lines:
        line_data = line_data.strip()
        if not line_data or line_data[0] == '#':
            continue

        line_data = line_data.split('\t')

        if line_data[1] == 'TYPE_ACCELEROMETER':
            acce.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_ACCELEROMETER_UNCALIBRATED':
            acce_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_GYROSCOPE':
            gyro.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_GYROSCOPE_UNCALIBRATED':
            gyro_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_MAGNETIC_FIELD':
            magn.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_MAGNETIC_FIELD_UNCALIBRATED':
            magn_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_ROTATION_VECTOR':
            ahrs.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_WIFI':
            sys_ts = line_data[0]
            ssid = line_data[2]
            bssid = line_data[3]
            rssi = line_data[4]
            lastseen_ts = line_data[6]
            wifi_data = [sys_ts, ssid, bssid, rssi, lastseen_ts]
            wifi.append(wifi_data)
            continue

        if line_data[1] == 'TYPE_BEACON':
            ts = line_data[0]
            uuid = line_data[2]
            major = line_data[3]
            minor = line_data[4]
            rssi = line_data[6]
            ibeacon_data = [ts, '_'.join([uuid, major, minor]), rssi]
            ibeacon.append(ibeacon_data)
            continue

        if line_data[1] == 'TYPE_WAYPOINT':
            waypoint.append([int(line_data[0]), float(line_data[2]), float(line_data[3])])

    acce = np.array(acce)
    acce_uncali = np.array(acce_uncali)
    gyro = np.array(gyro)
    gyro_uncali = np.array(gyro_uncali)
    magn = np.array(magn)
    magn_uncali = np.array(magn_uncali)
    ahrs = np.array(ahrs)
    wifi = np.array(wifi)
    ibeacon = np.array(ibeacon)
    waypoint = np.array(waypoint)

    return ReadData(acce, acce_uncali, gyro, gyro_uncali, magn, magn_uncali, ahrs, wifi, ibeacon, waypoint)

for path in test_paths:
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line_data in lines:
        line_data = line_data.split('\t')
        site_id = line_data[1]
        is_site = site_id.split(':')

        if is_site[0] == "SiteID":
            file = read_data_file(path)
            count = file.wifi.shape[0]
            site = list(filter(lambda x: x.site_id == site_id, site_list))[0]
            site.count += count
            print(site.count)
            break

for site in site_list:
    print(f"{site.site_id} number of wifi entries {site.count}")



