# Code used for Exploratory Data Analysis
# Analyzes train data for the 24 sites also available in test
# Result: 334038 wifi entries per floor on average

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys #runtime environment
import glob #pathname patterns
from io_f import read_data_file # read static file

#input data from kaggle. edit if used elsewhere
sys.path.append('../input/indoor-locationnavigation-2021/indoor-location-competition-20-master/indoor-location-competition-20-master')


# list of sites that occur in test dataset
site_list = ["5a0546857ecc773753327266", "5c3c44b80379370013e0fd2b", "5d27075f03f801723c2e360f", "5d27096c03f801723c31e5e0", 
             "5d27097f03f801723c320d97", "5d27099f03f801723c32511d", "5d2709a003f801723c3251bf", "5d2709b303f801723c327472",
            "5d2709bb03f801723c32852c", "5d2709c303f801723c3299ee", "5d2709d403f801723c32bd39", "5d2709e003f801723c32d896",
            "5da138274db8ce0c98bbd3d2", "5da1382d4db8ce0c98bbe92e", "5da138314db8ce0c98bbf3a0", "5da138364db8ce0c98bc00f1",
            "5da1383b4db8ce0c98bc11ab", "5da138754db8ce0c98bca82f", "5da138764db8ce0c98bcaa46", "5da1389e4db8ce0c98bd0547",
            "5da138b74db8ce0c98bd4774", "5da958dd46f8266d0737457b", "5dbc1d84c1eb61796cf7c010", "5dc8cea7659e181adb076a3f"]

wifi_entries = []

#get list of paths to all floors in each site
for site in site_list:
    floor_paths = list(glob.glob(f"../input/indoor-location-navigation/train/{site}/*"))
	# get list of paths to all paths on each floor
    for floor_path in floor_paths:
        path_paths = list(glob.glob(f"{floor_path}/*"))
        count = 0
		# access amount of wifi entries for each path and add to list
        for path_path in path_paths:
            file = read_data_file(path_path)
            count += file.wifi.shape[0] # get amount of wifi entries for floor
        wifi_entries.append(count) # list of wifi entries per floor
        
print(f"mean of WiFi pr. floor {sum(wifi_entries)/len(wifi_entries)}")

