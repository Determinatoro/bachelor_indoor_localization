# Code used for Exploratory Data Analysis
# Analyzes train data for the 24 sites also available in test
# Result: The mean of unique BSSID's per floor is 1160.4

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys # runtime environment
import glob # pathname patterns
from io_f import read_data_file # read static file

# input data from kaggle. edit if used elsewhere
sys.path.append('../input/indoor-locationnavigation-2021/indoor-location-competition-20-master/indoor-location-competition-20-master')


# list of sites that occur in test dataset
site_list = ["5a0546857ecc773753327266", "5c3c44b80379370013e0fd2b", "5d27075f03f801723c2e360f", "5d27096c03f801723c31e5e0", 
             "5d27097f03f801723c320d97", "5d27099f03f801723c32511d", "5d2709a003f801723c3251bf", "5d2709b303f801723c327472",
            "5d2709bb03f801723c32852c", "5d2709c303f801723c3299ee", "5d2709d403f801723c32bd39", "5d2709e003f801723c32d896",
            "5da138274db8ce0c98bbd3d2", "5da1382d4db8ce0c98bbe92e", "5da138314db8ce0c98bbf3a0", "5da138364db8ce0c98bc00f1",
            "5da1383b4db8ce0c98bc11ab", "5da138754db8ce0c98bca82f", "5da138764db8ce0c98bcaa46", "5da1389e4db8ce0c98bd0547",
            "5da138b74db8ce0c98bd4774", "5da958dd46f8266d0737457b", "5dbc1d84c1eb61796cf7c010", "5dc8cea7659e181adb076a3f"]

wifi_BSSIDs = [] # nparray of amount of unique BSSID's per floor

#file system order is: site/floor/path

site_count = 0
# get list of paths to all floors in each site
for site in site_list:
    floor_paths = list(glob.glob(f"../input/indoor-location-navigation/train/{site}/*"))
    floor_count = 0
    site_count += 1
	# get list of paths to all paths on each floor
    for floor_path in floor_paths:
        floor_count += 1
        path_paths = list(glob.glob(f"{floor_path}/*"))
        temp_BSSIDs = [] # temporary list of BSSID's. emptied for every floor.
        path_count = 0
		# get list of BSSID's for each path and add as individual entries to temp_BSSID's
        for path_path in path_paths:
            file = read_data_file(path_path)
            path_count += 1
            try:
                BSSIDs = list(file.wifi[:, 2])
            except:
                print(file.wifi) #print error for paths with no wifi access points
            temp_BSSIDs.extend(BSSIDs) # extends the list with values for each path
        
        unique_BSSIDs = np.unique(np.array(temp_BSSIDs)) # remove duplicates
        
        print(f"Site: {site_count}, floor: {floor_count}, total: {len(temp_BSSIDs)}, unique: {len(unique_BSSIDs)}")

        wifi_BSSIDs.append(len(unique_BSSIDs)) # add number of unique BSSID's for floor
              
print(f"mean of unique BSSIDs pr. floor {sum(wifi_BSSIDs)/len(wifi_BSSIDs)}")

