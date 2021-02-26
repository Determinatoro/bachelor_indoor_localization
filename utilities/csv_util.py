import csv
import io
from pathlib import Path
from data_classes.phone_data import ReadData, Site, Dataset, BuildingPath


def read_all_data_files(args_data, max_path_files=None):
    """
    goes through all data in root folder EXCEPT when site id's are supplied
    :param args_data: Of type ArgsData, from function parse_args
    :return: Object of type DataSet consisting of
        a) list of train data
        b) list of test data
    """
    test_data_path = args_data.root_data_path + "/test"
    train_data_path = args_data.root_data_path + "/train"

    # returns a list of Path objects
    test_path_filenames = get_files_in_folder(args_data.site_ids, test_data_path)
    train_path_filenames = get_files_in_folder(args_data.site_ids, train_data_path)

    # if max_path_files is set only that amount of files are used further
    if max_path_files is not None:
        test_path_filenames = test_path_filenames[:max_path_files]
        train_path_filenames = train_path_filenames[:max_path_files]

    # lists of sites
    test_sites = get_sites(args_data, test_path_filenames)
    train_sites = get_sites(args_data, train_path_filenames)

    return Dataset(train_sites, test_sites)


def get_files_in_folder(site_ids, data_path):
    """
    returns a list of csv file objects in folder data_path
    :param site_ids: a list of site id's whose files to get
    :param data_path: root data path
    :return: list of file objects where each correspond to a path on a given site
    """
    # if no specific sites are supplied return a list of all files
    path_obj = Path(data_path).resolve()
    if len(site_ids) == 0:
        return list(path_obj.glob("*.csv"))

    file_paths = []
    for site_id in site_ids:
        file_paths.extend(path_obj.glob(site_id + "*.csv"))

    return file_paths


def get_sites(args_data, path_filenames):
    """

    :param args_data: data type ArgsData
    :param path_filenames: list of path objects
    :return: list of site objects
    """
    temp_sites = []
    for file_path in path_filenames:
        # filter out extension
        file_name = file_path.name.split(".")[0]
        # returns list of strings separated by "_"
        split_file_name = file_name.split("_")
        length = len(split_file_name)

        # default values
        floor_id = None
        path_id = None
        site_id = None

        # identifies different parts of the name
        # site_floor_path OR site_path --> site
        if length >= 1:
            site_id = split_file_name[0]
        # site_path --> path
        if length == 2:
            path_id = split_file_name[1]
        # site_floor_path --> floor AND path
        if length == 3:
            floor_id = floor_converter(split_file_name[1])
            path_id = split_file_name[2]
        # stops if args_data has no site_id's OR the specific id is not in args_data
        if len(args_data.site_ids) > 0 and site_id not in args_data.site_ids:
            continue

        # inserts site id in list sites IF site_id IS NOT already in temp_sites
        sites = list(filter(lambda x: x.site_id == site_id, temp_sites))
        # if not in temp_sites make a Site object and add to temp_sites
        if len(sites) == 0:
            site = Site(site_id)
            temp_sites.append(site)
        # if already in temp_sites
        else:
            site = sites[0]

        # add information to path object
        path = BuildingPath(path_id)
        path.data = read_data_file(file_path)
        path.floor = floor_id

        # adds paths to site object
        site.paths.append(path)
        print(file_path.name)

    return temp_sites


def floor_converter(floor_name):
    """
    edits floor_name to match convention
    B1 --> -1 AND F1 --> 0
    :param floor_name: string floor name from the middle of file name
    :return: integer representing floor number
    """
    floor_id = None
    # removes F and decriments number
    if "F" in floor_name:
        floor_id = int(floor_name.replace("F", "")) - 1
    # removes b and makes number negative
    if "B" in floor_name:
        floor_id = -int(floor_name.replace("B", ""))

    return floor_id


def read_data_file(data_filename):
    """
    extracts data from csv file
    :param data_filename: Path object
    :return: an array of ReadData objects
    """
    # Read lines from file
    with open(data_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # Remove sep=, from the file
    lines = list(filter(lambda x: not x.startswith("sep"), lines))
    # Join the list of lines into a single string
    s = "\r\n".join(lines)
    # Parse the string as CSV
    dict_reader = csv.DictReader(io.StringIO(s))
    # Array containing all the data from the file
    all_data = []
    # Go through each CSV dictionary
    for dic in list(dict_reader):
        dic["timestamp"] = dic.pop('')
        type_casting(dic, "timestamp", int)
        type_casting(dic, "acce_x", float)
        type_casting(dic, "acce_y", float)
        type_casting(dic, "acce_z", float)
        type_casting(dic, "acce_accuracy", float)
        type_casting(dic, "acce_uncali_x", float)
        type_casting(dic, "acce_uncali_y", float)
        type_casting(dic, "acce_uncali_z", float)
        type_casting(dic, "acce_bias_x", float)
        type_casting(dic, "acce_bias_y", float)
        type_casting(dic, "acce_bias_z", float)
        type_casting(dic, "acce_uncali_accuracy", float)
        type_casting(dic, "magn_x", float)
        type_casting(dic, "magn_y", float)
        type_casting(dic, "magn_z", float)
        type_casting(dic, "magn_accuracy", float)
        type_casting(dic, "magn_uncali_x", float)
        type_casting(dic, "magn_uncali_y", float)
        type_casting(dic, "magn_uncali_z", float)
        type_casting(dic, "magn_bias_x", float)
        type_casting(dic, "magn_bias_y", float)
        type_casting(dic, "magn_bias_z", float)
        type_casting(dic, "magn_uncali_accuracy", float)
        type_casting(dic, "gyro_x", float)
        type_casting(dic, "gyro_y", float)
        type_casting(dic, "gyro_z", float)
        type_casting(dic, "gyro_accuracy", float)
        type_casting(dic, "gyro_uncali_x", float)
        type_casting(dic, "gyro_uncali_y", float)
        type_casting(dic, "gyro_uncali_z", float)
        type_casting(dic, "gyro_bias_x", float)
        type_casting(dic, "gyro_bias_y", float)
        type_casting(dic, "gyro_bias_z", float)
        type_casting(dic, "gyro_uncali_accuracy", float)
        type_casting(dic, "ahrs_x", float)
        type_casting(dic, "ahrs_y", float)
        type_casting(dic, "ahrs_z", float)
        type_casting(dic, "ahrs_accuracy", float)
        type_casting(dic, "wifi_ssid", str)
        type_casting(dic, "wifi_bssid", str)
        type_casting(dic, "wifi_rssi", int)
        type_casting(dic, "wifi_freq", int)
        type_casting(dic, "wifi_ls_ts", float)
        type_casting(dic, "uuid", str)
        type_casting(dic, "majorid", str)
        type_casting(dic, "minorid", str)
        type_casting(dic, "txpow", str)
        type_casting(dic, "beacon_rssi", int)
        type_casting(dic, "distance", float)
        type_casting(dic, "macaddr", str)
        type_casting(dic, "unix_time", float)
        type_casting(dic, "way_x", float)
        type_casting(dic, "way_y", float)

        data = ReadData()
        data.values = dic

        """
        for key in dic.keys():
            attr = getattr(data, key)
            t = type(attr)
            value = dic[key]
            if value == "nan":
                setattr(data, key, None)
                continue
            if t == int:
                setattr(data, key, int(value) if value != "" else None)
            elif t == float:
                setattr(data, key, float(value) if value != "" else None)
            elif t == str:
                setattr(data, key, dic[key])"""
        all_data.append(data)
    return all_data


def type_casting(dic, key, type):
    """
    type casts the value of dic[key] to type
    :param dic:
    :param key:
    :param type:
    :return: nothing. changes recorded in dic
    """
    if key not in dic:
        dic[key] = None
        return

    # get value of dic[key]
    value = dic.pop(key)

    if value == "" or value == "nan":
        dic[key] = None
        return

    # sets type of value according to type parameter
    if type == int:
        dic[key] = int(value)
    elif type == float:
        dic[key] = float(value)
    elif type == str:
        dic[key] = str(value)
    return
