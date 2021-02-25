import csv
import io
from pathlib import Path

from data_classes.phone_data import ReadData, Site, Dataset, BuildingPath


def read_all_data_files(args_data):
    test_data_path = args_data.root_data_path + "/test"
    train_data_path = args_data.root_data_path + "/train"
    # returns a list of Path objects
    test_path_filenames = get_files_in_folder(args_data.site_ids, test_data_path)
    train_path_filenames = get_files_in_folder(args_data.site_ids, train_data_path)

    # array of sites
    test_sites = get_sites(args_data, test_path_filenames)
    train_sites = get_sites(args_data, train_path_filenames)

    return Dataset(train_sites, test_sites)


def get_files_in_folder(site_ids, data_path):
    path_obj = Path(data_path).resolve()
    if len(site_ids) == 0:
        return list(path_obj.glob("*.csv"))

    file_paths = []
    for site_id in site_ids:
        file_paths.extend(list(path_obj.glob(site_id + "*.csv")))

    return file_paths


def get_sites(args_data, path_filenames):
    temp_sites = []
    for file_path in path_filenames:
        file_name = file_path.name.split(".")[0]
        split_file_name = file_name.split("_")
        length = len(split_file_name)

        floor_id = None
        path_id = None
        site_id = None

        if length >= 1:
            site_id = split_file_name[0]
        if length == 2:
            path_id = split_file_name[1]
        if length == 3:
            floor_id = floor_converter(split_file_name[1])
            path_id = split_file_name[2]
        if len(args_data.site_ids) > 0 and site_id not in args_data.site_ids:
            continue

        sites = list(filter(lambda x: x.site_id == site_id, temp_sites))
        if len(sites) == 0:
            site = Site(site_id)
            temp_sites.append(site)
        else:
            site = sites[0]

        path = BuildingPath(path_id)
        path.data = read_data_file(file_path)
        path.floor = floor_id

        site.paths.append(path)
        print(file_path.name)

    return temp_sites


def floor_converter(floor_name):
    floor_id = None
    if "F" in floor_name:
        floor_id = int(floor_name.replace("F", "")) - 1
    if "B" in floor_name:
        floor_id = -int(floor_name.replace("B", ""))

    return floor_id


def read_data_file(data_filename):
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
        data = ReadData()
        data.all_data = dic
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
