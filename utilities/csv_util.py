import csv
import io

from data_classes.phone_data import ReadData


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
        for key in dic.keys():
            attr = getattr(data, key)
            t = type(attr)
            value = dic[key]
            if t == int:
                setattr(data, key, int(value) if value != "" else None)
            elif t == float:
                setattr(data, key, float(value) if value != "" else None)
            elif t == str:
                setattr(data, key, dic[key])
        all_data.append(data)
    return all_data
