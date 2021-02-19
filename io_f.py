import csv
import io
from dataclasses import dataclass
import numpy as np


@dataclass
class ReadData:
    acce_x: np.ndarray
    acce_y: np.ndarray
    acce_z: np.ndarray
    timestamp: np.ndarray


def read_data_file(data_filename):
    acce_x = []
    acce_y = []
    acce_z = []
    timestamp = []

    # Read lines from file
    with open(data_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # Remove sep=, from the file
    lines = list(filter(lambda x: not x.startswith("sep"), lines))
    # Join the list of lines into a single string
    s = "\r\n".join(lines)
    # Parse the string as CSV
    reader = csv.DictReader(io.StringIO(s))

    for row in reader:
        acce_x.append(row['acce_x'])
        acce_y.append(row['acce_y'])
        acce_z.append(row['acce_z'])
        timestamp.append(row[''])

    return ReadData(acce_x, acce_y, acce_z, timestamp)