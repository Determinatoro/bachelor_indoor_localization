import sys

from utilities.csv_util import read_data_file
from utilities.plot_util import plot_data


def main(argv):
    # Check for desired number of parameters
    if len(argv) == 0:
        return
    # Read data from file
    file_data = read_data_file(argv[0])
    plot_data(file_data)


if __name__ == "__main__":
    main(sys.argv[1:])
