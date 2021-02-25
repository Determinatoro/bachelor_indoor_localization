import sys
import time

from utilities.args_util import parse_args
from utilities.csv_util import read_data_file, read_all_data_files
from utilities.plot_util import plot_data


def main(argv):
    args_data = parse_args(argv)
    if args_data is None:
        print("No arguments supplied. Please supply at least 1.")
        return

    # Read data from file
    file_data = read_all_data_files(args_data)
    # plot_data(file_data)
    print("Reading data finished!")


if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv[1:])
    print("--- %s seconds ---" % (time.time() - start_time))
