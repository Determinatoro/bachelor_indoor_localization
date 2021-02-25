import sys
import time

from utilities.args_util import parse_args
from utilities.csv_util import read_data_file, read_all_data_files
from utilities.plot_util import plot_data
from compute.positions import compute_step_positions

def main(argv):
    args_data = parse_args(argv)
    if args_data is None:
        print("No arguments supplied. Please supply at least 1.")
        return

    # Read data from file
    data_set = read_all_data_files(args_data)
    # plot_data(file_data)
    print("Reading data finished!")

    for site in data_set.train_data:
        for path in site.paths:
            path.step_position = compute_step_positions(path)

    print("Done computing")


if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv[1:])
    print("--- %s seconds ---" % (time.time() - start_time))
