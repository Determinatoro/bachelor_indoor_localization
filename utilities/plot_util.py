import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

from data_classes.phone_data import ReadData


def init_parameters_filter(sample_freq, warmup_data, cut_off_freq=2):
    order = 4
    filter_b, filter_a = signal.butter(order, cut_off_freq / (sample_freq / 2), 'low', False)
    zf = signal.lfilter_zi(filter_b, filter_a)
    _, zf = signal.lfilter(filter_b, filter_a, warmup_data, zi=zf)
    _, filter_zf = signal.lfilter(filter_b, filter_a, warmup_data, zi=zf)

    return filter_b, filter_a, filter_zf


def plot_data(file_data):
    x_values = []
    y_values = []
    x = 0
    for data in file_data:
        if data.acce_x is None:
            continue
        x_values.append(x)
        x += 1
        acce_data = [data.acce_x, data.acce_y, data.acce_z]
        acce_mag = np.sqrt(np.sum(d ** 2 for d in acce_data))
        y_values.append(acce_mag)

    x_values_slice = x_values[:2000]
    y_values_slice = y_values[:2000]

    plt.plot(x_values_slice, y_values_slice, color='red')
    plt.title('Acceleration Magnitude', fontsize=14)
    plt.xlabel('Index', fontsize=14)
    plt.ylabel('Magnitude', fontsize=14)
    plt.grid(True)
    plt.show()

    sample_freq = 50
    window_size = 22

    warmup_data = np.ones((window_size,)) * 9.81
    filter_b, filter_a, filter_zf = init_parameters_filter(sample_freq, warmup_data)

    x_filter_values = x_values_slice
    y_filtered_values = []

    for y in range(len(y_values_slice)):
        acce_mag = y_values_slice[y]
        acce_mag_filt, filter_zf = signal.lfilter(filter_b, filter_a, [acce_mag], zi=filter_zf)
        acce_mag_filt = acce_mag_filt[0]
        y_filtered_values.append(acce_mag_filt)

    plt.plot(x_filter_values, y_filtered_values, color='red')
    plt.title('Acceleration Magnitude Filtered', fontsize=14)
    plt.xlabel('Index', fontsize=14)
    plt.ylabel('Magnitude', fontsize=14)
    plt.grid(True)
    plt.show()
