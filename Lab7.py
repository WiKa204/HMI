import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def filter_emg(data: np.array, fs: int, Rs: int, notch: bool):
    print(f'długość sygnału: {len(data)}')
    width = int(len(data))
    cut_off = (fs/2) - 1
    numtaps, beta = signal.kaiserord(Rs, width / (0.5 * fs))
    print(f'numpas {numtaps}, beta: {beta}')
    filtr = signal.firwin(numtaps+1, cut_off, window=('kaiser', beta), pass_zero='highpass',
                  scale=False, nyq=0.5 * fs)
    plt.plot(filtr)
    plt.title('Odpowiedź impulsowa filtru - okno Kaisera')  # NA KRAŃCACH WIDAĆ NIECIĄGŁOŚCI
    plt.show()
    w, h = signal.freqz(filtr)
    w *= 0.5 * fs / np.pi  # Convert w to Hz
    signal_filtered = signal.lfilter(filtr, 1, data)
    # signal_filtered = signal.lfilter(w, h, data)  # data
    signal_filtered_zero_ph = signal.filtfilt(w, h, data, padlen=0)  # data
    # plt.plot(signal_filtered, label='po')
    # plt.plot(data, label='przed')
    # plt.title('Sygnał przed i po filtracji')
    # plt.legend()
    # plt.show()
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(data)
    axs[0].set_title('Sygnał przed filtracją')
    plt.grid()
    axs[1].plot(signal_filtered)
    axs[1].set_title('Sygnał po filtracji')
    plt.show()
    return signal_filtered, signal_filtered_zero_ph


def subsample_emg(data: np.array, fs: int, r: int, Rs: int):
    return data


def filter_force(data: np.array, fs: int):
    # filtr medianowy
    size = 11
    data = signal.medfilt(data, kernel_size=size)
    signal_filtered = data
    signal_filtered_zero_ph = data
    xf = fftfreq(len(data), 1 / fs)
    # plt.plot(xf, data)
    # plt.show()
    return signal_filtered, signal_filtered_zero_ph


# TODO 1:Wczytaj sygnał zawierający EMG brzuchatego łydki i napisz funkcję
# Funkcja filtrująca powinna:
# - usunąć artefakty ruchowe bez zmiany kształtu sygnału, gdzie tłumienie powinno wynosić nie mniej niż Rs [dB]
# - usunąć zakłócenie sieciowe i harmoniczne, gdzie oczekiwane stłumienie składowej (i przecieków) zakłócenia harmonicznego
# nie powinno być mniejsze niż -20dB (na tej podstawie proszę dobrać szerokość pasma) i powinno być włączane argumentem notch=True

train: np.array = pd.read_csv('emg.csv') # 15 - 5000 Hz
fs_train = 500

train.info()
data = train['emg']
plt.plot(data)
plt.show()
xf = fftfreq(len(data), 1/fs_train)
plt.plot(xf, np.abs(fft(data)))
plt.show()

signal_filtered, signal_filtered_zero_ph  = filter_emg(data, fs=fs_train, Rs=50, notch=True)


signal_2 = signal_filtered
signal_subsampled = subsample_emg(signal_2, fs=fs_train, r=3, Rs=30)



data_3 : np.array = pd.read_csv('force.csv')
fs_3 = 5000

# print(data_3.to_string())
# print(data_3.head())
# data_3['force'].plot()
# plt.show()
signal_3 = data_3['force']
signal_filtered, signal_zero_ph = filter_force(signal_3, fs=fs_3)

#########################################
