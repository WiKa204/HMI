# Dla wszystkich filtrów przyjmij, że w paśmie przepustowym
# oscylacje sygnału po pełnej filtracji (wykonanej przez daną funkcję)
# nie powinny zmienić się o więcej niż o 1%.
# Wszystkie dane wejściowe zawierająca tablice danych powinny być typu nd.array.
# Dobierz parametry filtrów tak by spełnić minimalne wymagania wzmocnienia/tłumienia
# i zniekstałcenia sygnałów oraz minimalizować opóźnienie filtra


import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fftfreq
import matplotlib.pyplot as plt


def filter_emg(data: np.array, fs: int, Rs: int, notch: bool):
    print(f'długość sygnału: {len(data)}')
    width = int(len(data))
    cut_off = (fs/2) - 1
    numtaps, beta = signal.kaiserord(Rs, width / (0.5 * fs))
    print(f'numpas {numtaps}, beta: {beta}')
    taps = signal.firwin(numtaps+1, cut_off, window=('kaiser', beta), pass_zero='highpass',
                  scale=False, nyq=0.5 * fs)
    w, h = signal.freqz(taps)
    w *= 0.5 * fs / np.pi  # Convert w to Hz
    signal_filtered = signal.lfilter(w, h, data) #data
    signal_filtered_zero_ph = signal.filtfilt(w, h, data, padlen=0) #data

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

train: np.array = pd.read_csv('emg.csv')
fs = 500

# train['emg'].plot()
# plt.show()

signal_1 = train['emg']
signal_filtered, signal_filtered_zero_ph  = filter_emg(signal_1, fs=fs, Rs=50, notch=True)


# 2. Dla przefiltrowanego sygnału napisz funkcję, która dokona subsamplingu sygnału
# o r razy (r jest typu int). Pamiętaj, żeby w przefiltrowanym sygnale nie było aliasów
# signal_subsampled = subsample_emg(signal_filtered, fs=500, r=3, Rs=30)
# Filtracja antyaliasingowa nie powinna istotnie zmieniać kształtu sygnału, oraz zapewniać, że ew aliasy będą stłumione o nie mniej niż o Rs (wyrażone w dB)

signal_2 = signal_filtered
signal_subsampled = subsample_emg(signal_2, fs=fs, r=3, Rs=30)

# 3. Wczytaj sygnał, który zawiera sygnał siły skurczu mięśnia wywołanej
# stymulacją elektryczną. Impulsy stymulacji elektrycznej 50-200μs i amplitudzie
# do 200V przenoszą się do ukłądu pomiarowego siły, i tworzą w zapisie charakterystyczne
# piki. Ponadto w przebiegu widoczny jest szum kwantyzacji przetwornika ADC.
# Zastanów się jak stosując filtrację można zmniejszyć amplitudę pików oraz
# wyeliminować szum kwantyzacji, nie modyfikując kształtu zarejestrowanego sygnału.
# Częstotliwość próbkowania sygnału fs=5kHz Należy wiedzieć że widmo wąskiego impulsu
# jest zbliżone do widma delty Diraca.
# signal_filtered, signal_zero_ph = filter_force(signal, fs)

data_3 : np.array = pd.read_csv('force.csv')
fs_3 = 5000

# print(data_3.to_string())
# print(data_3.head())
# data_3['force'].plot()
# plt.show()
signal_3 = data_3['force']
signal_filtered, signal_zero_ph = filter_force(signal_3, fs=fs_3)
