import numpy as np
import pandas as pd
from scipy import signal as sig
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix
from scipy.fft import fft, fftfreq

def rms(signal, window=500, stride=100, fs=5120):  # wartści długości okna i przesunięcia w [ms]
    data = signal.values
    # print(data)
    x_diff =0
    for i in range(0, len(data)-window, stride):
        for n in range(i, i+window, 1):
            x_diff += pow(data[n], 2)
            rsm = np.sqrt(x_diff/window)
    return rsm

def filter_emg(data: np.array, fs: int, Rs: int, notch: bool):
    # artefakty ruchowe - usunąć wszystko poniżej 15 Hz
    # filtr pasmowy dla zakłóceń sieciowych 50, 100, 150 ...
    xf = fftfreq(len(data), 1 / fs)
    width = 4
    cut_off = 15
    numtaps, beta = sig.kaiserord(Rs, width / (0.5 * fs))
    filtr = sig.firwin(numtaps + 1, cut_off, window=('kaiser', beta), pass_zero='highpass',
                          scale=False, fs=fs)
    w, h = sig.freqz(filtr, worN=2048, fs=fs)
    h_db = 20 * np.log10(np.abs(h))
    phase = np.degrees(np.angle(h))

    # filtracja sygnału
    signal_filtered = sig.lfilter(filtr, 1, data)  # data
    signal_filtered_zero_ph = sig.filtfilt(filtr, 1, data, padlen=0)  # data

    if notch == True:
        df = 6
        Rp = 20  # 20 max dla zadania
        fz = 244
        freqs = [50, 100, 150, 200]
        for fr in freqs:
            fd = fr - df
            fg = fr + df
            N = 501
            h = sig.firwin(N, (fd, fg), pass_zero='bandstop', fs=fs)
            signal_filtered = sig.lfilter(h, 1, signal_filtered)
            signal_filtered_zero_ph = sig.filtfilt(h, 1, signal_filtered_zero_ph, padlen=0)
        h = sig.firwin(N, fz, pass_zero='lowpass', fs=fs)
        signal_filtered = sig.lfilter(h, 1, signal_filtered)
        signal_filtered_zero_ph = sig.filtfilt(h, 1, signal_filtered_zero_ph, padlen=0)
    return signal_filtered, signal_filtered_zero_ph


def subsample_emg(data: np.array, fs: int, r: int, Rs: int):
    return data


def filter_force(data: np.array, fs: int):
    signal_filtered = data
    signal_filtered_zero_ph = data
    return signal_filtered, signal_filtered_zero_ph


def zc(signal, threshold: float = 0.1, window: float = 500, stride: float = 100, fs=5120):  # wartści długości okna i przesunięci[column columns_emg].iloc[]
    data = signal.values
    zc = 0
    data_filtred = data[(data > threshold) | (data < - threshold)]
    for i in range(0, len(data_filtred) - window, stride):
        for n in range(i + 1, i + window, 1):
            sign = data_filtred[n] * data_filtred[n - 1]
            if (sign < 0):
                zc += 1
    return zc


def find_threshold(signal, columns_emg=['EMG_8', 'EMG_9'], column_gesture='TRAJ_GT', idle_gesture_id=0):
    return signal.loc[signal[column_gesture] == idle_gesture_id, columns_emg].mean()


def norm_emg(signal, norm_coeffs):
    norm_signal = signal / norm_coeffs
    return norm_signal


def main():
    signal_mvc = pd.read_hdf('mvc.hdf5')

    data = signal_mvc['EMG_8']
    data.plot()
    plt.title('Sygnał testowy')
    plt.show()


    # norm_coeffs = rms(signal_mvc, window=500, stride=100, fs=5120)
    # norm_emgs = norm_emg(signal_mvc, norm_coeffs)
    # 1926Hz

if __name__ == '__main__':
    main()