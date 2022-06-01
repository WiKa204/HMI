import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix
from scipy.fft import fft, fftfreq

def rms(data):
    data_rms = data.copy()
    x_diff =0
    for i in range(0, len(data_rms), 1):
        print(i)
        # print(data_rms)
        x_diff += pow(data_rms[i], 2)
        data_rms[i] = np.sqrt(x_diff/(i+1))
    # rms = np.sqrt(x_diff/(len(data_rms)+1))
    return max(data_rms)

def filter_emg(data, fs: int, Rs: int, notch: bool):
    # f. górnoprzepustowy - 15 Hz - artefakt ruchowy
    xf = fftfreq(len(data), 1 / fs)
    width = 4
    cut_off = 15
    numtaps, beta = signal.kaiserord(Rs, width / (0.5 * fs))
    filtr = signal.firwin(numtaps + 1, cut_off, window=('kaiser', beta), pass_zero='highpass',
                          scale=False, fs=fs)
    # filtracja sygnału
    signal_filtered = signal.lfilter(filtr, 1, data)
    # notch filter - 50 Hz - f. eliptyczny - wyświetlić fouriera
    if notch==True:
        fp = np.array([48, 52]) # fpass [Hz]
        fz = np.array([49, 51])  # fstop [Hz]
        Rp = 3  # -3dB w paśmie przepustowym
        Rs = 50  # -100db w paśmie zaporowym
        fs = 500  # częstotliwość próbkowania
        N, fn = signal.ellipord(fp, fz, Rp, Rs, fs=fs)
        b, a = signal.ellip(N, Rp, Rs, fn, 'bandstop', fs=fs)
        signal_filtered = signal.filtfilt(b, a, signal_filtered, method="gust", padlen=0)
        xf = fftfreq(len(data), 1 / fs)
    return signal_filtered


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


def norm_emg(data, norm_coeffs):
    data_rms = data.copy()
    x_diff = 0
    for i in range(0, len(data_rms), 1):
        # print('normalizacja')
        x_diff += pow(data_rms[i], 2)
        data_rms[i] = np.sqrt(x_diff / (i + 1))
    rms = np.sqrt(x_diff/(len(data_rms)+1))
    print(len(data_rms))
    print(f'rms: {rms}')
    # norm_signal = data / norm_coeffs
    norm_signal = rms / norm_coeffs
    print(norm_signal)
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