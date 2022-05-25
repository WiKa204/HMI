import numpy as np
import pandas as pd
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix


def rms(signal: np.array, window=500, stride=100, fs=5120,
        columns_emg=['EMG_8', 'EMG_9']):  # wartści długości okna i przesunięcia w [ms]
    for column in columns_emg:
        # print(column)
        data = signal[column]
        # print(data)
        x_0 = np.mean(data)
        x_diff =0
        print(x_0)
        for i in range(0, len(data)-window, stride):
            # print(i)
            for n in range(i, i+window, 1):
                print()
                # x_diff += data[n:] - x_0
                # print(x_diff)
            # xc = np.cumsum(abs(signal) ** 2)
            # rms_i = np.sqrt((xc[window:] - xc[:-window]) / window)
    return signal[columns_emg].iloc[int(window / 1000 * fs)::int(stride / 1000 * fs)]


def zc(signal, threshold: float = 0.1, window: float = 500, stride: float = 100, fs=5120,
       columns_emg=['EMG_8', 'EMG_9']):  # wartści długości okna i przesunięci[column columns_emg].iloc[]
        # tsed = signal[(signal > threshold) | (signal < -threshold)]
        # s3 = np.sign(tsed)
        # s3[s3 == 0] = -1  # replace zeros with -1
        # zcs = len(np.where(np.diff(s3))[0])
    return signal[columns_emg].iloc[int(window / 1000 * fs)::int(stride / 1000 * fs)]


def find_threshold(signal, columns_emg=['EMG_8', 'EMG_9'], column_gesture='TRAJ_GT', idle_gesture_id=0):
    thresholds = {}
    for ch in columns_emg:
        ts_ch = signal[signal[column_gesture] == idle_gesture_id][ch].abs().mean()
        thresholds[ch] = ts_ch
    thresholds_df = pd.DataFrame(thresholds)
    return signal.loc[signal[column_gesture] == idle_gesture_id, columns_emg].mean()


def norm_emg(signal, norm_coeffs, columns_emg=['EMG_8', 'EMG_9']):
    coeffs = norm_coeffs[columns_emg]

    norm_sig = signal / norm_coeff
    return signal[columns_emg]


def k_mean(features, y_true):
    y_pred = y_true
    return confusion_matrix(y_true, y_pred)

# TODO 1: Wczytaj sygnał MVC, i sygnał treningowy
train = pd.read_hdf('./data/train.hdf5')
data = pd.read_hdf('Lab10i11_Interface/mvc.hdf5')

# TODO 2: Napisz funkcję rms, zc, które dla każdego kanału (kolumna columns_emg) wyznaczy wartości 3 opisanych powyżej cech
feature_rms = rms(train, window=500, stride=100, fs=5120,
                  columns_emg=['EMG_8', 'EMG_9'])  # wartści długości okna i przesunięcia w [ms]

# feature_zc = zc(signal, threshold=0.1, window=500, stride=100, fs=5120,
  #              columns_emg=['EMG_8', 'EMG_9'])  # wartści długości okna i przesunięcia w [ms]

# TODO 3: Wyznacz wartość progu dla funkcji zc - próg to 95% wszystkich próbek szumu
# threshold = find_threshold(signal, columns_emg=['EMG_8', 'EMG_9'], column_gesture='TRAJ_GT', idle_gesture_id=0)
