import numpy as np
import pandas as pd
from scipy import signal as sig
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix


def rms(signal, window=500, stride=100, fs=5120,
        columns_emg=['EMG_8', 'EMG_9']):  # wartści długości okna i przesunięcia w [ms]
    for column in columns_emg:
        # print(column)
        data = signal[column].values
        # print(data)
        # x_0 = np.mean(data)
        x_diff =0
        # print(x_0)
        for i in range(0, len(data)-window, stride):
            # print(i)
            for n in range(i, i+window, 1):
                # print(data[n])
                # x_diff += data[n] - x_0
                x_diff += pow(data[n], 2)
                # print(x_diff)
            rsm_i = np.sqrt(x_diff/window)
    return signal[columns_emg].iloc[int(window / 1000 * fs)::int(stride / 1000 * fs)]


def zc(signal, threshold: float = 0.1, window: float = 500, stride: float = 100, fs=5120,
       columns_emg=['EMG_8', 'EMG_9']):  # wartści długości okna i przesunięci[column columns_emg].iloc[]
        # tsed = signal[(signal > threshold) | (signal < -threshold)]
        # s3 = np.sign(tsed)
        # s3[s3 == 0] = -1  # replace zeros with -1
        # zcs = len(np.where(np.diff(s3))[0])
    for column in columns_emg:
        # print(column)
        data = signal[column].values
        zc = 0
        data_filtred = data[(data > threshold) | (data < - threshold)]
        for i in range(0, len(data) - window, stride):
            for n in range(i+1, i + window, 1):
                sign = data[n]*data[n-1]
                if(sign<0):
                    zc += 1
        print(zc)
    return signal[columns_emg].iloc[int(window / 1000 * fs)::int(stride / 1000 * fs)]


def find_threshold(signal, columns_emg=['EMG_8', 'EMG_9'], column_gesture='TRAJ_GT', idle_gesture_id=0):
    thresholds = {}
    for column in columns_emg:
        ts_column = signal[signal[column_gesture] == idle_gesture_id][column].abs().mean()
        thresholds[column] = ts_column
    thresholds_df = pd.DataFrame(thresholds, index=[0])
    # print(thresholds_df)
    return thresholds_df
        #signal.loc[signal[column_gesture] == idle_gesture_id, columns_emg].mean()


def norm_emg(signal, norm_coeffs, columns_emg=['EMG_8', 'EMG_9']):
    coeffs = norm_coeffs[columns_emg]

    # norm_sig = signal / norm_coeff
    return signal[columns_emg]


def k_mean(features, y_true):
    y_pred = y_true
    return confusion_matrix(y_true, y_pred)

# TODO 1: Wczytaj sygnał MVC, i sygnał treningowy
train = pd.read_hdf('./data/train.hdf5')
data = pd.read_hdf('Lab10i11_Interface/mvc.hdf5')
print(train.head())

# TODO 2: Napisz funkcję rms, zc, które dla każdego kanału (kolumna columns_emg) wyznaczy wartości 3 opisanych powyżej cech
feature_rms = rms(train, window=500, stride=100, fs=5120,
                  columns_emg=['EMG_8', 'EMG_9'])  # wartści długości okna i przesunięcia w [ms]

# feature_zc = zc(train, threshold=0.1, window=500, stride=100, fs=5120,
 #             columns_emg=['EMG_8', 'EMG_9'])  # wartści długości okna i przesunięcia w [ms]

# TODO 3: Wyznacz wartość progu dla funkcji zc - próg to 95% wszystkich próbek szumu
threshold = find_threshold(train, columns_emg=['EMG_8', 'EMG_9'], column_gesture='TRAJ_GT', idle_gesture_id=0)
print(threshold)