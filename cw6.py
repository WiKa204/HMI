import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def rms(data: np.array, window=500, stride=100, fs=5120,
        columns_emg=['EMG_8', 'EMG_9']):  # wartści długości okna i przesunięcia w [ms]
    for column in columns_emg:
        signal_data = data[column].values
        for i in range(0, len(signal_data)-window, stride):
             x_i = np.sum(np.abs(signal_data[i:(i+window)]) ** 2)
             signal_data[i:(i + window)] = np.sqrt(x_i/window)
        data[column] = signal_data
    return data[columns_emg].iloc[int(window / 1000 * fs)::int(stride / 1000 * fs)]


def zc(data, threshold: float = 0.1, window: float = 500, stride: float = 100, fs=5120,
       columns_emg=['EMG_8', 'EMG_9']):  # wartści długości okna i przesunięci[column columns_emg].iloc[]
    zc_count = []
    zc_i = []
    for column in columns_emg:
        signal_zc = data[column].values
        data_filtred = np.zeros(len(signal_zc))
        zc = 0
        thresh = threshold * (np.max(signal_zc))
        data_filtred[(signal_zc > thresh) | (signal_zc < - thresh)] = signal_zc[(signal_zc > thresh) | (signal_zc < - thresh)]

        # fig, axs = plt.subplots(1, 2)
        # axs[0].plot(signal_zc)
        # axs[0].set_title('sygnal')
        # plt.grid()
        # axs[1].plot(data_filtred)
        # axs[1].set_title('ZC dla kanału EMG_8')
        # plt.show()

        for i in range(0, len(data_filtred) - window, stride):
            zc_temp = 0
            for n in range(i, i + window, 1):
                if n % window == 0:
                    prev = data_filtred[n]
                    continue
                if data_filtred[n] != 0:
                    sign = data_filtred[n] * prev
                    prev = data_filtred[n]
                    if (sign < 0):
                        zc += 1
                        zc_temp += 1
                        print('hej')
                else:
                    continue
            data_filtred[i:(i + window)] = zc_temp
        #     zc_i.append(zc_temp)
        # zc_count.append(zc_i)
        data[column] = data_filtred
    return data[columns_emg].iloc[int(window/1000*fs)::int(stride/1000*fs)]


def find_threshold(data, columns_emg=['EMG_8', 'EMG_9'], column_gesture='TRAJ_GT', idle_gesture_id=0):
    # thresholds = {}
    for column in columns_emg:
        signal_thresh = data[column].values
        szum_signal = np.zeros(len(signal_thresh))
        szum_signal[data[column_gesture]==idle_gesture_id] = signal_thresh[data[column_gesture]==idle_gesture_id]
        szum = signal_thresh[data[column_gesture]==idle_gesture_id]
        threshold = 2*np.std(szum)
        # thresholds[column] = threshold
        data[column] = threshold
        #return thresholds
    return data.loc[data[column_gesture] == idle_gesture_id, columns_emg].mean()


def norm_emg(data, norm_coeffs, columns_emg=['EMG_8', 'EMG_9']):
    coeffs = norm_coeffs[columns_emg]
    for column in columns_emg:
        normal_signal = data[column].values
        normal_signal = normal_signal / norm_coeffs[column]
        data[column] = normal_signal
    return data[columns_emg]


def k_mean(features, y_true):
    y_pred = y_true
    return confusion_matrix(y_true, y_pred)

# TODO 1: Wczytaj sygnał MVC, i sygnał treningowy
train = pd.read_hdf('Lab10i11_Interface/train.hdf5')
signal_mvc = pd.read_hdf('Lab10i11_Interface/mvc.hdf5')

# signal_mvc[['EMG_8', 'EMG_9']].plot()
# plt.title('sygnał MVC')
# plt.show()
#
# train[['EMG_8', 'EMG_9']].plot()
# plt.title('sygnał treningowy')
# plt.show()


# TODO 2: Napisz funkcję rms, zc, które dla każdego kanału(kolumna
#  columns_emg) wyznaczy wartości 3 opisanych powyżej cech
# feature_rms = rms(train.copy(), window=500, stride=100, fs=5120,
#                  columns_emg=['EMG_8', 'EMG_9'])  # wartści długości okna i przesunięcia w [ms]

# plt.plot(train['EMG_8'], label='sygnal')
# plt.plot(feature_rms['EMG_8'], label='rms')
# plt.title('RMS dla kanału EMG_8')
# plt.legend()
# plt.show()
#
# plt.plot(train['EMG_9'], label='sygnal')
# plt.plot(feature_rms['EMG_9'], label='rms')
# plt.title('RMS dla kanału EMG_9')
# plt.legend()
# plt.show()

# feature_zc = zc(train.copy(), threshold=0.1, window=500, stride=100, fs=5120,
#                  columns_emg=['EMG_8', 'EMG_9'])  # wartści długości okna i przesunięcia w [ms]

# print(*feature_zc, sep='\n')

# plt.plot(train['EMG_8'], label='sygnal')
# plt.plot(feature_zc['EMG_8'], label='zc')
# plt.title('ZC dla kanału EMG_8')
# plt.legend()
# plt.show()
#
# plt.plot(train['EMG_9'], label='sygnal')
# plt.plot(feature_zc['EMG_9'], label='zc')
# plt.title('ZC dla kanału EMG_9')
# plt.legend()
# plt.show()

# TODO 3: Analizując kolumnę TRAJ_GT zauważ, że wartość gestu 0 odpowiadającą
#  brakowi ruchu(ręka jest w stanie neutralnym).Dla każdego kanału wyznacz
#  wartość progu dla funkcji `zc`, zakładając, że próg stanowi wartość,
#  dla której mieści się 95 % wszystkich próbek szumu
threshold = find_threshold(train.copy(), columns_emg=['EMG_8', 'EMG_9'], column_gesture='TRAJ_GT', idle_gesture_id = 0)

print(threshold)

# TODO 4: Napisz funkcję norm_emg normalizującą sygnał emg
# norm_coeffs = rms(signal_mvc.copy(), window=500, stride=100, fs=5120, columns_emg=['EMG_8', 'EMG_9']).max()
# norm_emg = norm_emg(feature_rms.copy(), norm_coeffs, columns_emg=['EMG_8', 'EMG_9'])

# print(norm_coeffs)

# fig, axs = plt.subplots(2, 2)
# axs[0, 0].plot(train['EMG_8'])
# axs[0, 0].set_title("sygnal EMG_8")
# plt.grid()
# axs[1, 0].plot(norm_emg['EMG_8'])
# axs[1, 0].set_title("norm EMG_8")
# plt.grid()
# axs[0, 1].plot(train['EMG_9'])
# axs[0, 1].set_title("sygnal EMG_9")
# plt.grid()
# axs[1, 1].plot(norm_emg['EMG_9'])
# axs[1, 1].set_title("norm EMG_9")
# plt.grid()
# fig.tight_layout()
# plt.show()