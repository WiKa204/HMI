import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def filter_emg(data, fs: int, Rs: int, notch: bool):
    # f. górnoprzepustowy - 15 Hz - artefakt ruchowy
    xf = fftfreq(len(data), 1 / fs)
    print(f'długość sygnału surowego: {len(data)}')
    width = 4
    cut_off = 15
    numtaps, beta = signal.kaiserord(Rs, width / (0.5 * fs))
    print(f'numpas {numtaps}, beta: {beta}')
    filtr = signal.firwin(numtaps + 1, cut_off, window=('kaiser', beta), pass_zero='highpass',
                          scale=False, fs=fs)
    # plt.plot(filtr)
    # plt.title('Odpowiedź impulsowa filtru - okno Kaisera')
    # plt.show()
    # filtracja sygnału
    signal_filtered = signal.lfilter(filtr, 1, data)
    # plt.plot(xf, signal_filtered)
    # plt.title('Sygnał bez artefaktów ruchowych')
    # plt.show()
    print(f'długość sygnału przefiltrowanego: {len(signal_filtered)}')
    # notch filter - 50 Hz - f. eliptyczny - wyświetlić fouriera
    if notch==True:
        fp = np.array([48, 52]) # fpass [Hz]
        fz = np.array([49, 51])  # fstop [Hz]
        Rp = 3  # -3dB w paśmie przepustowym
        Rs = 50  # -100db w paśmie zaporowym
        N, fn = signal.ellipord(fp, fz, Rp, Rs, fs=fs)
        b, a = signal.ellip(N, Rp, Rs, fn, 'bandstop', fs=fs)
        signal_filtered = signal.filtfilt(b, a, signal_filtered, method="gust", padlen=0)
        # plt.plot(b)
        # plt.title('Odpowiedź impulsowa filtru eliptycznego')
        # plt.show()
    return signal_filtered

def rms(data, fs=5120):
    data_rms = data.copy()
    # print(data)
    x_diff =0
    for i in range(0, len(data_rms), 1):
        x_diff += pow(data_rms[i], 2)
        data_rms[i] = np.sqrt(x_diff/(i+1))
    rms = np.sqrt(x_diff/(len(data_rms)+1))
    plt.plot(data)
    plt.plot(data_rms)
    plt.title('RMS')
    plt.show()
    print(f'rms: {rms}')
    print(f'max: {max(data_rms)}')
    return max(data_rms)

def norm_emg(data, norm_coeffs):
    norm_signal = data / norm_coeffs
    return norm_signal

train = pd.read_hdf('train.hdf5')
mvc = pd.read_hdf('mvc.hdf5')

# print(train.columns)
# train.filter(regex='EMG_').plot()
# train.filter(regex='EMG_[1]').plot()
# train.filter(regex='EMG_9').plot()
# train[['EMG_9', 'TRAJ_GT']].plot()
# plt.show()

# print(mvc.columns)
# mvc['EMG_8'].plot()
# plt.show()

fs = 1920

signal_filtered = filter_emg(mvc['EMG_8'], fs=500, Rs=50, notch=True)

data = mvc['EMG_8']
Y = fft(data.values)
T = 32
dt = 1.0 / fs
t = np.arange(0, T, dt)


# fig, axs = plt.subplots(1, 2)
# plt.grid()
# axs[0].plot(t, data)
# axs[0].set_title("Surowy sygnał")
# plt.grid()
# axs[1].plot(t, signal_filtered)
# axs[1].set_title("Przefiltowany sygnał")
# plt.show()

fs = 500
xf = fftfreq(len(data), 1/fs)
# fig, axs = plt.subplots(1, 2)
# plt.grid()
# axs[0].plot(xf, np.abs(Y))
# axs[0].set_title("Transformata surowego sygnału")
# plt.grid()
# axs[1].plot(xf, np.abs(fft(signal_filtered)))
# axs[1].set_title("Transformata przefiltrowanego sygnału")
# plt.show()

max_rms = rms(data=signal_filtered, fs=500)
data_signal_filtered = filter_emg(train['EMG_8'], fs=500, Rs=50, notch=True)

data_signal = train['EMG_8']
Y = fft(data_signal.values)
T = 32
dt = 1.0 / fs
t = np.arange(0, T, dt)


fig, axs = plt.subplots(1, 2)
plt.grid()
axs[0].plot(data_signal)
axs[0].set_title("Surowy sygnał")
plt.grid()
axs[1].plot(data_signal_filtered)
axs[1].set_title("Przefiltowany sygnał")
plt.show()

fs = 500
xf = fftfreq(len(data_signal), 1/fs)
fig, axs = plt.subplots(1, 2)
plt.grid()
axs[0].plot(xf, np.abs(Y))
axs[0].set_title("Transformata surowego sygnału")
plt.grid()
axs[1].plot(xf, np.abs(fft(data_signal_filtered)))
axs[1].set_title("Transformata przefiltrowanego sygnału")
plt.show()

norm_emgs = norm_emg(data_signal_filtered, max_rms)
fig, axs = plt.subplots(1, 2)
plt.grid()
axs[0].plot(data_signal_filtered)
axs[0].set_title("Przefiltowany sygnał")
plt.grid()
axs[1].plot(norm_emgs)
axs[1].set_title("Znormalizowany sygnał")
plt.show()

# decymacja sygnału - f. dolnoprzepustowy - połowa częstotliwości i wybieram co 5-tą próbkę