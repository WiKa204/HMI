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

    if notch==True:

        # Pass band ripple in dB
        Ap = 0.4

        # Stop band attenuation in dB
        As = 50
        fp = np.array([48, 52]) # fpass [Hz]
        fz = np.array([49, 51])  # fstop [Hz]
        Rp = 3  # -3dB w paśmie przepustowym
        Rs = 100  # -100db w paśmie zaporowym
        fs = 500  # częstotliwość próbkowania
        N, fn = signal.ellipord(fp, fz, Rp, Rs, fs=fs)
        b, a = signal.ellip(N, Rp, Rs, fn, 'bandstop', fs=fs)
        signal_filtered = signal.filtfilt(b, a, signal_filtered, method="gust", padlen=0)

    return signal_filtered

train = pd.read_hdf('train.hdf5')
mvc = pd.read_hdf('mvc.hdf5')

# print(train.columns)
# train.filter(regex='EMG_').plot()
# train.filter(regex='EMG_[1]').plot()
# train.filter(regex='EMG_9').plot()
# train[['EMG_9', 'TRAJ_GT']].plot()
# plt.show()

# print(mvc.columns)
# mvc['EMG_9'].plot()
# plt.show()

fs = 1920

signal_filtered = filter_emg(mvc['EMG_9'], fs=500, Rs=50, notch=True)

data = mvc['EMG_9']
Y = fft(data.values)
T = 32
dt = 1.0 / fs
t = np.arange(0, T, dt)


fig, axs = plt.subplots(1, 2)
plt.grid()
axs[0].plot(t, data)
axs[0].set_title("Surowy sygnał")
plt.grid()
axs[1].plot(t, signal_filtered)
axs[1].set_title("Przefiltowany sygnał")
plt.show()

fs = 500
xf = fftfreq(len(data), 1/fs)
fig, axs = plt.subplots(1, 2)
plt.grid()
axs[0].plot(xf, np.abs(Y))
axs[0].set_title("Transformata surowego sygnału")
plt.grid()
axs[1].plot(xf, np.abs(fft(signal_filtered)))
axs[1].set_title("Transformata przefiltrowanego sygnału")
plt.show()

# notch filter - 50 Hz - f. eliptyczny - wyświetlić fouriera
# decymacja sygnału - f. dolnoprzepustowy - połowa częstotliwości i wybieram co 5-tą próbkę