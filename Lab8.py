import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


# Dla wszystkich filtrów przyjmij, że w paśmie przepustowym oscylacje sygnału po pełnej filtracji
# (wykonanej przez daną funkcję) nie powinny zmienić się o więcej niż o 1%.
# 1% = 20 dB
def filter_emg(data: np.array, fs: int, Rs: int, notch: bool):
    # artefakty ruchowe - usunąć wszystko poniżej 15 Hz
    # filtr pasmowy dla zakłóceń sieciowych 50, 100, 150 ...
    xf = fftfreq(len(data), 1 / fs)
    print(f'długość sygnału: {len(data)}')
    width = 4
    cut_off = 15
    numtaps, beta = signal.kaiserord(Rs, width / (0.5 * fs))
    print(f'numpas {numtaps}, beta: {beta}')
    filtr = signal.firwin(numtaps + 1, cut_off, window=('kaiser', beta), pass_zero='highpass',
                          scale=False, fs=fs)
    plt.plot(filtr)
    plt.title('Odpowiedź impulsowa filtru - okno Kaisera')  # NA KRAŃCACH WIDAĆ NIECIĄGŁOŚCI
    plt.show()
    w, h = signal.freqz(filtr, worN=2048, fs=fs)
    h_db = 20 * np.log10(np.abs(h))
    phase = np.degrees(np.angle(h))

    plt.figure()
    fig1, ax1 = plt.subplots(2, sharex=True, tight_layout=True, figsize=(8, 5))
    ax1[0].plot(w, h_db)
    ax1[0].set_xlim(0, 125)
    ax1[0].set_ylim(bottom=-80)
    ax1[0].set_xlabel('Częstotliwość [Hz]')
    ax1[0].set_ylabel('Poziom widma [dB]')
    ax1[0].set_title('Charakterystyka częstotliwościowa')
    ax1[0].grid()

    ax1[1].axvline(3000, c='k', lw=1, ls='--')
    ax1[1].plot(w, phase)
    ax1[1].grid()
    ax1[1].set_ylim(-180, 180)
    ax1[1].set_xlabel('Częstotliwość [Hz]')
    ax1[1].set_ylabel('Faza [°]')
    ax1[1].set_title('Charakterystyka fazowa filtru GP 15 Hz')
    plt.show()

    ideal = np.ones(len(data))
    ideal[xf<15] = -50
    # charakterystyka częstotliwościowa + sygnał
    plt.plot(xf, ideal, label="Idealny")
    plt.plot(w, h_db, label="zaprojektowany")
    plt.xlim(0, 125)
    plt.ylim(bottom=-80)
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Poziom widma [dB]')
    plt.title('Charakterystyka częstotliwościowa')
    plt.legend()
    plt.grid()
    plt.show()

    # filtracja sygnału
    signal_filtered = signal.lfilter(filtr, 1, data) # data
    signal_filtered_zero_ph = signal.filtfilt(filtr, 1, data, padlen=0)  # data
    plt.plot(xf, signal_filtered_zero_ph)
    plt.title('Sygnał zero phase')
    plt.show()
    plt.plot(xf, np.abs(fft(signal_filtered_zero_ph)))
    plt.title('Widmo zero phase')
    plt.show()

    # widmo sygnału przed i po filtracji
    plt.figure()
    plt.plot(xf, data, label='Oryginalny')
    plt.plot(xf, signal_filtered, label='Po filtracji')
    plt.xlim(0, 125)
    # plt.ylim(bottom=0)
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Poziom widma [dB]')
    plt.legend()
    plt.grid()
    plt.title('Widmo sygnału przed i po filtracji')
    plt.show()
    Y = fft(signal_filtered)
    plt.plot(xf, np.abs(Y))
    plt.title('Widmo sygnału emg po filtracji')
    plt.show()

    if notch==True:
        df = 2
        Rp = 20  # 20 max dla zadania
        freqs = [50, 100, 150, 200, 250]
        N, fn = signal.ellipord(47, fs/2, Rp, Rp, fs=fs)
        be, ae = signal.ellip(N, Rp, Rp, fn, 'low', fs=fs)
        we, he = signal.freqz(be, ae, 2048, fs=fs)
        hed = 20 * np.log10(np.abs(he))
        for fr in freqs:
            fd = fr - df
            fg = fr + df
            if fg>250:
                fg=249
            N = 501
            h = signal.firwin(N, (fd, fg), pass_zero='bandstop', fs=fs)
            signal_filtered = signal.lfilter(h, 1, signal_filtered)

            plt.plot(signal_filtered, label='po')
            plt.plot(data, label='przed')
            plt.title(f'Sygnał przed i po filtracji składowej {fr} Hz')
            plt.legend()
            plt.show()

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(data)
    axs[0].set_title('Sygnał przed filtracją')
    plt.grid()
    axs[1].plot(signal_filtered)
    axs[1].set_title('Sygnał po filtracji')
    plt.show()

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(xf, np.abs(fft(data)))
    axs[0].set_title('Widmo przed filtracją')
    plt.grid()
    axs[1].plot(xf, np.abs(fft(signal_filtered)))
    axs[1].set_title('Widmo po filtracji')
    plt.show()

    signal_filtered = data
    signal_filtered_zero_ph = data
    return signal_filtered, signal_filtered_zero_ph


def subsample_emg(data: np.array, fs: int, r: int, Rs: int):
    return data


def filter_force(data: np.array, fs: int):

    signal_filtered = data
    signal_filtered_zero_ph = data
    return signal_filtered, signal_filtered_zero_ph

# TODO 1:Wczytaj sygnał zawierający EMG brzuchatego łydki i napisz funkcję
# Funkcja filtrująca powinna:
# - usunąć artefakty ruchowe bez zmiany kształtu sygnału, gdzie tłumienie powinno wynosić nie mniej niż Rs [dB]
# - usunąć zakłócenie sieciowe i harmoniczne, gdzie oczekiwane stłumienie składowej (i przecieków) zakłócenia harmonicznego
# nie powinno być mniejsze niż -20dB (na tej podstawie proszę dobrać szerokość pasma) i powinno być włączane argumentem notch=True
# Funkcja powinna zwracać:
# sygnał po filtracji: signal_filtered,
# sygnał po filtracji zerofazowej: signal_filtered_zero_ph

emg: np.array = pd.read_csv('emg.csv') # 15 - 5000 Hz
fs = 500

emg.info()
data = emg['emg']
plt.plot(data)
plt.title('Sygnał emg')
plt.show()
xf = fftfreq(len(data), 1/fs)
plt.plot(xf, np.abs(fft(data)))
plt.title('Widmo sygnału emg')
plt.show()

signal_filtered, signal_filtered_zero_ph = filter_emg(data, fs=500, Rs=50, notch=True)

# TODO 2: Dla przefiltrowanego sygnału napisz funkcję, która dokona subsamplingu sygnału o r razy (r jest typu int).
#  Pamiętaj, żeby w przefiltrowanym sygnale nie było aliasów
# Filtracja antyaliasingowa nie powinna istotnie zmieniać kształtu sygnału, oraz zapewniać, że ew aliasy będą stłumione
# o nie mniej niż o Rs (wyrażone w dB)

signal_subsampled = subsample_emg(signal_filtered, fs=500, r=3, Rs=30)

# TODO 3: Wczytaj sygnał, który zawiera sygnał siły skurczu mięśnia wywołanej stymulacją elektryczną. Impulsy stymulacji
#  elektrycznej  50−200μs  i amplitudzie do 200V przenoszą się do ukłądu pomiarowego siły, i tworzą w zapisie
#  charakterystyczne piki. Ponadto w przebiegu widoczny jest szum kwantyzacji przetwornika ADC. Zastanów się jak stosując
#  filtrację można zmniejszyć amplitudę pików oraz wyeliminować szum kwantyzacji, nie modyfikując kształtu zarejestrowanego
#  sygnału. Częstotliwość próbkowania sygnału fs=5kHz Należy wiedzieć że widmo wąskiego impulsu jest zbliżone do widma delty Diraca.


force: np.array = pd.read_csv('force.csv')
print(force.info())
print(force.head())
plt.plot(force['force'].squeeze())
plt.title('Sygnał force')
plt.show()
fs = 5000

# print(data_3.to_string())
# print(data_3.head())
# data_3['force'].plot()
# plt.show()
signal_3 = force['force']
signal_filtered, signal_zero_ph = filter_force(force, fs)


def filter_emg2(data: np.array, fs: int, Rs: int, notch: bool):
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


