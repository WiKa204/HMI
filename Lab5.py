import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft


def sin(f=50, T=2, Fs=500, phi=0):
    dt = 1.0 / Fs
    t = np.arange(0, T, dt)
    s = np.sin(2 * np.pi * f * t + phi)
    return (s, t)

def cos(f=50, T=2, Fs=500, phi=0):
    dt = 1.0 / Fs
    t = np.arange(0, T, dt)
    s = np.cos(2 * np.pi * f * t + phi)
    return (s, t)

# Sygnał bazowy
n = 1000
fs = 48000
t = np.arange(n) / fs
fr = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]).reshape(-1, 1)
x = np.sum(np.sin(2 * np.pi * t * fr), axis=0)
x = x + 0.05 * np.random.randn(len(x))
x = x / np.max(np.abs(x))
# normalizacja częstotliowści
xf = fftfreq(len(x), 1/fs)
# plt.plot(x)
# plt.title('sygnał bazowy')
# plt.show()

# TODO 1 Dla sygnału  fs=48kHz , zaprojektuj 4 filtry o długości  N=101  próbek z oknem Hamminga:
#  górno przepustowy ( fp=3kHz ), dolno-przepustowy ( fp=3kHz ), pasmowo przepustowy ( 1−3kHz ),
#  pasmowo zaporowy ( 1−3kHz ). Wyświetl charakterystyki fazowe oraz odpowiedzi impulsowe.
def zad1():
    N = 101
    fc = 1000
    fp = 3000
    y = fft(x)
    plt.plot(xf, np.abs(y))
    plt.title('Transformata Fouriera sygnału bazowego')
    plt.show()
    filtr_hp = sig.firwin2(N, [0, fp, fp+200, fs / 2], [0, 0, 1, 1], N + 1, fs=fs)                      #highpass
    filtr_lp = sig.firwin2(N, [0, fp-200, fp, fs / 2], [1, 1, 0, 0], N + 1, fs=fs)                      #lowpass
    filtr_bp = sig.firwin2(N, [0, fc-200, fc, fp, fp+200, fs / 2], [0, 0, 1, 1, 0, 0], N + 1, fs=fs)    #bandpass
    filtr_bs = sig.firwin2(N, [0, fc-200, fc, fp, fp+200, fs / 2], [1, 1, 0, 0, 1, 1], N + 1, fs=fs)    #bandstop

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(filtr_hp)
    axs[0, 0].set_title('Odpowiedź impulsowa filtru - górno przepustowego')
    plt.grid()
    axs[1, 0].plot(filtr_lp)
    axs[1, 0].set_title('Odpowiedź impulsowa filtru - dolno przepustowego')
    axs[1, 0].sharex(axs[0, 0])
    plt.grid()
    axs[0, 1].plot(filtr_bp)
    axs[0, 1].set_title('Odpowiedź impulsowa filtru - pasmowo przepustowego')
    plt.grid()
    axs[1, 1].plot(filtr_bs)
    axs[1, 1].set_title('Odpowiedź impulsowa filtru - pasmowo zaporowego')
    plt.grid()
    fig.tight_layout()
    plt.show()

    # charakterystyka częstotliwościowa
    w_hp, hf_hp = sig.freqz(filtr_hp, worN=2048, fs=fs)
    w_lp, hf_lp = sig.freqz(filtr_lp, worN=2048, fs=fs)
    w_bp, hf_bp = sig.freqz(filtr_bp, worN=2048, fs=fs)
    w_bs, hf_bs = sig.freqz(filtr_bs, worN=2048, fs=fs)

    hfdb_hp = 20 * np.log10(np.abs(hf_hp))
    phase_hp = np.degrees(np.angle(hf_hp))

    hfdb_lp = 20 * np.log10(np.abs(hf_lp))
    phase_lp = np.degrees(np.angle(hf_lp))

    hfdb_bp = 20 * np.log10(np.abs(hf_bp))
    phase_bp = np.degrees(np.angle(hf_bp))

    hfdb_bs = 20 * np.log10(np.abs(hf_bs))
    phase_bs = np.degrees(np.angle(hf_bs))

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(w_hp, phase_hp)
    axs[0, 0].set_ylim(-180, 180)
    axs[0, 0].set_xlabel('Częstotliwość [Hz]')
    axs[0, 0].set_ylabel('Poziom widma [dB]')
    axs[0, 0].set_title('Charakterystyka fazowa filtru GP 3 kHz')
    plt.grid()
    axs[1, 0].plot(w_lp, phase_lp)
    axs[1, 0].set_ylim(-180, 180)
    axs[1, 0].set_xlabel('Częstotliwość [Hz]')
    axs[1, 0].set_ylabel('Poziom widma [dB]')
    axs[1, 0].set_title('Charakterystyka fazowa filtru DP 3 kHz')
    axs[1, 0].sharex(axs[0, 0])
    plt.grid()
    axs[0, 1].plot(w_bp, phase_bp)
    axs[0, 1].set_ylim(-180, 180)
    axs[0, 1].set_xlabel('Częstotliwość [Hz]')
    axs[0, 1].set_ylabel('Poziom widma [dB]')
    axs[0, 1].set_title('Charakterystyka fazowa filtru PP 1-3 kHz')
    plt.grid()
    axs[1, 1].plot(w_bs, phase_bs)
    axs[1, 1].set_ylim(-180, 180)
    axs[1, 1].set_xlabel('Częstotliwość [Hz]')
    axs[1, 1].set_ylabel('Poziom widma [dB]')
    axs[1, 1].set_title('Charakterystyka fazowa filtru PZ 1-3 kHz')
    plt.grid()
    fig.tight_layout()
    plt.show()

# TODO 2 Zakładając, że  fs=500Hz , a okres obseracji wynosi  T=2s  przygotować wektor sygnału testowego o postaci:
#  x(t)= sin(3⋅2π⋅t)+cos(10⋅2π⋅t)+cos(25⋅2π⋅t)+sin(35⋅2π⋅t)+sin(50⋅2π⋅t)+sin(100⋅2π⋅t)
def zad2():
    fs=500
    s1, t_s1 = sin(f=3)
    s2, t_s2 = sin(f=35)
    s3, t_s3 = sin(f=50)
    s4, t_s4 = sin(f=100)

    c1, t_c1 = cos(f=10)
    c2, t_c2 = cos(f=25)
    x = s1+s2+s3+s4+c1+c2
    plt.plot(t_c2, x)
    plt.title('Sygnał testowy')
    plt.show()
    y = fft(x)
    xf = fftfreq(len(x), 1/fs)
    plt.plot(xf, np.abs(y))
    plt.title('Transformata sygnału testowego')
    plt.show()
    t = t_s1
    return (x, t)

# TODO 3: Zaprojektować filtr dolnoprzepustowy nierekursywny, który umożliwi stłumienie składowych 50 i 100Hz.
#  Dla okna Kaisera przyjąć, że tłumienie w paśmie zaporowym nie powinno być mniejsze niż 55dB,
#  zaś oscylacje w paśmie przepustowym nie powinny przekraczać 0.1%. Szerokość pasma przejściowego ustalić na 5Hz.
#  Do projektowania wykorzystać metodę firwin oraz okno Kaisera.
#  Po zaprojektowaniu przefiltruj sygnał i wyznacz opóźnienie filtra.
def zad3(signal: np.array, time: np.array, cut_off: int):
    fs =500
    Rs = 55
    width = 5
    cut_off = cut_off
    numtaps, beta = sig.kaiserord(Rs, width / (0.5 * fs))
    print(f'numpas {numtaps}, beta: {beta}')
    taps = sig.firwin(numtaps, cut_off, window=('kaiser', beta), pass_zero='lowpass',
                         scale=False, nyq=0.5 * fs)
    w, h = sig.freqz(taps)
    w *= 0.5 * fs / np.pi  # Convert w to Hz
    ideal = w < cut_off  # The "ideal" frequency response.
    deviation = np.abs(np.abs(h) - ideal)
    deviation[(w > cut_off - 0.5 * width) & (w < cut_off + 0.5 * width)] = np.nan

    plt.plot(w, 20 * np.log10(np.abs(deviation)))
    plt.xlim(0, 0.5 * fs)
    plt.ylim(-90, -50)
    plt.grid(alpha=0.25)
    plt.axhline(-55, color='r', ls='--', alpha=0.3)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Deviation from ideal (dB)')
    plt.title('Lowpass Filter Frequency Response')
    plt.show()

    # Zastosowanie filtru
    y = sig.lfilter(taps, 1, signal)
    plt.plot(y, label='po')
    plt.plot(signal, label='przed')
    plt.title('Sygnał przed i po filtracji')
    plt.legend()
    plt.show()
    return taps

# TODO 4: Zaprojektuj filtr pasmowo-zaporowy dla składowej 25Hz i 50Hz, załóż że pasmo zaporowe ma szerokość 1Hz,
#  a pasmo przejściowe o szerokości 5Hz. Wybierz metodę najmniejszych kwadratów (firls).
#  Jakie jest najsłabsze tłumienie w paśmie zaporowym?
def zad4(signal: np.array, xf: np.array):
    fs = 500
    N = 101
    h = sig.firls(N, [0, 19.5, 24.5, 25.5, 30.5, 44.5, 49.5, 50.5, 55.5, fs / 2], [1, 1, 0, 0, 1, 1, 0, 0, 1, 1], fs=fs)
    plt.plot(h)
    plt.title('Odpowiedź impulsowa filtru - metoda najmniejszych kwadratów')
    plt.show()

    y = sig.lfilter(h, 1, signal)
    plt.plot(y, label='po')
    plt.plot(signal, label='przed')
    plt.title('Sygnał przed i po filtracji')
    plt.legend()
    plt.show()

    plt.plot(xf, np.abs(fft(y)), label='po')
    plt.plot(xf, np.abs(fft(signal)), label='przed')
    plt.title('Sygnał przed i po filtracji - transformata Fouriera')
    plt.legend()
    plt.show()

    w, hf = sig.freqz(h, worN=248, fs=fs)  # odp. imp. filtra, rozdzielczość (ilość próbek), fs sygnału
    plt.plot(w, 20 * np.log10(np.abs(hf)))  # przedstawienie w skali logarytmicznej
    plt.title('Analiza częstotliwościowa')
    plt.show()

def main():
    # zad1()
    x, t = zad2()
    fs=500
    xf = fftfreq(len(x), 1 / fs)
    h_50 = zad3(signal=x, time=t, cut_off=50)
    h_100 = zad3(signal=x, time=t, cut_off=100)
    h = h_50 * h_100
    y = sig.lfilter(h, 1, x)
    plt.plot(y, label='po')
    plt.plot(x, label='przed')
    plt.title('Sygnał przed i po filtracji')
    plt.legend()
    plt.show()

    # Analiza za pomocą transformaty Fouriera
    plt.plot(xf, np.abs(fft(y)), label='po')
    plt.plot(xf, np.abs(fft(x)), label='przed')
    plt.title('Sygnał przed i po filtracji - transformata Fouriera')
    plt.legend()
    plt.show()

    # metoda do analizy częstotliwościowej
    w, hf = sig.freqz(h, worN=248, fs=fs)  # odp. imp. filtra, rozdzielczość (ilość próbek), fs sygnału
    plt.plot(w, 20 * np.log10(np.abs(hf)))  # przedstawienie w skali logarytmicznej
    plt.title('Analiza częstotliwościowa')
    plt.show()

    zad4(signal=x, xf=xf)

if __name__ == '__main__':
    main()