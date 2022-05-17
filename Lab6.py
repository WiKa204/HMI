import numpy as np
import scipy.signal as sig
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft

matplotlib.rcParams['figure.figsize'] = (8, 4)

# Metoda okienkowa i wizualizacja
def wprowadzenie1():
    """Filtracja zakłóceń z sygnału."""

    wav = np.load('./data/distorted.npy')
    wav = wav / np.max(np.abs(wav))
    fs = 48000
    t = np.arange(len(wav)) / fs

    # oryginalny - postać czasowa
    plt.figure()
    plt.plot(t[:500], wav[:500])
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.title('Sygnał oryginalny - postać czasowa')

    # oryginalny - widmo
    spectrum = np.fft.rfft(wav * np.hamming(2048))
    spdb = 20 * np.log10(np.abs(spectrum) / 1024)
    f = np.fft.rfftfreq(2048, 1 / 48000)
    plt.figure(figsize=(8, 4))
    plt.plot(f, spdb)
    ideal = np.zeros_like(spdb)
    ideal[f > 3000] = -60
    # plt.plot(f, ideal, c='r')
    plt.xlim(0, 12000)
    plt.ylim(bottom=-60)
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Poziom widma [dB]')
    plt.title('Sygnał oryginalny - postać widmowa')

    # filtr dolnoprzepustowy
    fc = 1900  # częstotliwość graniczna
    N = 101  # długość filtru
    h = sig.firwin(N, fc, pass_zero='lowpass', fs=48000) # projektowanie metodą okien

    # z, p, k = sig.tf2zpk(h, 1)
    # print(len(z), len(p))

    # odpowiedź impulsowa
    plt.figure(figsize=(10, 4))
    plt.stem(h, use_line_collection=True)
    plt.xlabel('Indeks')
    # plt.ylabel('Amplituda')
    plt.title('Odpowiedź impulsowa filtru')
    # plt.title('Współczynniki h')

    # charakterystyka częstotliwościowa
    w, hf = sig.freqz(h, worN=2048, fs=48000)
    hfdb = 20 * np.log10(np.abs(hf))
    phase = np.degrees(np.angle(hf))

    plt.figure()
    fig1, ax1 = plt.subplots(2, sharex=True, tight_layout=True, figsize=(8, 5))
    ax1[0].plot(w, hfdb)
    ax1[0].set_xlim(0, 12000)
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
    ax1[1].set_title('Charakterystyka fazowa filtru DP 3 kHz')

    # charakterystyka częstotliwościowa + sygnał
    plt.figure()
    plt.plot(f, spdb)
    plt.plot(w, hfdb)
    plt.xlim(0, 12000)
    plt.ylim(bottom=-80)
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Poziom widma [dB]')
    plt.title('Charakterystyka częstotliwościowa')
    plt.grid()

    # filtracja sygnału
    y = sig.lfilter(h, 1, wav)      # 1 = mianownik transmitancji filtra y(t) = h(t)/1  *   x(t)    # splot sygnałów w dziedzinie czasu
    ysp = np.fft.rfft(y * np.hamming(2048))
    yspdb = 20 * np.log10(np.abs(ysp) / 1024)

    # widmo sygnału przed i po filtracji
    plt.figure()
    plt.plot(f, spdb, c='#a0a0a0', label='Oryginalny')
    plt.plot(f, yspdb, label='Po filtracji')
    plt.xlim(0, 12000)
    # plt.ylim(bottom=0)
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Poziom widma [dB]')
    plt.legend()
    plt.grid()
    plt.title('Widmo sygnału przed i po filtracji')

    # postać czasowa przed i po filtracji
    fig7, ax7 = plt.subplots(2, figsize=(8, 5), sharex=True, tight_layout=True)
    ax7[0].plot(t[:500], wav[:500], label='Originalny')
    ax7[1].plot(t[:500], y[50:550], label='Przetworzony')
    ax7[-1].set_xlabel('Czas [s]')
    for a in ax7:
        a.set_ylabel('Amplituda')
        a.legend(loc='upper right')
    ax7[0].set_title('Sygnał oryginalny i przetworzony - postać czasowa')

    plt.show() #

# Okreslanie dowolnej liczby współczynników
def wprowadzenie2():
    n = 1000
    fs = 4800
    t = np.arange(n) / fs
    fr = np.array([500, 1000, 1500, 2000, 2500]).reshape(-1, 1)
    fr_ideal = np.array([500, 1000]).reshape(-1, 1)
    x = np.sum(np.sin(2 * np.pi * t * fr), axis=0)
    x_ideal = np.sum(np.sin(2 * np.pi * t * fr_ideal), axis=0)

    # x = x + 0.05 * np.random.randn(len(x))      # Dodawanie szumu
    # x = x / np.max(np.abs(x))       # normalizacja sygnału, dzielenie przez maksimum

    h = sig.firwin2(801, [0, 1250, 1300, fs / 2], [1, 1, 0, 0], window='hamming', fs=fs)
    # sig.firwin2(długość filtru, [wartości częstotliwości], [wzmocnienia], okno, częstotliwość próbkowania)
    # częstotliwości podaje się parami
    y = sig.lfilter(h, 1, x) # wyznaczenie odpowiedzi czasowej

    fig, ax = plt.subplots(2, sharex=True, tight_layout=True, figsize=(8, 5))
    ax[0].plot(x[:n])
    # ax[1].plot(y[:n])
    # Nałożenie sygnałów, uwdględnienie oóźnienia filtru
    # ax[1].plot(x_ideal[:n], label='sygnał idealny')
    # ax[1].plot(y[400:n], label='sygnał po filtracji')
    # Różnica sygnałów
    ax[1].plot(y[400:n]-x_ideal[:n-400])
    for a in ax:
        a.grid()
        a.set_ylabel('Amplituda')
    ax[1].set_xlabel('Nr próbki')
    ax[0].set_title('Sygnał wejściowy')
    ax[1].set_title('Sygnał po filtracji (N=801)')
    plt.legend()
    plt.show()

def zad1():
    # Sygnał bazowy
    n = 1000
    fs = 48000
    t = np.arange(n) / fs
    fr = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]).reshape(-1, 1)
    x = np.sum(np.sin(2 * np.pi * t * fr), axis=0)
    x = x + 0.05 * np.random.randn(len(x))
    x = x / np.max(np.abs(x))
    # normalizacja częstotliowści
    xf = fftfreq(len(x), 1 / fs)
    plt.plot(x)
    plt.title('sygnał bazowy')
    plt.show()

    N = 101
    fp = 3000
    fs = 48000
    filtr_hp = sig.firwin2(N, [0, fp-200, fp, fs / 2], [0, 0, 1, 1], N + 1, fs=fs)  # highpass

    plt.plot(filtr_hp)
    plt.title('Odpowiedź impulsowa filtru - górno przepustowego')
    plt.show()

    print(len(filtr_hp))
    window = np.zeros(101)
    window[49:52] = filtr_hp[49:52]
    plt.plot(window)
    plt.show()
    Rs = 55
    width = 5
    end = np.zeros(len(x)-101)
    numtaps, beta = sig.kaiserord(Rs, width / (0.5 * fs))
    w = np.append(sig.windows.kaiser(len(filtr_hp), beta), end)
    plt.plot(w)
    plt.title('Kaiser window')
    plt.show()

    # Zadanie2
    # Zastosowanie filtru
    y = sig.lfilter(w, 1, x)
    plt.plot(y, label='po')
    plt.plot(x, label='przed')
    plt.title('Sygnał przed i po filtracji')
    plt.legend()
    plt.show()

    #Zadanie3
    zi = sig.lfilter_zi(w, 1)
    out, zi = sig.lfilter(w, 1, x, zi=zi)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(out, label='po')
    axs[0, 0].plot(x, label='przed')
    axs[0, 0].set_title("'Sygnał przed i po filtracji - 1'")
    plt.legend()
    plt.grid()
    out, zi = sig.lfilter(w, 1, x, zi=zi)
    axs[1, 0].plot(out, label='po')
    axs[1, 0].plot(x, label='przed')
    axs[1, 0].set_title("Sygnał przed i po filtracji - 2")
    plt.grid()
    out, zi = sig.lfilter(w, 1, x, zi=zi)
    axs[1, 0].sharex(axs[0, 0])
    axs[0, 1].plot(out, label='po')
    axs[0, 1].plot(x, label='przed')
    axs[0, 1].set_title("Sygnał przed i po filtracji - 3")
    plt.grid()
    out, zi = sig.lfilter(w, 1, x, zi=zi)
    axs[1, 1].plot(out, label='po')
    axs[1, 1].plot(x, label='przed')
    axs[1, 1].set_title("Sygnał przed i po filtracji - 4")
    plt.grid()
    fig.tight_layout()
    plt.show()

def main():
    # wprowadzenie1()
    # wprowadzenie2()
    zad1()


if __name__=='__main__':
    main()