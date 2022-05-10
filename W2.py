import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Sygnał bazowy
n = 1000
t = np.arange(n) / 48000
fr = np.array([500, 1000, 1500, 2000, 2500]).reshape(-1, 1)
x = np.sum(np.sin(2 * np.pi * t * fr), axis=0)
x = x + 0.05 * np.random.randn(len(x))
x = x / np.max(np.abs(x))
# normalizacja częstotliowści
xf = fftfreq(len(x), 1/48000)

# TODO Wyświetlenie sygnału bazowego
def zad1():
    plt.plot(x)
    plt.show()

# TODO Wyświetlenie widma sygnału
def zad2():
    plt.plot(xf, np.abs(fft(x)))
    plt.show()

# TODO Zaprojektowanie filtru FIR używając metody okien, który wytnie składowe powyżej 1200 Hz
def zad3():
    fs = 48000
    N = 101 # 501 # zwiększenie liczby próbek wprowadza opóźnienie
    h = sig.firwin2(N, [0, 1200, 1400, fs/2], [1, 1, 0, 0], N+1, window=None, fs=fs)
    plt.plot(h)
    plt.title('Odpowiedź impulsowa filtru - okno prostokątne') # NA KRAŃCACH WIDAĆ NIECIĄGŁOŚCI
    plt.show()
    h = sig.firwin2(N, [0, 1200, 1400, fs / 2], [1, 1, 0, 0], N + 1, window='blackman', fs=fs)
    plt.plot(h)
    plt.title('Odpowiedź impulsowa filtru - okno Blackmana')
    plt.show()
    h = sig.firwin2(N, [0, 1200, 1400, fs / 2], [1, 1, 0, 0], N + 1, window='hann', fs=fs)
    plt.plot(h)
    plt.title('Odpowiedź impulsowa filtru - okno Hanna')  # NA KRAŃCACH WYGŁADZENIE
    plt.show()


    # Zastosowanie filtru
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
    w, hf = sig.freqz(h, worN=248, fs=fs)   # odp. imp. filtra, rozdzielczość (ilość próbek), fs sygnału
    plt.plot(w, 20*np.log10(np.abs(hf)))      # przedstawienie w skali logarytmicznej
    plt.title('Analiza częstotliwościowa')
    plt.show()

    # Polepszenie filtra, by mocniej tłumił - metoda najmniejszych kwadratów
    h = sig.firls(N, [0, 1200, 1400, fs / 2], [1, 1, 0, 0], fs=fs)
    plt.plot(h)
    plt.title('Odpowiedź impulsowa filtru - metoda najmniejszych kwadratów')
    plt.show()

def zad4():
    fs = 48000
    N = 1001
    h = sig.firwin2(N, [0, 1200, 1400, fs / 2], [1, 1, 0, 0], N + 1, window=None, fs=fs)

    plt.plot(np.abs(fft(h)))
    plt.title('Charakterystyka filtru')
    plt.show()

def main():
    # zad2()
    zad3()
    # zad4()

if __name__ == '__main__':
    main()