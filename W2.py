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
    N = 100
    h = sig.firwin2(N, [0, 1200, 1400, fs/2], [1, 1, 0, 0], N+1, window=None, fs=fs)
    plt.plot(h)
    plt.title('Odpowiedź impulsowa filtru')
    plt.show()

def zad4():
    fs = 48000
    N = 1001
    h = sig.firwin2(N, [0, 1200, 1400, fs / 2], [1, 1, 0, 0], N + 1, window=None, fs=fs)
    plt.plot(h)
    plt.title('Odpowiedź impulsowa filtru z wydłużoną liczbą próbek')
    plt.show()

    plt.plot(np.abs(fft(h)))
    plt.title('Charakterystyka filtru')
    plt.show()

def main():
    # zad2()
    zad3()

if __name__ == '__main__':
    main()