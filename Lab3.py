import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq, ifft

def Zad1():
    def sin(f=1, T=1, Fs=128, phi=0):
        dt = 1.0 / Fs
        t = np.arange(0, T, dt)
        s = np.sin(2 * np.pi * f * t + phi)
        return (s, t)

    Fs = 100
    T = 0.1
    f_1 = 10
    f_2 = 22

    (x1, t) = sin(f=f_1, T=T, Fs=Fs)
    (x2, t) = sin(f=f_2, T=T, Fs=Fs)
    x3 = x1 * x2

    x_z1 = np.append(x1, np.zeros(10))
    x_z2 = np.append(x2, np.zeros(10))
    x_z3 = np.append(x3, np.zeros(10))

    X1 = fft(x1)
    X2 = fft(x2)
    X3 = fft(x3)

    X_z1 = fft(x_z1)
    X_z2 = fft(x_z2)
    X_z3 = fft(x_z3)

    xf1 = fftfreq(len(X1), 1 / Fs)
    xf2 = fftfreq(len(X2), 1 / Fs)
    xf3 = fftfreq(len(X3), 1 / Fs)

    xf_z1 = fftfreq(len(X_z1), 1 / Fs)
    xf_z2 = fftfreq(len(X_z2), 1 / Fs)
    xf_z3 = fftfreq(len(X_z3), 1 / Fs)

    x_w1 = np.append(signal.windows.blackman(len(x1)) * x1, np.zeros(10))
    x_w2 = np.append(signal.windows.blackman(len(x2)) * x2, np.zeros(10))
    x_w3 = np.append(signal.windows.blackman(len(x3)) * x3, np.zeros(10))

    X_w1 = fft(x_w1)
    X_w2 = fft(x_w2)
    X_w3 = fft(x_w3)

    fig, axs = plt.subplots(3, 4)
    axs[0, 0].plot(t, x1)
    axs[0, 0].set_title("f=10")
    plt.grid()
    axs[1, 0].plot(t, x2)
    axs[1, 0].set_title("f=22")
    plt.grid()
    axs[2, 0].plot(t, x3)
    axs[2, 0].set_title("x1*x2")
    plt.grid()
    axs[0, 1].stem(xf1, np.abs(X1), use_line_collection=True)
    axs[0, 1].set_title("widmo f=10")
    plt.grid()
    axs[1, 1].stem(xf2, np.abs(X2), use_line_collection=True)
    axs[1, 1].set_title("widmo f=22")
    plt.grid()
    axs[2, 1].stem(xf3, np.abs(X3), use_line_collection=True)
    axs[2, 1].set_title("widmo x1*x2")
    plt.grid()
    axs[0, 2].stem(xf_z1, np.abs(X_z1), use_line_collection=True)
    axs[0, 2].set_title("zero_padding; f=10")
    plt.grid()
    axs[1, 2].stem(xf_z2, np.abs(X_z2), use_line_collection=True)
    axs[1, 2].set_title("zero_padding; f=22")
    plt.grid()
    axs[2, 2].stem(xf_z3, np.abs(X_z3), use_line_collection=True)
    axs[2, 2].set_title("zero_padding; x1*x2")
    plt.grid()
    axs[0, 3].stem(xf_z1, np.abs(X_w1), use_line_collection=True)
    axs[0, 3].set_title("zero_padding+Hann; f=10")
    plt.grid()
    axs[1, 3].stem(xf_z2, np.abs(X_w2), use_line_collection=True)
    axs[1, 3].set_title("zero_padding+Hann; f=22")
    plt.grid()
    axs[2, 3].stem(xf_z3, np.abs(X_w3), use_line_collection=True)
    axs[2, 3].set_title("zero_padding+Hann; x1*x2")
    plt.grid()
    fig.tight_layout()
    plt.show()


def main():
    Zad1()

if __name__ == '__main__':
    main()