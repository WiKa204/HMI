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

def Zad2(nr=1):

    def sin(f=1, T=1, fs=32, phi=0):  # np.pi/2
        dt = 1.0 / fs
        t = np.arange(0, T, dt)
        s = np.sin(2 * np.pi * f * t + phi)
        return (s, t)

    def cos(f=16, T=1, fs=32, phi=0):  # np.pi/2
        dt = 1.0 / fs
        t = np.arange(0, T, dt)
        c = np.cos(2 * np.pi * f * t + phi)
        return (c, t)

    Tobs = 1
    fs = 32

    (S1, t1) = sin(f=1)
    (S2, t2) = sin(f=10)
    (S3, t3) = sin(f=16)
    (S4, t4) = sin(f=17)
    (C16, T16) = cos()

    Y1 = fft(S1)
    Y2 = fft(S2)
    Y3 = fft(S3)
    Y4 = fft(S4)
    Y16 = fft(C16)

    N = int(fs * Tobs)

    X = fftfreq(N, 1 / fs)

    if(nr == 1):
        plt.figure()
        plt.stem(X, np.abs(Y1), use_line_collection=True)
        plt.grid()
        plt.show()
        plt.stem(X, np.abs(Y2), use_line_collection=True)
        plt.grid()
        plt.show()
        plt.stem(X, np.abs(Y3), use_line_collection=True)
        plt.grid()
        plt.show()
        plt.stem(X, np.abs(Y4), use_line_collection=True)
        plt.grid()
        plt.show()
        plt.stem(X, np.abs(Y16), use_line_collection=True)
        plt.grid()
        plt.show()

    if(nr==2):

        x1 = ifft(Y1)
        x2 = ifft(Y2)
        x3 = ifft(Y3)
        x4 = ifft(Y4)

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(t1, x1)
        axs[0, 0].set_title("f=1")
        plt.grid()
        axs[1, 0].plot(t2, x2)
        axs[1, 0].set_title("f=10")
        plt.grid()
        axs[1, 0].sharex(axs[0, 0])
        axs[0, 1].plot(t3, x3)
        axs[0, 1].set_title("f=16")
        plt.grid()
        axs[1, 1].plot(t4, x4)
        axs[1, 1].set_title("f=17")
        plt.grid()
        fig.tight_layout()
        plt.show()

def Zad3():
    def sin(T=1, fs=10, phi=np.pi / 5):
        delta_f = 1 / T
        dt = 1.0 / fs
        t = np.arange(0, T, dt)
        s1 = np.sin(2 * np.pi * t)  # *2
        s2 = np.sin(2 * np.pi * 3 * t + phi)  # *4
        s = s1 + s2
        return (s, t)

    Tobs = 1
    Tobs1 = 10
    Tobs2 = 100
    fs = 10
    fs1 = 100
    fs2 = 1000

    (S1, t1) = sin()
    (S2, t2) = sin(fs=fs1)
    (S3, t3) = sin(fs=fs2)
    (S4, t4) = sin(T=Tobs1)
    (S5, t6) = sin(T=Tobs2)

    Y1 = fft(S1)
    Y2 = fft(S2)
    Y3 = fft(S3)
    Y4 = fft(S4)
    Y5 = fft(S5)

    N1 = int(fs * Tobs)
    N2 = int(fs1 * Tobs)
    N3 = int(fs2 * Tobs)
    N4 = int(fs * Tobs1)
    N5 = int(fs * Tobs2)

    X1 = fftfreq(N1, 1 / fs)
    X2 = fftfreq(N2, 1 / fs1)
    X3 = fftfreq(N3, 1 / fs2)
    X4 = fftfreq(N4, 1 / fs)
    X5 = fftfreq(N5, 1 / fs)

    fig, axs = plt.subplots(3, 2)
    axs[0, 0].stem(X1, np.abs(Y1), use_line_collection=True)
    axs[0, 0].set_title("T=1s, fs=10Hz")
    plt.grid()
    axs[1, 0].stem(X2, np.abs(Y2), use_line_collection=True)
    axs[1, 0].set_title("T=1s, fs=100Hz")
    plt.grid()
    axs[0, 1].stem(X3, np.abs(Y3), use_line_collection=True)
    axs[0, 1].set_title("T=1s, fs=1000Hz")
    plt.grid()
    axs[1, 1].stem(X4, np.abs(Y4), use_line_collection=True)
    axs[1, 1].set_title("T=10s, fs=10Hz")
    plt.grid()
    axs[2, 0].stem(X5, np.abs(Y5), use_line_collection=True)
    axs[2, 0].set_title("T=100s, fs=10Hz")
    plt.grid()
    fig.tight_layout()
    plt.show()

    print(f'Rozdzielczość częstotliwościowa 1s, 10 Hz: {1 / Tobs}')
    print(f'Rozdzielczość częstotliwościowa 1s, 100 Hz: {1 / Tobs}')
    print(f'Rozdzielczość częstotliwościowa 1s, 1000 Hz: {1 / Tobs}')
    print(f'Rozdzielczość częstotliwościowa 10s, 10 Hz: {1 / Tobs1}')
    print(f'Rozdzielczość częstotliwościowa 100s, 10 Hz: {1 / Tobs2}')

def Zad4():
    data = pd.read_hdf('./data/lab1_emg.hdf5')
    EMG_15 = data['EMG_15']
    down_sampling = EMG_15.iloc[::10]
    plt.figure()
    xf = fftfreq(len(data), 1/5120)
    xf2 = fftfreq(len(down_sampling), 1 / 5120)
    plt.plot(xf, np.abs(fft(data['EMG_15'].values)))
    plt.show()
    plt.plot(xf, np.abs(fft(EMG_15.values)))
    plt.plot(xf2, np.abs(fft(down_sampling.values)))
    plt.show()


def main():
    Zad1()
    # Zad2()
    # Zad2(2)
    # Zad3()
    # Zad4()

if __name__ == '__main__':
    main()