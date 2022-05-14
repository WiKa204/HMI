import matplotlib
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft

matplotlib.rcParams['figure.figsize'] = (8, 4)

wav = np.load('./data/distorted.npy')
data = np.load('./data/distorted.npy')
plt.plot(data)
plt.title('Sygnał bazowy surowy')
plt.show()
# wav = wav / np.max(np.abs(wav))
fs = 48000
t = np.arange(len(wav)) / fs
dt = 1.0/fs
# t = np.arange(0, len(wav), dt)
plt.plot(t, wav)
plt.title('Sygnał bazowy')
plt.show()

# TODO 1 Dla sygnału  fs=48kHz , zaprojektuj 4 filtry o długości  N=101  próbek z oknem Hamminga:
#  górno przepustowy ( fp=3kHz ), dolno-przepustowy ( fp=3kHz ), pasmowo przepustowy ( 1−3kHz ),
#  pasmowo zaporowy ( 1−3kHz ). Wyświetl charakterystyki fazowe oraz odpowiedzi impulsowe.
def zad1():
    N = 101
    fc = 1000
    fp = 3000
    xf = fftfreq(len(wav), 1/fs)
    plt.plot(xf, np.abs(wav))
    plt.title('Transformata Fouriera sygnału bazowego')
    plt.show()
    y = fHG(wav)  \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\'\
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



def main():
    zad1()

if __name__ == '__main__':
    main()