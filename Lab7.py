import numpy as np
import scipy.signal as signal
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft


fc = 3000
fz = 4500
rp = 1
rs = 60
N = 8

def sin(f=50, T=20, Fs=500, phi=0):
    dt = 1.0 / Fs
    t = np.arange(0, T, dt)
    s = np.sin(2 * np.pi * f * t + phi)
    return (s, t)

def cos(f=50, T=20, Fs=500, phi=0):
    dt = 1.0 / Fs
    t = np.arange(0, T, dt)
    s = np.cos(2 * np.pi * f * t + phi)
    return (s, t)

def plot_response(w, hf, hfb=None, label=''):
    fig, ax = plt.subplots(tight_layout=True)
    ax.axvline(3000, lw=1, ls=(0, (10, 5)), c='k')
    if hfb is not None:
        ax.plot(w, hfb, c='#c0c0c0', label='Butterworth')
    ax.plot(w, hf, label=label)
    ax.grid()
    ax.legend()
    ax.set_xlim(0, 12000)
    ax.set_ylim(-120, 5)
    ax.set_xlabel('Częstotliwość [Hz]')
    ax.set_ylabel('Wzmocnienie [dB]')
    # ax.set_title('Charakterystyka częstotliwościowa filtru IIR')
    return ax

# filtr Butterwortha
fp = 3000 # fpass [Hz]
fz = 4500 # fstop [Hz]
Rp = 1 # -1dB w paśmie przepustowym
Rs = 60 # -60db w paśmie zaporowym
fs = 48000 # częstotliwość próbkowania


N, fn = signal.buttord(fp, fz, Rp, Rs, fs=fs)
bb, ab = signal.butter(N, fn, 'low', fs=fs)
wb, hb = signal.freqz(bb, ab, 2048, fs=fs)
hbd = 20 * np.log10(np.abs(hb))

# filtr Czebyszewa I
N, fn = signal.cheb1ord(fp, fz, Rp, Rs, fs=fs)
bc1, ac1 = signal.cheby1(N, Rp, fn, 'low', fs=fs)
wc1, hc1 = signal.freqz(bc1, ac1, 2048, fs=fs)
hc1d = 20 * np.log10(np.abs(hc1))

# filtr Czebyszewa II
N, fn = signal.cheb2ord(fp, fz, Rp, Rs, fs=fs)
bc2, ac2 = signal.cheby2(N, Rs, fn, 'low', fs=fs)
wc2, hc2 = signal.freqz(bc2, ac2, 2048, fs=fs)
hc2d = 20 * np.log10(np.abs(hc2))


# filtr Eliptyczny
N, fn = signal.ellipord(fp, fz, Rp, Rs, fs=fs)
be, ae = signal.ellip(N, Rp,Rs, fn, 'low', fs=fs)
we, he = signal.freqz(be, ae, 2048, fs=fs)
hed = 20 * np.log10(np.abs(he))


# wszystko razem
ax = plot_response(wb, hbd, None, 'Butterworth')
ax.plot(wc1, hc1d, label='Czebyszew I')
ax.plot(wc2, hc2d, label='Czebyszew II')
ax.plot(we, hed, label='Eliptyczny')
ax.legend()

# charakterystyka fazowa
fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(8, 6))
for a in ax.ravel():
    a.axvline(3000, lw=1, ls=(0, (10, 5)), c='k')
    a.grid()
    a.set_xlim(0, 8000)
    a.set_ylim(-180, 180)
ax[0][0].plot(wb, np.degrees(np.angle(hb)))
ax[0][0].set_title('Butterworth')
ax[0][1].plot(wc1, np.degrees(np.angle(hc1)))
ax[0][1].set_title('Czebyszew I')
ax[1][0].plot(wc2, np.degrees(np.angle(hc2)))
ax[1][0].set_title('Czebyszew II')
ax[1][1].plot(we, np.degrees(np.angle(he)))
ax[1][1].set_title('Eliptyczny')
for a in ax[1]:
    a.set_xlabel('Częstotliwość [Hz]')
for a in (ax[0][0], ax[1][0]):
    a.set_ylabel('Faza [°]')

# opóźnienie grupowe
fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(8, 6))
for a in ax.ravel():
    a.axvline(3000, lw=1, ls=(0, (10, 5)), c='k')
    a.grid()
    a.set_xlim(0, 8000)
ax[0][0].plot(*signal.group_delay((bb, ab), 2048, fs=fs))
ax[0][0].set_title('Butterworth')
ax[0][1].plot(*signal.group_delay((bc1, ac1), 2048, fs=fs))
ax[0][1].set_title('Czebyszew I')
ax[1][0].plot(*signal.group_delay((bc2, ac2), 2048, fs=fs))
ax[1][0].set_title('Czebyszew II')
ax[1][1].plot(*signal.group_delay((be, ae), 2048, fs=fs))
ax[1][1].set_title('Eliptyczny')
for a in ax[1]:
    a.set_xlabel('Częstotliwość [Hz]')
for a in (ax[0][0], ax[1][0]):
    a.set_ylabel('Opóźnienie [próbki]')
plt.show()

### filtracja sygnału
t = np.arange(500)
x1 = np.sin(2 * np.pi * t * 500 / fs)
x2 = np.sin(2 * np.pi * t * 2500 / fs) * 0.5
x = x1 + x2
# sygnały po filtracji
y1 = signal.lfilter(be, ae, x1)
y2 = signal.lfilter(be, ae, x2)
y = signal.lfilter(be, ae, x)

fig, ax = plt.subplots(2, figsize=(8, 6), tight_layout=True)
ax[0].plot(x1)
ax[0].plot(y1)
ax[1].plot(x2)
ax[1].plot(y2)
for aa in ax:
    aa.grid()
    aa.set_ylabel('Amplituda')
ax[1].set_xlabel('Nr próbki')
ax[0].set_title('Sygnał 500Hz przed i po filtracji')
ax[1].set_title('Sygnał 2500Hz przed i po filtracji')
plt.show()

def zad1():
    fs = 48000
    N = 4
    fp = 3000
    fz = 4500
    f2 = 1000
    Rp = 1  # -1dB w paśmie przepustowym
    Rs = 60  # -60db w paśmie zaporowym

    # filtr Czebyszewa I -- highpass
    N, fn = signal.cheb1ord(fp, fz, Rp, Rs, fs=fs)
    print(f'Długość wyznaczona HP: {N}')
    bc_highpass, ac_highpass = signal.cheby1(4, Rp, fn, 'highpass', fs=fs)
    wc_highpass, hc_highpass = signal.freqz(bc_highpass, ac_highpass, 2048, fs=fs)
    hcd_highpass = 20 * np.log10(np.abs(hc_highpass))

    # filtr Czebyszewa I -- lowpass
    N, fn = signal.cheb1ord(fp, fp + 200, Rp, Rs, fs=fs)
    print(f'Długość wyznaczona LP: {N}')
    bc_lowpass, ac_lowpass = signal.cheby1(4, Rp, fn, 'lowpass', fs=fs)
    wc_lowpass, hc_lowpass = signal.freqz(bc_lowpass, ac_lowpass, 2048, fs=fs)
    hcd_lowpass = 20 * np.log10(np.abs(hc_lowpass))

    # filtr Czebyszewa I -- bandpass
    N, fn = signal.cheb1ord([f2, fp], [f2 - 200, fp + 200], Rp, Rs, fs=fs)
    print(f'Długość wyznaczona BP: {N}')
    bc_bandpass, ac_bandpass = signal.cheby1(4, Rp, fn, 'bandpass', fs=fs)
    wc_bandpass, hc_bandpass = signal.freqz(bc_bandpass, ac_bandpass, 2048, fs=fs)
    hcd_bandpass = 20 * np.log10(np.abs(hc_bandpass))

    # filtr Czebyszewa I -- bandstop
    N, fn = signal.cheb1ord([f2, fp], [f2 - 200, fp + 200], Rp, Rs, fs=fs)
    print(f'Długość wyznaczona BP: {N}')
    bc_bandstop, ac_bandstop = signal.cheby1(4, Rp, fn, 'bandstop', fs=fs)
    wc_bandstop, hc_bandstop = signal.freqz(bc_bandstop, ac_bandstop, 2048, fs=fs)
    hcd_bandstop = 20 * np.log10(np.abs(hc_bandstop))

    # wszystko razem
    ax = plot_response(wc_highpass, hcd_highpass, None, 'Highpass')
    ax.plot(wc_lowpass, hcd_lowpass, label='Lowpass')
    ax.plot(wc_bandpass, hcd_bandpass, label='Bandpass')
    ax.plot(wc_bandstop, hcd_bandstop, label='Bandstop')
    ax.legend()
    ax.set_title('Charakterystyki częstotliwościowe')
    plt.plot()

    plt.figure()
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(bc_highpass)
    axs[0, 0].set_title("HP")
    plt.grid()
    axs[1, 0].plot(bc_lowpass)
    axs[1, 0].set_title("LP")
    plt.grid()
    axs[0, 1].plot(bc_bandpass)
    axs[0, 1].set_title("BP")
    plt.grid()
    axs[1, 1].plot(bc_bandstop)
    axs[1, 1].set_title("BS")
    plt.grid()
    fig.tight_layout()
    plt.show()

def zad2():
    fs = 500
    s1, t_s1 = sin(f=3)
    s2, t_s2 = sin(f=35)
    s3, t_s3 = sin(f=50)
    s4, t_s4 = sin(f=100)

    c1, t_c1 = cos(f=10)
    c2, t_c2 = cos(f=25)
    x = s1 + s2 + s3 + s4 + c1 + c2
    plt.plot(t_c2, x)
    plt.title('Sygnał testowy')
    plt.show()
    y = fft(x)
    xf = fftfreq(len(x), 1 / fs)
    plt.plot(xf, np.abs(y))
    plt.title('Transformata sygnału testowego')
    plt.show()
    t = t_s1
    return (x, t)

def zad3(data: np.ndarray, time: np.ndarray):
    fp = 45  # fpass [Hz]
    fz = 450  # fstop [Hz]
    Rp = 60  # 60 dB w paśmie przepustowym
    Rs = 65  # -65db w paśmie zaporowym
    fs = 500  # częstotliwość próbkowania

    # filtr Czebyszewa II
    N, fn = signal.cheb2ord(wp=fp, ws=fz, gpass=Rp, gstop=Rs, fs=fs)
    bc2, ac2 = signal.cheby2(N, Rs, fn, 'low', fs=fs)
    wc2, hc2 = signal.freqz(bc2, ac2, 2048, fs=fs)
    hc2d = 20 * np.log10(np.abs(hc2))

    plt.plot(wc2, hc2d)
    plt.title('Filtr Czebyszewa II LP')
    plt.show()

    # Zastosowanie filtru
    y = signal.lfilter(bc2, ac2, data)
    plt.plot(y, label='po')
    plt.plot(data, label='przed')
    plt.title('Sygnał przed i po filtracji')
    plt.legend()
    plt.show()

    y_fft = fft(y)
    xf = fftfreq(len(y_fft), 1 / fs)
    plt.plot(xf, np.abs(y_fft))
    plt.title('Transformata sygnału przefiltrowanego')
    plt.show()

    ### filtracja sygnału
    t = np.arange(500)
    s1 = np.sin(2 * np.pi * t * 3 / fs)
    s2 = np.sin(2 * np.pi * t * 35 / fs)
    c1 = np.cos(2 * np.pi * t * 10 / fs)
    c2 = np.cos(2 * np.pi * t * 25 / fs)
    x = s1 + s2 + c1 + c2
    # sygnały po filtracji
    y1 = signal.lfilter(bc2, ac2, s1)
    y2 = signal.lfilter(bc2, ac2, s2)
    y3 = signal.lfilter(bc2, ac2, c1)
    y4 = signal.lfilter(bc2, ac2, c2)
    y = signal.lfilter(bc2, ac2, x)

    fig, ax = plt.subplots(5, figsize=(8, 6), tight_layout=True)
    ax[0].plot(s1)
    ax[0].plot(y1)
    ax[1].plot(s2)
    ax[1].plot(y2)
    ax[2].plot(c1)
    ax[2].plot(y3)
    ax[3].plot(c2)
    ax[3].plot(y4)
    ax[4].plot(x)
    ax[4].plot(y)
    for aa in ax:
        aa.grid()
        aa.set_ylabel('Amplituda')
    ax[1].set_xlabel('Nr próbki')
    ax[0].set_title('Sygnał 3Hz przed i po filtracji')
    ax[1].set_title('Sygnał 10Hz przed i po filtracji')
    ax[2].set_title('Sygnał 25Hz przed i po filtracji')
    ax[3].set_title('Sygnał 35Hz przed i po filtracji')
    ax[4].set_title('Sygnał idealny i sygnał po filtracji')
    plt.show()

def main():
   # zad1()
    sygnal, t = zad2()
    zad3(data=sygnal, time=t)

if __name__=='__main__':
    main()