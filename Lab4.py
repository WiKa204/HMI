import pylab as py
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft

def sin(T = 1, fs = 500):
    delta_f = 1/T
    dt = 1.0/fs
    t = np.arange(0,T,dt)
    s1 = np.sin(2*np.pi*10*t)
    s2 = np.sin(2*np.pi*45*t)
    s3 = np.sin(2*np.pi*50*t)
    s = s1 + 0.5*2 + 2*s3
    return (s,t)

T_2 = 1.11

s1, t1 = sin()
s2, t2 = sin(T=T_2)
y_1 = fft(s1)
y_2 = fft(s2)

N1=int(500)
N2=int(500*T_2)

X1 = fftfreq(N1, 1/500)
X2 = fftfreq(N2, 1/500)

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(t1, s1)
axs[0, 0].set_title("T=1")
plt.grid()
axs[1, 0].plot(t2, s2)
axs[1, 0].set_title("T=1.11")
plt.grid()
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].stem(np.abs(y_1))
axs[0, 1].set_title("T=1")
plt.grid()
axs[1, 1].stem(np.abs(y_2))
axs[1, 1].set_title("T=1.11")
plt.grid()
fig.tight_layout()
plt.show()

filtr = np.ones(500)
filtr[50] = 0
filtr[450] = 0
fir = np.fft.ifft(filtr)

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(t1, s1)
axs[0, 0].set_title("T=1")
plt.grid()
axs[1, 0].plot(t1, fir)
axs[1, 0].set_title("odpowiedź impulsowa fitra")
plt.grid()
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].stem(np.abs(y_1))
axs[0, 1].set_title("T=1")
plt.grid()
axs[1, 1].stem(np.abs(filtr))
axs[1, 1].set_title("Widmo filtru")
plt.grid()
fig.tight_layout()
plt.show()