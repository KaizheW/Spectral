from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import Convolution as conv
pi = np.pi

def f(uk, N):
    k = np.arange(-N//2, N//2)
    vk = k*uk*1.0j
    wk = conv.convolution1(uk,vk,N)
    return -wk

N = 64
x = np.arange(N)*2.0*pi/N
x0 = pi
sigma = 0.5
u = np.exp(-(x-x0)*(x-x0) / (2*sigma*sigma))
# v = (np.roll(u, -1) - np.roll(u, 1))/(4*pi/N)
k = np.arange(-N//2, N//2)
h = 0.01

uk = np.roll(np.fft.fft(u), N//2)
# vk = np.roll(np.fft.fft(v), N//2)

for t in range(50):
	k1 = h*f(uk, N)
	k2 = h*f(uk + 0.5*k1, N)
	k3 = h*f(uk + 0.5*k2, N)
	k4 = h*f(uk + k3, N)
	uk = uk + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
	
u = np.fft.ifft(np.roll(uk, N//2))
plt.plot(x, u.real)
plt.show()


