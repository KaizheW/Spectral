from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
pi = np.pi

N = 128
nu = 0.5
x, y = np.meshgrid(np.arange(0, 2*pi, 2*pi/N), np.arange(0, 2*pi, 2*pi/N))
kx, ky = np.meshgrid(np.arange(-N//2, N//2), np.arange(-N//2, N//2))
dt = 0.01
t = 0.0
finalt = 500
# w = -np.exp(-((x-pi/5)**2+(y-4*pi/5)**2)/0.03)\
# +np.exp(-((x-pi/5)**2+(y-6*pi/5)**2)/0.03)
# +np.exp(-((x-6*pi/5)**2+(y-6*pi/5)**2)/0.4)
w = np.zeros([N,N])
w[N//2,:] = 1.0
wk = np.roll(np.roll(np.fft.fft2(w), N//2, axis=0), N//2, axis=1)
Lapk = kx**2+ky**2
Lapk[N//2, N//2] = 1.0

for t in range(finalt):
    psik = wk/Lapk
    u = np.fft.ifft2(np.roll(np.roll( 1j*ky*psik, N//2, axis=1), N//2, axis=0)).real
    v = np.fft.ifft2(np.roll(np.roll(-1j*kx*psik, N//2, axis=1), N//2, axis=0)).real
    w_x = np.fft.ifft2(np.roll(np.roll(1j*kx*wk, N//2, axis=1), N//2, axis=0)).real
    w_y = np.fft.ifft2(np.roll(np.roll(1j*ky*wk, N//2, axis=1), N//2, axis=0)).real
    ugradw = u*w_x + v*w_y
    ugradwk = np.roll(np.roll(np.fft.fft2(ugradw), N//2, axis=0), N//2, axis=1)
    wk = 1.0/(1.0/dt+0.5*nu*Lapk)*((1/dt-0.5*nu*Lapk)*wk-ugradwk)

w = np.fft.ifft2(np.roll(np.roll(wk, N//2, axis=1), N//2, axis=0)).real
plt.quiver(x, y, u, v, units='width')
plt.imshow(w,origin='lower',extent=([0, 2*pi, 0, 2*pi]))
plt.colorbar()
plt.show()
