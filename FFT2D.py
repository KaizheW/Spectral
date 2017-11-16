from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
pi = np.pi

N = 64
x = np.arange(N)*2.0*pi/N
k = np.arange(-N//2, N//2)

U = np.random.rand(N,N)
Uk = np.roll(np.roll(np.fft.fft2(U), N//2, axis=0), N//2, axis=1)/N**2
# print U
# plt.imshow(Uk.real)
# plt.colorbar()
# plt.axis('equal')
# plt.show()
U = np.fft.ifft2(np.roll(np.roll(Uk, N//2, axis=1), N//2, axis=0))*N**2
