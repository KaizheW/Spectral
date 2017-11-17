from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
pi = np.pi

N = 1024
print N/(2*pi)
X, Y = np.meshgrid(np.arange(0, 2*pi, 2*pi/N), np.arange(0, 2*pi, 2*pi/N))
KX, KY = np.meshgrid(np.arange(-N//2, N//2), np.arange(-N//2, N//2))

U = np.zeros([N,N],complex)
V = np.zeros([N,N],complex)
U[N//2:3*N//4, N//4:N//2] = 1.0
V[N//2:3*N//4, N//4:N//2] = 1.0
Uk = np.roll(np.roll(np.fft.fft2(U), N//2, axis=0), N//2, axis=1)/N**2
Vk = np.roll(np.roll(np.fft.fft2(V), N//2, axis=0), N//2, axis=1)/N**2

dUk = 1j*Uk*KX
dVk = 1j*Vk*KY
dU = np.fft.ifft2(np.roll(np.roll(dUk, N//2, axis=1), N//2, axis=0))*N**2
dV = np.fft.ifft2(np.roll(np.roll(dVk, N//2, axis=1), N//2, axis=0))*N**2
# plt.figure()
# plt.title('fff')
# Q = plt.quiver(X, Y, U, V, units='width')
# plt.imshow(U)
# plt.colorbar()
# plt.show()

# print U
plt.imshow(dV.real)
plt.colorbar()
plt.axis('equal')
plt.show()
# U = np.fft.ifft2(np.roll(np.roll(Uk, N//2, axis=1), N//2, axis=0))*N**2


