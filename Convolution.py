from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
pi = np.pi

N = 8
x = np.arange(N)*2*pi/N
k = np.arange(-N//2, N//2)
# print k
u = np.random.rand(N)
# u = np.cos(x) + np.sin(x+0.5)
v = np.sin(x) + np.cos(x+0.2)
w = u*v

uk = np.fft.fft(u)/N
vk = np.fft.fft(v)/N
wk = np.fft.fft(w)/N
shift = np.zeros(N//2, complex)
shift = np.copy(uk[:N//2])
uk[:N//2] = np.copy(uk[N//2:])
uk[N//2:] = np.copy(shift)
shift = np.copy(vk[:N//2])
vk[:N//2] = np.copy(vk[N//2:])
vk[N//2:] = np.copy(shift)
shift = np.copy(wk[:N//2])
wk[:N//2] = np.copy(wk[N//2:])
wk[N//2:] = np.copy(shift)

wk2 = np.zeros(N, complex)
for i in k:
    for m in k:
        n = i - m
        if n >= -N//2 and n <= N//2-1:
            wk2[i+N//2] = wk2[i+N//2] + uk[m+N//2] * vk[n+N//2]
            print i, m, n
        n = i + N - m
        if n >= -N//2 and n <= N//2-1:
            wk2[i+N//2] = wk2[i+N//2] + uk[m+N//2] * vk[n+N//2]
            print i, m, n
        n = i - N - m
        if n >= -N//2 and n <= N//2-1:
            wk2[i+N//2] = wk2[i+N//2] + uk[m+N//2] * vk[n+N//2]
            print i, m, n

print wk
print wk2

# plt.plot(x,vk.real)
# plt.show()
