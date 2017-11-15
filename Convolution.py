from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
pi = np.pi

def convolution1(uk,vk,N):
    wk = np.zeros(N, complex)
    k = np.arange(-N//2, N//2)
    for i in k:
        for m in k:
            n = i - m
            if n >= -N//2 and n <= N//2-1:
                wk[i+N//2] = wk[i+N//2] + uk[m+N//2] * vk[n+N//2]/N
            n = i + N - m
            if n >= -N//2 and n <= N//2-1:
                wk[i+N//2] = wk[i+N//2] + uk[m+N//2] * vk[n+N//2]/N
            n = i - N - m
            if n >= -N//2 and n <= N//2-1:
                wk[i+N//2] = wk[i+N//2] + uk[m+N//2] * vk[n+N//2]/N
    return wk

# plt.plot(x,vk.real)
# plt.show()
