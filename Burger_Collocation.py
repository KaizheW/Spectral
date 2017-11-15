from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
pi = np.pi

N = 64
x = np.arange(N)*2.0*pi/N
x0 = pi
sigma = 0.5
u = np.exp(-(x-x0)*(x-x0) / (2*sigma*sigma))
h = 0.01
# du = np.sin(2.0*x) # Theoritical differentiation

D = np.zeros([N,N], complex)
for l in range(N):
    for i in range(N):
        for k in range(int(-N/2), int(N/2-1)):
            D[l,i] = D[l,i] + 1j*k*np.exp(2j*k*(l-i)*pi/N)
        D[l,i] = D[l,i]/N

for t in range(50):
    Du1 = np.zeros(N, complex) # Calculated differentiation
    for l in range(N):
        for i in range(N):
            Du1[l] = Du1[l] + D[l,i] * u[i]
    k1 = -h*u*(Du1.real)
    Du2 = np.zeros(N,complex)
    for l in range(N):
        for i in range(N):
            Du2[l] = Du2[l] + D[l,i] * (u[i] + 0.5*k1[i])
    k2 = -h*(u+0.5*k1)*(Du2.real)
    Du3 = np.zeros(N, complex)
    for l in range(N):
        for i in range(N):
            Du3[l] = Du3[l] + D[l,i] * (u[i] + 0.5*k2[i])
    k3 = -h*(u+0.5*k2)*(Du3.real)
    Du4 = np.zeros(N, complex)
    for l in range(N):
        for i in range(N):
            Du4[l] = Du4[l] + D[l,i] * (u[i] + k3[i])
    k4 = -h*(u+k3)*(Du4.real)
    u = u + (k1 + 2*k2 + 2*k3 + k4)/6.0

fig = plt.figure()
# a = np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
plt.plot(x,u)
fig.savefig('Burger1.pdf')
# plt.plot(x, Du.real)
plt.show()
