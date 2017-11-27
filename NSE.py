from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
pi = np.pi

class NSE:

    def __init__(self, N, tfinal, cfl):
        self.N = N
        self.cfl = cfl
        self.t = 0.0
        self.tfinal = tfinal
        self.x, self.y = np.meshgrid(np.arange(0, 2*pi, 2*pi/N), np.arange(0, 2*pi, 2*pi/N))
        self.kx, self.ky = np.meshgrid(np.arange(-N//2, N//2), np.arange(-N//2, N//2))
        self.dx = 2*pi/N
        self.dy = 2*pi/N
        self.ip = np.arange(N)+1
        self.ip[N-1] = 0
        # self.jp = np.arange(N)+1
        # self.jp[N-1] = 0
        self.im = np.arange(N)-1
        self.im[0] = N-1
        # self.jm = np.arange(N)-1
        # self.jm[0] = N-1
        self.ic = np.arange(N)
        # self.jc = np.arange(N)

    def InitialKH(self):
        self.Re = 1000.0
        self.Pe = 1000.0
        U0 = 1.0
        Pj = 20.0
        Rj = pi/2.0
        Ax = 0.5
        Lambda = pi
        v = np.zeros([self.N, self.N])
        u1 = (np.ones([self.N, self.N]) + np.tanh((np.ones([self.N, self.N]) \
        - abs(pi*np.ones([self.N, self.N])-self.y)/Rj)*Pj/2.0))*U0/2.0
        u2 = Ax * np.sin(2*pi*self.x/Lambda)
        u = u1*(np.ones([self.N, self.N])+u2)
        p = np.zeros([self.N, self.N])
        return u, v, p

    def Heatf(self):
        return 2.0*np.sin(self.x)*np.cos(self.y)

    def Convectionf(self):
        return (2.0*np.cos(self.x)*np.cos(self.y)+2.0)*np.sin(self.x)*np.cos(self.y)

    def Getdt(self):
        return (self.cfl/2.0) / (1.0/(self.dx**2)+1.0/(self.dy**2))

    def GetnNSdt(self, u, v):
        return self.cfl/np.max(abs(u/self.dx)+abs(v/self.dy))

    def du2dx(self, u):
        return (((u + u[:, self.ip])/2)**2-((u + u[:, self.im])/2)**2)/self.dx

    def duvdy(self, u, v):
        vpm = np.copy(v[:, self.im])
        vpm = np.copy(vpm[self.ip, :])
        return (((u+u[self.ip,:])/2.0)*((v[self.ip,:]+vpm)/2.0) \
        - ((u+u[self.im,:])/2.0)*((v+v[:,self.im])/2.0))/self.dy

    def dv2dy(self, v):
        return (((v + v[self.ip, :])/2)**2-((v + v[self.im, :])/2)**2)/self.dy

    def duvdx(self, u, v):
        upm = np.copy(u[:, self.ip])
        upm = np.copy(upm[self.im, :])
        return (((v+v[:,self.ip])/2.0)*((u[:,self.ip]+upm)/2.0) \
        - ((v+v[:,self.im])/2.0)*((u+u[self.im,:])/2.0))/self.dx

    def Laplace(self, phi):
        return (phi[:,self.ip] - 2*phi + phi[:,self.im])/self.dx**2 + \
        (phi[self.ip,:] - 2*phi + phi[self.im,:])/self.dy**2

    def Hu(self, u, v):
        return - self.du2dx(u) - self.duvdy(u,v)

    def Hv(self, u, v):
        return - self.duvdx(u,v) - self.dv2dy(v)

    def Vorticity(self, u, v):
        w = (v - v[:, self.im])/self.dx - (u - u[self.im, :])/self.dy
        return w

    def LaplaceFFT(self):
        Lk = 2*(np.cos(2*pi*self.kx/self.N)+np.cos(2*pi*self.ky/self.N)-2)/(2*pi/self.N)**2
        Lk[self.N//2, self.N//2] = 1.0
        return Lk

    def dxFFT(self):
        dxk = 1j*np.sin(2*pi*self.kx/self.N)/(2*pi/self.N)
        dxk[self.N//2, self.N//2] = 1.0
        return dxk

    def dyFFT(self):
        dyk = 1j*np.sin(2*pi*self.ky/self.N)/(2*pi/self.N)
        dyk[self.N//2, self.N//2] = 1.0
        return dyk

    def HeatExplicit(self):
        dt = self.Getdt()
        u = np.zeros([self.N,self.N])
        f = self.Heatf()
        for ti in range(50):
            lu = (u[self.ip,:] - 2*u + u[self.im,:])/(self.dx**2)\
            + (u[:,self.ip] - 2*u + u[:,self.im])/(self.dy**2)
            u = u + dt*(f + lu)
        plt.imshow(u)
        plt.show()

    def HeatImplicit(self):
        N = self.N
        dt = self.Getdt()
        u = np.zeros([N, N],float)
        f = self.Heatf()
        fk = np.roll(np.roll(np.fft.fft2(f), N//2, axis=0), N//2, axis=1)
        fk[N//2,N//2] = 0.0
        Lk = self.LaplaceFFT()
        for ti in range(1000):
            uk = np.roll(np.roll(np.fft.fft2(u), N//2, axis=0), N//2, axis=1)
            B = dt*(fk + Lk*uk)
            A = np.ones([N,N]) - Lk*dt/2.0
            duk = B/A
            du = np.fft.ifft2(np.roll(np.roll(duk, N//2, axis=0), N//2, axis=1))
            u = u + du
        plt.imshow(u.real, extent = [0.0, 2*pi, 0.0, 2*pi])
        plt.show()

    def ConvectionImplicit(self):
        N = self.N
        dt = self.Getdt()
        u = np.zeros([N,N], float)
        f = self.Convectionf()
        Hn = f - self.du2dx(u)
        Hn1 = f - self.du2dx(u)
        Hn1k = np.roll(np.roll(np.fft.fft2(Hn1), N//2, axis=0), N//2, axis=1)
        Lk = self.LaplaceFFT()
        for ti in range(1000):
            uk = np.roll(np.roll(np.fft.fft2(u), N//2, axis=0), N//2, axis=1)
            Hnk = np.roll(np.roll(np.fft.fft2(Hn), N//2, axis=0), N//2, axis=1)
            B = dt*(1.5*Hnk - 0.5*Hn1k + Lk*uk)
            A = np.ones([N,N]) - Lk*dt/2.0
            duk = B/A
            du = np.fft.ifft2(np.roll(np.roll(duk, N//2, axis=0), N//2, axis=1))
            u = u + du
            Hn1 = np.copy(Hn)
            Hn1k = np.copy(Hnk)
            Hn = f - self.du2dx(u)
        plt.imshow(u.real, extent = [0.0, 2*pi, 0.0, 2*pi])
        plt.show()

    def HeatPoisson(self):
        N = self.N
        f = self.Heatf()
        Lk = self.LaplaceFFT()
        fk = np.roll(np.roll(np.fft.fft2(f), N//2, axis=0), N//2, axis=1)
        fk[N//2, N//2] = 0.0
        uk = -fk/Lk
        u = np.fft.ifft2(np.roll(np.roll(uk, N//2, axis=0), N//2, axis=1))
        plt.imshow(u.real, extent = [0.0, 2*pi, 0.0, 2*pi])
        plt.show()

    def NSEMAIN(self,u,v,p):
        N = self.N
        Hnu = self.Hu(u,v)
        Hnv = self.Hv(u,v)
        Hnu1 = self.Hu(u,v)
        Hnv1 = self.Hv(u,v)
        Hnu1k = np.roll(np.roll(np.fft.fft2(Hnu1), N//2, axis=0), N//2, axis=1)
        Hnv1k = np.roll(np.roll(np.fft.fft2(Hnv1), N//2, axis=0), N//2, axis=1)
        Lk = self.LaplaceFFT()
        while self.t < self.tfinal:
            dt = self.GetnNSdt(u,v)
            if self.t + dt > self.tfinal:
                dt = self.tfinal - self.t
            px = (p[:,self.ip] - p[:,self.im])/(2*self.dx)
            py = (p[self.ip,:] - p[self.im,:])/(2*self.dy)
            pxk = np.roll(np.roll(np.fft.fft2(px), N//2, axis=0), N//2, axis=1)
            pyk = np.roll(np.roll(np.fft.fft2(py), N//2, axis=0), N//2, axis=1)
            Hnuk = np.roll(np.roll(np.fft.fft2(Hnu), N//2, axis=0), N//2, axis=1)
            Hnvk = np.roll(np.roll(np.fft.fft2(Hnv), N//2, axis=0), N//2, axis=1)
            Lu = self.Laplace(u)
            Lv = self.Laplace(v)
            Luk = np.roll(np.roll(np.fft.fft2(Lu), N//2, axis=0), N//2, axis=1)
            Lvk = np.roll(np.roll(np.fft.fft2(Lv), N//2, axis=0), N//2, axis=1)
            Bu = dt*(- pxk + 1.5*Hnuk - 0.5*Hnu1k + Luk/self.Re)
            Bv = dt*(- pyk + 1.5*Hnvk - 0.5*Hnv1k + Lvk/self.Re)
            Bu[N//2, N//2] = 0.0
            Bv[N//2, N//2] = 0.0
            A = np.ones([N,N]) - dt*Lk/(2.0*self.Re)
            duk = Bu/A
            dvk = Bv/A
            du = np.fft.ifft2(np.roll(np.roll(duk, N//2, axis=0), N//2, axis=1)).real # Real
            dv = np.fft.ifft2(np.roll(np.roll(dvk, N//2, axis=0), N//2, axis=1)).real # Real
            us = u + du
            vs = v + dv
            # usx = (us[:,self.ip] - us[:,self.im])/(2*self.dx)
            usx = (us[:,self.ip] - us)/(self.dx)
            # vsy = (vs[self.ip,:] - vs[self.im,:])/(2*self.dy)
            vsy = (vs[self.ip,:] - vs)/(self.dy)
            Qk = np.roll(np.roll(np.fft.fft2((usx+vsy)/dt), N//2, axis=0), N//2, axis=1)
            Qk[N//2, N//2] = 0.0
            phik = Qk/Lk
            phi = np.fft.ifft2(np.roll(np.roll(phik, N//2, axis=0), N//2, axis=1)).real # Real
            # u = us - dt*(phi[:, self.ip] - phi[:, self.im])/(2.0*self.dx)
            u = us - dt*(phi - phi[:, self.im])/(self.dx)
            # v = vs - dt*(phi[:, self.ip] - phi[self.im, :])/(2.0*self.dy)
            v = vs - dt*(phi - phi[self.im, :])/(self.dy)
            p = p + phi - 0.5*dt*self.Laplace(phi)/self.Re
            Hnu1 = np.copy(Hnu)
            Hnv1 = np.copy(Hnv)
            Hnu1k = np.copy(Hnuk)
            Hnv1k = np.copy(Hnvk)
            Hnu = self.Hu(u,v)
            Hnv = self.Hv(u,v)
            self.t = self.t + dt
            print self.t
        return u,v
    
    def NSEMAIN2(self, u, v, p):
        N = self.N
        Hnu = self.Hu(u,v)
        Hnv = self.Hv(u,v)
        Hnu1 = self.Hu(u,v)
        Hnv1 = self.Hv(u,v)
        Hnu1k = np.roll(np.roll(np.fft.fft2(Hnu1), N//2, axis=0), N//2, axis=1)
        Hnv1k = np.roll(np.roll(np.fft.fft2(Hnv1), N//2, axis=0), N//2, axis=1)
        Lk = self.LaplaceFFT()
        Dxk = self.dxFFT()
        Dyk = self.dyFFT()
        while self.t < self.tfinal:
            dt = self.GetnNSdt(u,v)
            if self.t + dt > self.tfinal:
                dt = self.tfinal - self.t
            pk = np.roll(np.roll(np.fft.fft2(p), N//2, axis=0), N//2, axis=1)
            Hnuk = np.roll(np.roll(np.fft.fft2(Hnu), N//2, axis=0), N//2, axis=1)
            Hnvk = np.roll(np.roll(np.fft.fft2(Hnv), N//2, axis=0), N//2, axis=1)
            uk = np.roll(np.roll(np.fft.fft2(u), N//2, axis=0), N//2, axis=1)
            vk = np.roll(np.roll(np.fft.fft2(v), N//2, axis=0), N//2, axis=1)
            Bu = dt*(- Dxk*pk + 1.5*Hnuk - 0.5*Hnu1k + Lk*uk/self.Re)
            Bv = dt*(- Dyk*pk + 1.5*Hnvk - 0.5*Hnv1k + Lk*vk/self.Re)
            Bu[N//2, N//2] = 0.0
            Bv[N//2, N//2] = 0.0
            A = np.ones([N,N]) - dt*Lk/(2.0*self.Re)
            duk = Bu/A
            dvk = Bv/A
            du = np.fft.ifft2(np.roll(np.roll(duk, N//2, axis=0), N//2, axis=1)).real # Real
            dv = np.fft.ifft2(np.roll(np.roll(dvk, N//2, axis=0), N//2, axis=1)).real # Real
            us = u + du
            vs = v + dv
            usk = np.roll(np.roll(np.fft.fft2(us), N//2, axis=0), N//2, axis=1)
            vsk = np.roll(np.roll(np.fft.fft2(vs), N//2, axis=0), N//2, axis=1)
            Qk = (Dxk*usk+Dyk*vsk)/dt
            Qk[N//2, N//2] = 0.0
            phik = Qk/Lk
            phi = np.fft.ifft2(np.roll(np.roll(phik, N//2, axis=0), N//2, axis=1)).real # Real
            # u = us - dt*(phi[:, self.ip] - phi[:, self.im])/(2.0*self.dx)
            u = us - dt*(phi - phi[:, self.im])/(self.dx)
            # v = vs - dt*(phi[:, self.ip] - phi[self.im, :])/(2.0*self.dy)
            v = vs - dt*(phi - phi[self.im, :])/(self.dy)
            p = p + phi - 0.5*dt*self.Laplace(phi)/self.Re
            Hnu1 = np.copy(Hnu)
            Hnv1 = np.copy(Hnv)
            Hnu1k = np.copy(Hnuk)
            Hnv1k = np.copy(Hnvk)
            Hnu = self.Hu(u,v)
            Hnv = self.Hv(u,v)
            self.t = self.t + dt
            print self.t
        return u,v


solver = NSE(256,0.8,0.5)
u, v, p = solver.InitialKH()
u, v = solver.NSEMAIN(u,v,p)
# print u,v
w = solver.Vorticity(u, v)
plt.imshow(w.real)
plt.show()
