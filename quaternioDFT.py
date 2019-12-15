import os
import numpy as np 
import matplotlib.pyplot as plt
import cv2

#sum(x[n]exp(-i*2pi*k*n/N),n,0,N-1)
#sum(x[n,m]exp(-i*2pi*k*n/N)exp(-j*2pi*l*n/N))

def QFT(A0,Ai,Aj,Ak):
    ha=A0+(1j)*Ai
    hb=Aj+(1j)*Ak
    H1=np.fft.fft2(ha)
    H1=np.fft.fftshift(H1)
    H2=np.fft.fft2(np.fliplr(hb))#j
    H2=np.fft.fftshift(H2)
    Hc0=np.real(H1)
    Hci=np.imag(H1)
    Hcj=np.real(H2)
    Hck=np.imag(H2)
    T0=(Hc0+np.fliplr(Hc0))/2+(Hck-np.fliplr(Hck))/2
    Ti=(Hci+np.fliplr(Hci))/2-(Hcj-np.fliplr(Hcj))/2
    Tj=(Hcj+np.fliplr(Hcj))/2+(Hci-np.fliplr(Hci))/2
    Tk=(Hck+np.fliplr(Hck))/2-(Hc0-np.fliplr(Hc0))/2
    
    return T0,Ti,Tj,Tk

def iQFT(A0,Ai,Aj,Ak):
    
    ha=A0+(1j)*Ai
    hb=Aj+(1j)*Ak
    H1=np.fft.ifft2(ha)
    H1=np.fft.fftshift(H1)
    H2=np.fft.ifft2(np.flip(hb,axis=1))#j
    H2=np.fft.fftshift(H2)
    Hc0=np.real(H1)
    Hci=np.imag(H1)
    Hcj=np.real(H2)
    Hck=np.imag(H2)
    T0=(Hc0+np.flip(Hc0,axis=1))/2+(Hck-np.flip(Hck,axis=1))/2
    Ti=(Hci+np.flip(Hci,axis=1))/2-(Hcj-np.flip(Hcj,axis=1))/2
    Tj=(Hcj+np.flip(Hcj,axis=1))/2+(Hci-np.flip(Hci,axis=1))/2
    Tk=(Hck+np.flip(Hck,axis=1))/2-(Hc0-np.flip(Hc0,axis=1))/2
    T0=np.fft.fftshift(T0)
    Ti=np.fft.fftshift(Ti)
    Tj=np.fft.fftshift(Tj)
    Tk=np.fft.fftshift(Tk)
    return T0,Ti,Tj,Tk

def FT(A0,Ai):
    ha=A0+(1j)*Ai
    H1=np.fft.fft2(ha)
    T0=np.real(H1)
    Ti=np.imag(H1)
    return T0,Ti

def filters(x,lf):
    rows,cols=x.shape
    X=0*(x+(1j)*x)
    
    for i in range(rows):
        X[i,:]=np.fft.fft(x[i,:])
        #X[i,:]=np.fft.fftshift(X[i,:])

    Xr=np.real(X)
    Xi=np.imag(X)

    Y=X
    Z=X

    for i in range(cols):
        Y[:,i]=np.fft.fft(Xr[:,i])
        #Y[:,i]=np.fft.fftshift(Y[:,i])
        Z[:,i]=np.fft.fft(Xi[:,i])
        #Z[:,i]=np.fft.fftshift(Z[:,i])

    Xr=np.real(Y)
    Xi=np.real(Z)
    Xj=np.imag(Y)
    Xk=np.imag(Z)

    crow=rows//2
    ccol=cols//2

   #Xr[crow-lf:crow+lf,ccol-lf:ccol+lf]=0
    #Xi[crow-lf:crow+lf,ccol-lf:ccol+lf]=0
    #Xj[crow-lf:crow+lf,ccol-lf:ccol+lf]=0
    #Xk[crow-lf:crow+lf,ccol-lf:ccol+lf]=0

    #for i in range(cols):
        #Xr[:,i]=np.fft.ifftshift(Xr[:,i])
        #Xi[:,i]=np.fft.ifftshift(Xr[:,i])
        #Xj[:,i]=np.fft.ifftshift(Xr[:,i])
        #Xk[:,i]=np.fft.ifftshift(Xr[:,i])

    #for i in range(rows):
        #Xr[i,:]=np.fft.ifftshift(Xr[i,:])
        #Xi[i,:]=np.fft.ifftshift(Xi[i,:])
        #Xj[i,:]=np.fft.ifftshift(Xj[i,:])
        #Xk[i,:]=np.fft.ifftshift(Xk[i,:])

    #exxp(ibla)*(Xr+i*Xi+j*Xj+k*Xk)*exp(j)

    for i in range(rows):
        Fr=np.fft.ifft(Xr[:,i])
        Fi=np.fft.ifft(Xi[:,i])
        Fj=np.fft.ifft(Xj[:,i])
        Fk=np.fft.ifft(Xk[:,i])
        Xr[i,:]=np.real(Fr)-np.imag(Fj)
        Xi[i,:]=np.real(Fi)-np.imag(Fk)
        Xj[i,:]=np.real(Fj)
        Xk[i,:]=np.real(Fk)+np.imag(Fi)

    for i in range(cols):  
        Fr=np.fft.ifft(Xr[:,i])
        Fi=np.fft.ifft(Xi[:,i])
        Fj=np.fft.ifft(Xj[:,i])
        Fk=np.fft.ifft(Xk[:,i])
        Xr[:,i]=np.real(Fr)-np.imag(Fi)
        Xi[:,i]=np.real(Fi)
        Xj[:,i]=np.real(Fj)-np.imag(Fk)
        Xk[:,i]=np.real(Fk)+np.imag(Fj)
    
    return Xr







    







