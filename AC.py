import numpy as np 
import cv2
def AC(x):
    crow, ccol = x.shape
    Maskx=0*x
    Masky=0*x
    Maskx[:,:ccol]=-1
    Masky[:,:crow]=-1
    Mask=1+0.5*(Maskx+Masky)
    f=np.fft.fft2(x)
    fshift = np.fft.fftshift(f)
    rows, cols = x.shape
    crow,ccol = rows//2 , cols//2
    fshift=fshift*Mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    enveloppe= np.imag(img_back)
    dx=np.diff(x,axis=1)[:-1,:]
    I=dx==0
    Y= np.arctan(np.diff(x,axis=0)[:,:-1]/np.diff(x,axis=1)[:-1,:])
    Y[I]=np.pi/2
    changeO1=np.diff(x,axis=0)[:,:-1]*np.sin(Y)+np.diff(x,axis=1)[:-1,:]*np.cos(Y)
    changeN1=np.diff(enveloppe,axis=0)[:,:-1]*np.sin(Y)+np.diff(enveloppe,axis=1)[:-1,:]*np.cos(Y)
    return np.minimum(np.abs(changeO1-changeN1),np.abs(changeO1+changeN1))
    
    










    return enveloppe3