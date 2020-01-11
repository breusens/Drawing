import numpy as np 
import cv2
def fftwostripe(x,lf):
    crow, ccol = x.shape
    Maskx=0*x
    Masky=0*x
    Maskx[:,:ccol//2]=-1
    Masky[:,:crow//2]=-1
    Mask=0.5*(Maskx+Masky.T)
    f=np.fft.fft2(x)
    fshift = np.fft.fftshift(f)
    rows, cols = x.shape
    crow,ccol = rows//2 , cols//2
    fshift=fshift*(1+Mask)
    fshift[crow-lf:crow+lf, ccol-lf:ccol+lf] = 0
    f_ishift = np.fft.ifftshift(fshift)
    stripes=f_ishift.copy()
    stripes[:,1:]=0
    img_back = np.fft.ifft2(f_ishift)
    stripes= np.fft.ifft2(stripes)
    img_back1 = np.real(img_back)
    env1=np.imag(img_back)
    stripes=np.real(stripes)
    X=np.exp(img_back1)/(1+np.exp(img_back1))
    E=np.exp(env1)/(1+np.exp(env1))
    dx=np.diff(x,axis=1)[:-1,:]
    I=dx==0
    dy=np.diff(x,axis=0)[:,:-1]
    J=dy==0
    Y= np.arctan(np.diff(x,axis=0)[:,:-1]/np.diff(x,axis=1)[:-1,:])
    Y[I]=np.pi/2
    Y[np.logical_and(I,J)]=np.pi/4
    OIT1=Y
    changeO1=np.diff(env1,axis=0)[:,:-1]*np.sin(Y)+np.diff(env1,axis=1)[:-1,:]*np.cos(Y)
    changeN1=np.diff(img_back1,axis=0)[:,:-1]*np.sin(Y)+np.diff(img_back1,axis=1)[:-1,:]*np.cos(Y)
    NM=np.abs((0*np.abs(changeO1)-np.abs(changeN1)))
    NM2=(changeO1-changeN1)

    #img_back1=(NM[:-1,:-1]+NM[1:,:-1]+NM[1:,1:]+NM[:-1,:1])/4
    
    return NM, NM2
    










    return enveloppe3