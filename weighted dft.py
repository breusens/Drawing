import numpy as np

def WDFT(a,Block,lf):
    cols,rws=a.shape()
    W=np.exp((1j)*np.pi/Block*np.arange(cols))
    WR=np.real(W)
    WI=np.imag(W)
    WR=WR*WR
    WI=WI*WI
    

    return A


