import numpy as np

def GravitationalFilter(R,n,N):
    yV=np.tile(np.arange(-N,N+1),(2*N+1,1))
    r=np.sqrt((yV.T)*(yV.T)+yV*yV)
    y= -np.minimum(r,R)+(1/n)*np.power(R,(n+1))/(np.maximum(np.power(r,n),np.power(R,n)))
    y=y-np.min(y)
    y=y/np.sum(y)
    return y
