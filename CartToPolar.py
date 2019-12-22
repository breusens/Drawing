import numpy as np
def CartToPolar(x):
    
    r=np.sqrt(x[:,0]*x[:,0]+x[:,1]*x[:,1]+x[:,2]*x[:,2])
    y=np.delete(x,np.where(r==0),0)
    I=y[:,0]==0
    r=np.sqrt(y[:,0]*y[:,0]+y[:,1]*y[:,1]+y[:,2]*y[:,2])
    phi=np.arctan(y[:,1]/y[:,0])
    phi[I]=np.pi/2
    theta=np.arccos(y[:,2]/r)
    return r,phi,theta
