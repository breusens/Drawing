import numpy as np
def Diff3D(Image):

#(R(x)-R(0))^2+(G(x)-G(0))^2+(B(x)-B(0))^2
#R(x)=(sin(T)*DRy+cos(T)*DRx)^2+(sin(T)*DGy+cos(T)*DGx)^2+(sin(T)*DBy+cos(T)*DBx)^2
#2*(sin(T)*DRx+cos(T)*DRy)*(cos(T)*DRx-sin(T)*DRy)+2*(sin(T)*DRx+cos(T)*DRy)*(cos(T)*DRx-sin(T)*DRy)+2*(sin(T)*DRx+cos(T)*DRy)*(cos(T)*DRx-sin(T)*DRy)
#Drx^2*(sin(T)*cos(T))+Drx*Dry*(cos(T)^2-sin(T)^2)-Dry^2*sin(T)*cos(T)
#(Drx^2-Dry^2)/(Drx*Dry)=(sin(T)/cos(T)-cos(T)/sin(T))
#u=(x-1/x)
#x^2-x*u-1=0
#(x-u/2)^2=1+u^2/4
#tan(T)=u/2+_sqrt(1+u^2/4)

    Drx=np.diff(Image[:-1,:,:],axis=1)
    Dry=np.diff(Image[:,:-1,:],axis=0)
    Drx2=Drx*Drx
    Dry2=Dry*Dry
    Drxy=Drx*Dry
    Drx2=np.sum(Drx,axis=2)
    Dry2=np.sum(Dry,axis=2)
    Drxy=np.sum(Drxy,axis=2)
    u=(Dry2-Drx2)/(Drxy)
    tT1=u/2+np.sqrt(1+u*u/4)
    tT2=u/2-np.sqrt(1+u*u/4)
    T1=np.arctan(tT1)
    T2=np.arctan(tT2)
    T1[np.where(Drxy==0)]=np.pi/2
    T2[np.where(Drxy==0)]=0
    D1=np.tile(np.expand_dims(np.sin(T1),axis=2),(1,1,3))*Dry+np.tile(np.expand_dims(np.cos(T1),axis=2),(1,1,3))*Drx
    D2=np.tile(np.expand_dims(np.sin(T2),axis=2),(1,1,3))*Dry+np.tile(np.expand_dims(np.cos(T2),axis=2),(1,1,3))*Drx
    D1=np.sum(D1*D1,axis=2)
    D2=np.sum(D2*D2,axis=2)
    T=T1
    wrong=np.where(D2>D1)
    T[wrong]=T2[wrong]
    return np.tile(np.expand_dims(np.sin(T),axis=2),(1,1,3))*Dry+np.tile(np.expand_dims(np.cos(T),axis=2),(1,1,3))*Drx







