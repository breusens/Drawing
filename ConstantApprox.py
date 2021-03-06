import numpy as np
import cv2

def OnePass(X,F,T):
    dst = cv2.filter2D(X,-1,F)
    L0=0
    L1=0.99
    B0=np.where(dst>L0)
    B1=np.where(dst>L1)
    V0=X[B0].sum/B0.sum
    V1=X[B1].sum/B1.sum
    Y=0*X
    if (V0>T):
        Y[B0]=1
    else:
        if (V1>T):
            while np.abs(L0-L1)<0.01:
                # V0(1-x)+x*V1=T
                #x=(T-V0)/(V1-V0)
                LN=L0+(T-V0)/(V1-V0)*(L1-L0)
                BN=np.where(dst>LN)
                VN=X[BN].sum/BN.sum
                if (T>VN):
                    L0=LN
                    B0=BN
                    V0=VN
                else:
                    L1=LN
                    B1=BN
                    V1=VN
            Y[B1]=1
    return Y


def ConstantApprox(X,F,T):
    improve=True
    Y=0*X
    Z=X
    while improve:
        U=OnePass(X,F,T)
        improve=false
        if np.any(U>0):
            Z=Z-2*U
            Y=Y+U
            improve=True
    return Y










