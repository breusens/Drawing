import numpy as np

def NearestNeighbours(n):
    X=np.arange(-n,n+1)
    Y=[]
    for rs in X:
        for cl in X:
            if (cl*cl+rs*rs)<=n*n:
                Y.append((X[rs],X[cl]))
    Z=np.zeros((len(Y),2))
    for counter,point in enumerate(Y):
        Z[counter,:]=np.array(point)

    return  Z.astype(int)

