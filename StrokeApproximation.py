import numpy as np
from GravitationalFilter import GravitationalFilter
from get_neighbours import get_neighbours
import cv2

def OneStroke(X):
    rows,cols=X.shape
    Y=np.zeros((rows,cols))
    filt=GravitationalFilter(1,3,20)
    dst = -cv2.filter2D(X,-1,filt)   
    bottom=np.argmin(dst)
    coord=np.unravel_index(bottom,(rows,cols))
    if (X[coord]>0):
        line=[]
        black= X[coord]<0
        line.append(coord)
        this=coord
        neighbours=get_neighbours(this, exclude_p=True, shape=(rows,cols))
        dif=np.zeros(neighbours.shape[0])
        for counter, point in enumerate(neighbours):
            dif[counter]=(-dst[this]+dst[tuple(point)])/np.linalg.norm(this-point)
        volg1=np.argmin(dif)
        direct1=neighbours[volg1]-this
        for counter, point in enumerate(neighbours):
            check=np.dot(point-this,direct1)
            if check>0:
                dif[counter]=np.max(dif)
        volg2=np.argmin(dif)
        direct2=neighbours[volg2]-this
        volg1=neighbours[volg1]
        volg2=neighbours[volg2]
        black=X[tuple(volg1)]>0
        volg=volg1
        direct=direct1
        while black:
            line.append(tuple(volg))
            neighbours=get_neighbours(volg, exclude_p=True, shape=(rows,cols))
            dif=np.zeros(neighbours.shape[0])
            this=volg
            for counter, point in enumerate(neighbours):
                dif[counter]=(-dst[tuple(volg)]+dst[tuple(point)])/np.linalg.norm(volg-point)
                if np.dot(point-volg,direct1)<=0:
                    dif[counter]=1E9
            volg=np.argmin(dif)
            volg=neighbours[volg]
            direct=volg-this
            if (X[tuple(volg)]<0 or dst[tuple(volg)]>0 or np.max(volg)>=rows-1 or np.min(volg)<=0 or line.count(tuple(volg))==1):
                black=False
        black=X[tuple(volg2)]>0
        volg=volg2
        direct=direct2
        while black:
            line.append(tuple(volg))
            neighbours=get_neighbours(volg, exclude_p=True, shape=(rows,cols))
            dif=np.zeros(neighbours.shape[0])
            this=volg
            for counter, point in enumerate(neighbours):
                dif[counter]=(-dst[tuple(volg)]+dst[tuple(point)])/np.linalg.norm(volg-point)
                if np.dot(point-volg,direct2)<=0:
                    dif[counter]=1E9
            volg=np.argmin(dif)
            volg=neighbours[volg]
            direct=volg-this
            if  (X[tuple(volg)]<0 or dst[tuple(volg)]>0 or np.max(volg)>=rows-1 or np.min(volg)<=0 or line.count(tuple(volg))==1):
                black=False
        for point in line:
            Y[tuple(point)]=1
    return Y

def StrokeApproximation(X):
    progress=True
    rows,cols=X.shape
    Y=np.zeros((rows,cols))
    i=0
    while progress:
        i=i+1
        Z=OneStroke(X-2*Y)
        progress=False
        if np.max(Z)>0:
            progress=True
            Y=Y+Z
        if (i%200==0):
            print(i)
    return Y

