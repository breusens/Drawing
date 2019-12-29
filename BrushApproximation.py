import numpy as np
from GravitationalFilter import GravitationalFilter
from get_neighbours import get_neighbours
import cv2
from NearestNeighbours import NearestNeighbours
from OneStroke import OneStroke 
import cProfile



def BrushApproximation(X,width):
    progress=True
    rows,cols=X.shape
    Y=np.zeros((rows,cols))
    i=0
    filt=GravitationalFilter(1,3,20)
    dst = -cv2.filter2D(X,-1,filt)
    while progress:
        i=i+1
        #cProfile.runctx('OneStroke(X-2*Y,width,dst,filt)', globals=globals(), locals=locals())
        Z,DF=OneStroke(X-2*Y,width,dst,filt)
        dst=dst+2*DF
        progress=False
        if np.max(Z)>0:
            progress=True
            Y=Y+Z
        if (i%200==0):
            print(i)
    return Y

