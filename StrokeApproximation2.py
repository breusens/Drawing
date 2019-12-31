import numpy as np
from GravitationalFilter import GravitationalFilter
from get_neighbours import get_neighbours
import cv2
import matplotlib.pyplot as plt

def OneStroke(X,dst):
    degree=3
    rows,cols=X.shape
    Y=np.zeros((rows,cols))
    filt=GravitationalFilter(np.sqrt(2),degree,20)
    #dstc = -cv2.filter2D(X,-1,filt)   
    dst2=dst
    bottom=np.argmin(dst)
    coord=np.unravel_index(bottom,(rows,cols))
    Y=0*X
    if (X[coord]>0):
        line=[]
        black= X[coord]<0
        line.append(coord)
        Y[tuple(coord)]=1
        this=coord
        xmin=max(coord[1]-20,0)
        xmax=min(coord[1]+21,cols)
        ymin=max(coord[0]-20,0)
        ymax=min(coord[0]+21,rows)
        xmin2=xmin-(coord[1]-20)
        xmax2=41+xmax-(coord[1]+21)
        ymin2=ymin-(coord[0]-20)
        ymax2=41+ymax-(coord[0]+21)
        
        neighbours=get_neighbours(this, exclude_p=True, shape=(rows,cols))
        dif=np.zeros(neighbours.shape[0])
        for counter, point in enumerate(neighbours):
            dif[counter]=(-dst[this]+dst[tuple(point)])/np.power(np.linalg.norm(this-point),1)
        volg1=np.argmin(dif)
        direct1=neighbours[volg1]-this
        dst[ymin:ymax,xmin:xmax]=dst[ymin:ymax,xmin:xmax]+2*filt[ymin2:ymax2,xmin2:xmax2]
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
            coord=volg
            Y[tuple(coord)]=1
            xmin=max(coord[1]-20,0)
            xmax=min(coord[1]+21,cols)
            ymin=max(coord[0]-20,0)
            ymax=min(coord[0]+21,rows)
            xmin2=xmin-(coord[1]-20)
            xmax2=41+xmax-(coord[1]+21)
            ymin2=ymin-(coord[0]-20)
            ymax2=41+ymax-(coord[0]+21)
           
            
            neighbours=get_neighbours(volg, exclude_p=True, shape=(rows,cols))
            dif=np.zeros(neighbours.shape[0])
            this=volg
            for counter, point in enumerate(neighbours):
                dif[counter]=(-dst[tuple(volg)]+dst[tuple(point)])/np.power(np.linalg.norm(volg-point),1)
                if np.dot(point-volg,direct)<=0:
                    dif[counter]=1E9
            volg=np.argmin(dif)
            volg=neighbours[volg]
            direct=volg-this
            dst[ymin:ymax,xmin:xmax]=dst[ymin:ymax,xmin:xmax]+2*filt[ymin2:ymax2,xmin2:xmax2]
            if (X[tuple(volg)]<0 or dst[tuple(volg)]>0 or np.max(volg)>=rows-1 or np.min(volg)<=0 or line.count(tuple(volg))==1):
                black=False
        black=X[tuple(volg2)]>0
        volg=volg2
        direct=direct2
        while black:
            line.append(tuple(volg))
            coord=volg
            Y[tuple(coord)]=1
            xmin=max(coord[1]-20,0)
            xmax=min(coord[1]+21,cols)
            ymin=max(coord[0]-20,0)
            ymax=min(coord[0]+21,rows)
            xmin2=xmin-(coord[1]-20)
            xmax2=41+xmax-(coord[1]+21)
            ymin2=ymin-(coord[0]-20)
            ymax2=41+ymax-(coord[0]+21)
            
            neighbours=get_neighbours(volg, exclude_p=True, shape=(rows,cols))
            dif=np.zeros(neighbours.shape[0])
            this=volg
            for counter, point in enumerate(neighbours):
                dif[counter]=(-dst[tuple(volg)]+dst[tuple(point)])/np.power(np.linalg.norm(volg-point),1)
                if np.dot(point-volg,direct)<=0:
                    dif[counter]=1E9
            volg=np.argmin(dif)
            volg=neighbours[volg]
            direct=volg-this
            dst[ymin:ymax,xmin:xmax]=dst[ymin:ymax,xmin:xmax]+2*filt[ymin2:ymax2,xmin2:xmax2]
            if  (X[tuple(volg)]<0 or dst[tuple(volg)]>0 or np.max(volg)>=rows-1 or np.min(volg)<=0 or line.count(tuple(volg))==1):
                black=False
        xmax=0
        xmin=cols
        ymax=0
        ymin=rows
        #for point in line:
            #Y[tuple(point)]=1
            #xmax=max(xmax,point[1])
            #xmin=min(xmin,point[1])
            #ymax=max(ymax,point[0])
            #ymin=min(ymin,point[0])
        #xmax=min(xmax+22,cols)
        #xmin=max(xmin-21,0)
        #ymax=min(ymax+22,rows)
        #ymin=max(ymin-21,0)
        #dst2=0*dst2
        #dst2[ymin:ymax,xmin:xmax]=2*cv2.filter2D(Y[ymin:ymax,xmin:xmax],-1,filt)
        #dst2=dst+dst2
        
    return Y,dst

def StrokeApproximation(X):
    progress=True
    rows,cols=X.shape
    Y=np.zeros((rows,cols))
    i=0
    filt=GravitationalFilter(np.sqrt(2),3,20)
    dst = -cv2.filter2D(X,-1,filt)
    while progress:
        i=i+1
        Z, dst=OneStroke(X-2*Y,dst)
        progress=False
        if np.max(Z)>0:
            progress=True
            Y=Y+Z
        if (i%1000==0):
            cv2.imshow('cur',1-Y)
            cv2.waitKey(1)
    return Y

