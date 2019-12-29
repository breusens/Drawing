import numpy as np
from GravitationalFilter import GravitationalFilter
from get_neighbours import get_neighbours
import cv2
from NearestNeighbours import NearestNeighbours
from AddSubMatrix import AddSubMatrix

def offset(coord,hood,shape):
    hood=coord+hood
    valid = np.all((hood < np.array(shape)) & (hood >= 0), axis=1)
    hood=hood[valid]
    return hood

def OneStroke(X,width,dst,filt):
    x,y=filt.shape
    srwos=x//2
    scols=y//2
    fx,fy=filt.shape
    fx=fx//2
    fy=fy//2
    Hood1=NearestNeighbours(width[0])
    Hood2=NearestNeighbours(width[1])
    rows,cols=X.shape
    Y=np.zeros((rows,cols))
    DF=np.zeros((rows,cols))
    #filt=GravitationalFilter(1,3,20)
    #dst = -cv2.filter2D(X,-1,filt)   
    #u=250*(dst2+1)/2
    #v=250*(dst+1)/2
    #cv2.imshow('appr',u)
    #cv2.imshow('exact',v)
    #cv2.waitKey()
    bottom=np.argmin(dst)
    coord=np.unravel_index(bottom,(rows,cols))


    Thood1=offset(coord,Hood1,X.shape)
    n1,u1=Thood1.shape
    Thood2=offset(coord,Hood2,X.shape)
    n2,u2=Thood2.shape


    pb1=np.sum(X[Thood1[:,0],Thood1[:,1]])/n1
    pb2=np.sum(X[Thood2[:,0],Thood2[:,1]])/n2
    
    if (pb1>0.5):
        line1=[]
        line2=[]

        black= X[coord]<0
        line1=line1+Thood1.tolist()
        line2=line2+Thood2.tolist()

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
        Thood1=offset(volg1,Hood1,X.shape)
        n1,u1=Thood1.shape
        Thood2=offset(volg1,Hood2,X.shape)
        n2,u2=Thood2.shape


        pb1=np.sum(X[Thood1[:,0],Thood1[:,1]])/n1
        pb2=np.sum(X[Thood2[:,0],Thood2[:,1]])/n2
        black=pb1>0.5

        volg=volg1
        direct=direct1
        while black:
            line1=line1+Thood1.tolist()
            line2=line2+Thood2.tolist()
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
            Thood1=offset(volg,Hood1,X.shape)
            n1,u1=Thood1.shape
            Thood2=offset(volg,Hood2,X.shape)
            n2,u2=Thood2.shape
            pb1=np.sum(X[Thood1[:,0],Thood1[:,1]])/n1
            pb2=np.sum(X[Thood2[:,0],Thood2[:,1]])/n2
            black=pb1>0.5
            if (X[tuple(volg)]<0 or dst[tuple(volg)]>0 or np.max(volg)>=rows-1 or np.min(volg)<=0):
                black=False
        
        black=X[tuple(volg2)]>0
        volg=volg2
        Thood1=offset(volg2,Hood1,X.shape)
        n1,u1=Thood1.shape
        Thood2=offset(volg2,Hood2,X.shape)
        n2,u2=Thood2.shape


        pb1=np.sum(X[Thood1[:,0],Thood1[:,1]])/n1
        pb2=np.sum(X[Thood2[:,0],Thood2[:,1]])/n2
        black=pb1>0.5
        direct=direct2
        while black:
            line1=line1+Thood1.tolist()
            line2=line2+Thood2.tolist()
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
            Thood1=offset(volg,Hood1,X.shape)
            n1,u1=Thood1.shape
            Thood2=offset(volg,Hood2,X.shape)
            n2,u2=Thood2.shape
            pb1=np.sum(X[Thood1[:,0],Thood1[:,1]])/n1
            pb2=np.sum(X[Thood2[:,0],Thood2[:,1]])/n2
            black=pb1>0.5
            if  (X[tuple(volg)]<0 or dst[tuple(volg)]>0 or np.max(volg)>=rows-1 or np.min(volg)<=0):
                black=False   
        l1=np.asarray(line1)
        l2=np.asarray(line2)
        pb1=np.sum(X[l1[:,0],l1[:,1]])/len(line1)
        pb2=np.sum(X[l2[:,0],l2[:,1]])/len(line2)
        if pb2>0.8:
            mnx=max(np.min(l2[:,0])-fx,0)
            mxx=min(np.max(l2[:,0])+fx,rows)
            mny=max(np.min(l2[:,1])-fy,0)
            mxy=min(np.max(l2[:,1])+fy,cols)

            for point in line2:
                if Y[tuple(point)]==0:
                    #AddSubMatrix(DF,filt,point)
                    Y[tuple(point)]=1
            DF[mnx:mxx,mny:mxy] = cv2.filter2D(Y[mnx:mxx,mny:mxy],-1,filt)
            
        else:
            mnx=max(np.min(l1[:,0])-fx,0)
            mxx=min(np.max(l1[:,0])+fx,rows)
            mny=max(np.min(l1[:,1])-fy,0)
            mxy=min(np.max(l1[:,1])+fy,cols)
            for point in line1:
                if Y[tuple(point)]==0:
                    Y[tuple(point)]=1
                    #AddSubMatrix(DF,filt,point)
            DF[mnx:mxx,mny:mxy] = cv2.filter2D(Y[mnx:mxx,mny:mxy],-1,filt)  

    return Y,DF
