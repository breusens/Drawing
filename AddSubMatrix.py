import numpy as np
from performance import performance

def AddSubMatrix(DF,filt,point):
    rows,cols=DF.shape
    x,y=filt.shape
    srwos=x//2
    scols=y//2
    
    strow=point[0]-srwos
    enrow=point[0]+srwos+1
    stcol=point[1]-scols
    encol=point[1]+scols+1
    
    clrow0=min(strow,0)
    clrow1=max(enrow-rows,0)
    clcol0=min(stcol,0)
    clcol1=max(encol-cols,0)

    strow=strow-clrow0
    enrow=enrow-clrow1
    stcol=stcol-clcol0
    encol=encol-clcol1

    clrow0=-clrow0
    clrow1=x-clrow1
    clcol0=-clcol0
    clcol1=y-clcol1

    performance(DF,filt,strow,enrow,stcol,encol,clrow0,clrow1,clcol0,clcol1)
    #DF[strow:enrow,stcol:encol]+=filt[clrow0:clrow1,clcol0:clcol1]


X=np.zeros((10,10))
position=[5,5]
Y=np.ones((5,5))
AddSubMatrix(X,Y,position)
print(X)



