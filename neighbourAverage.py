import numpy as np
def neighbourAverage(x):
    
    diffs1=np.abs(np.diff(x,axis=0))
    diffs2=np.abs(np.diff(x,axis=1))
    UpDiff=diffs1[:-1,1:-1]
    DnDiff=diffs1[1:,1:-1]
    LDiff=diffs2[1:-1,:-1]
    RDiff=diffs2[1:-1,1:]
    Up=np.logical_and(UpDiff<DnDiff,UpDiff<LDiff,UpDiff<RDiff)
    Dn=np.logical_and(DnDiff<UpDiff,DnDiff<LDiff,DnDiff<RDiff)
    R=np.logical_and(RDiff<DnDiff,RDiff<LDiff,RDiff<UpDiff)
    L=np.logical_and(LDiff<DnDiff,LDiff<UpDiff,LDiff<RDiff)

    z=x[1:-1,1:-1]

    UA=x[:-2,1:-1]
    DA=x[2:,1:-1]
    LA=x[1:-1,:-2]
    RA=x[1:-1,2:]

    z[Up]=0.5*(z+UA)[Up]
    z[Dn]=0.5*(z+DA)[Dn]
    z[R]=0.5*(z+RA)[R]
    z[L]=0.5*(z+LA)[L]

    return z


    




