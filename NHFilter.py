import numpy as np
def NHFilter():
    flt=np.ones((3,3))
    flt[1,1]=0
    flt=flt/8
    return flt