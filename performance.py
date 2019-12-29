import numpy as np

def performance(DF,filt,strow,enrow,stcol,encol,clrow0,clrow1,clcol0,clcol1):
    DF[strow:enrow,stcol:encol]+=filt[clrow0:clrow1,clcol0:clcol1]