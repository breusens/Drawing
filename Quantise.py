import numpy as np

def Quantise(X,step):
    ((X+step//2)//step)*step
