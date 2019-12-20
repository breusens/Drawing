import numpy as np
import matplotlib.pyplot as plt

def smallsquare(colour,image1,image2):
    rows,cols=image1.shape
    dif1=np.abs(image1-colour[rows//2,cols//2])
    dif2=np.abs(colour[rows//2,cols//2]-image2)
    sumc1=np.zeros(rows)
    sumc2=np.zeros(rows)
    for i in range(rows):
        for j in range(cols):

            start=i-rows//2
            end=rows-i-rows//2-1
            this=((j)*end+(rows-1-j)*start)/(rows-1)
            this=np.sign(this)*np.ceil(abs(this))
            index=rows//2+this.astype(int)
            sumc1[i]=sumc1[i]+dif1[index,j]
            sumc2[i]=sumc2[i]+dif2[index,j]
    pc1=np.sum(sumc1>sumc1[rows//2])/rows
    pc2=np.sum(sumc2>sumc2[rows//2])/rows
    return (np.pi/2+np.arctan(100*(pc1-0.9)))*(np.pi/2+np.arctan(100*(0.55-pc2)))/(np.pi*np.pi)

def linefinder(colour,image1,image2,block):
    rows,cols=image1.shape
    ri=np.zeros((rows,cols))
    ran=np.arange(block,rows-block-1)
    for x in ran:
        print(x)
        for y in ran:
            ri[x,y]=smallsquare(colour[x-block:x+block+1,y-block:y+block+1],image1[x-block:x+block+1,y-block:y+block+1],image2[x-block:x+block+1,y-block:y+block+1])
    return ri




