import os
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage.filters as filter
from quaternioDFT import filters
from linefinder import linefinder
from fftwostripe import fftwostripe

my_list = os.listdir('../ffhq-dataset/images1024x1024')
x=0
y=0
N=0
scale_percent = 5
# exp(x)/(1+exp(x))  =z
# x=log(z/(1-z))
NBlock=32
lf=1

for folder in my_list:
    print(folder)
    fdir='../ffhq-dataset/images1024x1024/'+folder
    imagelist=os.listdir(fdir)
    for imagefile in imagelist:
        imf=fdir+'/'+imagefile
        beeld = cv2.imread(imf, cv2.IMREAD_UNCHANGED)
        if not beeld is None:
            b1=beeld[:,:,0]
            b2=beeld[:,:,1]
            b3=beeld[:,:,2]
            test=beeld.copy()
            test=np.maximum((test+1)/258,1/258)
            test=np.log(test/(1-test))
            b1=test[:,:,0]
            b2=test[:,:,1]
            b3=test[:,:,2]

            le=np.zeros((1023,1023,3))

            le[:,:,0]=fftwostripe(b1,1)
            le[:,:,1]=fftwostripe(b2,1)
            le[:,:,2]=fftwostripe(b3,1)

            lw=255-np.exp(le)/(1+np.exp(le))*255
            lb= np.exp(le)/(1+np.exp(le))*255





            cv2.imshow("Input Image",beeld)
            cv2.imshow("white",lw)
            cv2.imshow("white",lb)
            
           

            cv2.waitKey()
            





