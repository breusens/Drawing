import os
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage.filters as filter
from quaternioDFT import filters
from linefinder import linefinder
from fftwostripe import fftwostripe
from AC import AC
from neighbourAverage import neighbourAverage
from CartToPolar import CartToPolar
from GravitationalFilter import GravitationalFilter
from NHFilter import NHFilter
from StrokeApproximation import StrokeApproximation
from ConstantApprox import ConstantApprox
from Diff3D import Diff3D
from bilateral_approximation import bilateral_approximation

my_list = os.listdir('../ffhq-dataset/images1024x1024')
x=0
y=0
N=0
scale_percent = 5
# exp(x)/(1+exp(x))  =z
# x=log(z/(1-z))
NBlock=32
lf=30

for folder in my_list:
    print(folder)
    fdir='../ffhq-dataset/images1024x1024/'+folder
    imagelist=os.listdir(fdir)
    for imagefile in imagelist:
        imf=fdir+'/'+imagefile
        beeld = cv2.imread(imf, cv2.IMREAD_UNCHANGED)
        if not beeld is None:
            beeld=cv2.cvtColor(beeld, cv2.COLOR_BGR2HSV)
            beeld[:,:,0]=beeld[:,:,0]
            a1=beeld[:,:,0]
            a2=beeld[:,:,1]
            a3=beeld[:,:,2]
            a1=bilateral_approximation(a1,a1,10,10)
            #a2=bilateral_approximation(a2,a2,10,10)
            #a3=bilateral_approximation(a3,a3,10,10)
            AV=beeld.astype('int')
            AV[:,:,0]=a1
            AV[:,:,1]=a2
            AV[:,:,2]=a3
            DT0=beeld-AV
            n1=np.minimum(DT0[:,:,0],0)
            n2=np.minimum(DT0[:,:,1],0)
            n3=np.minimum(DT0[:,:,2],0)
            DT0[:,:,0]=DT0[:,:,0]-n1-(n2+n3)/2
            DT0[:,:,1]=DT0[:,:,1]-n2-(n1+n3)/2
            DT0[:,:,2]=DT0[:,:,2]-n3-(n2+n1)/2
            BV=AV.astype('double')
            BV[:,:,0]=255/179*BV[:,:,0]
            test=np.maximum((BV+1)/258,1/258)
            test=np.log(test/(1-test))
            le=Diff3D(test)
            lb1=np.diff(test[:,:-1,:],axis=0)
            lb2=np.diff(test[:-1,:,:],axis=1)
            lu=np.sqrt(lb1*lb1+lb2*lb2)
            lu=lu/(1+lu)*255
            x=lu[:,:,1].copy()
            lu[:,:,1]=lu[:,:,2]
            lu[:,:,2]=x

            lv=255-2*lu
            lu=255-lu
            
            lu[:,:,0]=a1[:-1,:-1]
            lv[:,:,0]=a1[:-1,:-1]
            lu=lu.astype('uint8')
            lv=lv.astype('uint8')
            #lu[:,:,1]=255-lu[:,:,1]
            AV[:,:,1]=250
            AV[:,:,2]=250
            AV=AV.astype('uint8')
            diff1=lu[:,:,1]
            diff2=lu[:,:,2]
            lu=cv2.cvtColor(lu, cv2.COLOR_HSV2BGR)
            lv=cv2.cvtColor(lv, cv2.COLOR_HSV2BGR)
            beeld=cv2.cvtColor(beeld, cv2.COLOR_HSV2BGR)
            AV=cv2.cvtColor(AV, cv2.COLOR_HSV2BGR)
            cv2.imshow("Input Image",beeld)
            cv2.imshow("diff1",diff1)
            cv2.imshow("diff2",diff2)
            cv2.imshow("color0",lv)
            cv2.imshow("old",lu)
            cv2.imshow("a2",a2)
            cv2.imshow("AV",AV)
            
            cv2.waitKey()
            #plt.show()
            