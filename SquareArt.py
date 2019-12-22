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

my_list = os.listdir('../ffhq-dataset/images1024x1024')
x=0
y=0
N=0
scale_percent = 5
# exp(x)/(1+exp(x))  =z
# x=log(z/(1-z))
NBlock=32
lf=10

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
            ls=np.zeros((1023,1023,3))

            le[:,:,0], ls[:,:,0]=fftwostripe(b1,lf)
            le[:,:,1], ls[:,:,1]=fftwostripe(b2,lf)
            le[:,:,2], ls[:,:,2]=fftwostripe(b3,lf)

            la=np.zeros((1023,1023,3))
            la[:,:,:]=le[:,:,:]
           

            #for i in range(100):
            #    la[1:-1,1:-1,0]=neighbourAverage(la[:,:,0])
            #    la[1:-1,1:-1,1]=neighbourAverage(la[:,:,1])
            #    la[1:-1,1:-1,2]=neighbourAverage(la[:,:,2])

           



            lw=255-np.exp(le)/(1+np.exp(le))*255
            lb= 255-le/(1+le)*255
            law= 255-la/(1+la)*255
            lab=la/(1+la)*255

            X=le/(1+le)

            cols,rows,chs=X.shape

            r,phi,theta=CartToPolar(np.reshape(X,(cols*rows,chs)))
            #plt.hist2d(phi,theta)
            #M*(Min(r,R))^2/r^2
            #-min(r,R)+R^2/max(r,R)
            #R
            #-R+R=0
            #-R+4*R^2/2*R=-R+2*R=+R

            filt=GravitationalFilter(1,1,20)
            ld1=255*X
            ld=np.zeros((cols,rows,3))

            for level in range(10):

                ld2=np.ones((cols,rows))
                ld2[np.where(ld1[:,:,0]<25)]=-1
                NHF=NHFilter()
            
                for repeats in range(100):
                    avdst=cv2.filter2D(ld2,-1,NHF)
                    I=np.logical_and(ld2==1,avdst<-5/8)
                    J=np.logical_and(ld2==-1,avdst>5/8)
                    ld2[I]=-1
                    ld2[J]=1

                dst = cv2.filter2D(ld2,-1,filt)
                maxi=np.argmax(dst)
                ld2 = 255-(ld2+1)*255
                dst1 = (dst+1)/2

                ld2=np.ones((cols,rows))
                ld2[np.where(ld1[:,:,1]<25)]=-1
                NHF=NHFilter()
            
                for repeats in range(100):
                    avdst=cv2.filter2D(ld2,-1,NHF)
                    I=np.logical_and(ld2==1,avdst<-5/8)
                    J=np.logical_and(ld2==-1,avdst>5/8)
                    ld2[I]=-1
                    ld2[J]=1

                dst = cv2.filter2D(ld2,-1,filt)
                maxi=np.argmax(dst)
                ld2 = 255-(ld2+1)*255
                dst2 = (dst+1)/2

                ld2=np.ones((cols,rows))
                ld2[np.where(ld1[:,:,2]<25)]=-1
                NHF=NHFilter()
            
                for repeats in range(100):
                    avdst=cv2.filter2D(ld2,-1,NHF)
                    I=np.logical_and(ld2==1,avdst<-5/8)
                    J=np.logical_and(ld2==-1,avdst>5/8)
                    ld2[I]=-1
                    ld2[J]=1

                dst = cv2.filter2D(ld2,-1,filt)
                maxi=np.argmax(dst)
                ld2 = 255-(ld2+1)*255
                dst3 = (dst+1)/2

                
                
                ld[:,:,0]=ld[:,:,0]+25*dst1
                ld[:,:,1]=ld[:,:,0]+25*dst2
                ld[:,:,2]=ld[:,:,0]+25*dst3
                ld1=255*X-ld

            ld=255-ld
            cv2.imshow("Input Image",beeld)
            cv2.imshow("white",lw.astype('uint8'))
            cv2.imshow("black",lb.astype('uint8'))
            cv2.imshow("filter",dst.astype('uint8'))
            cv2.imshow("prefilter",ld.astype('uint8'))
           

            cv2.waitKey()
            #plt.show()
            





