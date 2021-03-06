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
from sklearn.decomposition import FastICA

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
            #beeld = cv2.cvtColor(beeld, cv2.COLOR_BGR2Lab)
            #beeld[:,:,0]=255/179*beeld[:,:,0]
            transformer = FastICA(n_components=3,random_state=0)
            cols,rows,chs=beeld.shape
            test=beeld.copy()
            test=np.maximum((test+1)/258,1/258)
            test=np.log(test/(1-test))
            test=np.reshape(test,(cols*rows,chs))
            test = transformer.fit_transform(test)
            test=np.reshape(test,(cols,rows,chs))

            b1=test[:,:,0]/np.max(np.abs(test[:,:,0]))
            b2=test[:,:,1]/np.max(np.abs(test[:,:,1]))
            b3=test[:,:,2]/np.max(np.abs(test[:,:,2]))

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

           



            lw=255*le/(1+le)
            lb= 255*ls/(1+ls)
            law= la/(1+la)*255
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

            filt=GravitationalFilter(1,3,20)
            ld1=255*X
            ld=np.zeros((cols,rows,3))

            #for level in range(14):

             #   ld2=np.ones((cols,rows))
              #  ld2[np.where(ld1[:,:,0]<20)]=-1
               # dst1=ConstantApprox(ld2,filt,0.95)
#
 #               ld2=np.ones((cols,rows))
  #              ld2[np.where(ld1[:,:,1]<20)]=-1
   #             dst2=ConstantApprox(ld2,filt,0.95)
#
 #               ld2=np.ones((cols,rows))
  #              ld2[np.where(ld1[:,:,2]<20)]=-1
   #             dst3=ConstantApprox(ld2,filt,0.95)

    #            ld[:,:,0]=ld[:,:,0]+20*dst1
     #           ld[:,:,1]=ld[:,:,1]+20*dst2
      #          ld[:,:,2]=ld[:,:,2]+20*dst3
       #         dst3=(1+dst3)/2
        #        ld1=255*X-ld   

                #cv2.imshow('temp',255-ld.astype('uint8'))
                #cv2.waitKey(1)

            ld=255-ld
            #lw[:,:,0]=179/255*lw[:,:,0]
            #lb[:,:,0]=179/255*lb[:,:,0]
            #lw = cv2.cvtColor(lw.astype('uint8'), cv2.COLOR_Lab2BGR)
            #lb = cv2.cvtColor(lb.astype('uint8'), cv2.COLOR_Lab2BGR)
            #lw=transformer.inverse_transform(np.reshape(lw,(cols*rows,chs)))
            #lw=np.reshape(lw,(cols,rows,chs))
            #lb=transformer.inverse_transform(np.reshape(lb,(cols*rows,chs)))
            #lb=np.reshape(lb,(cols,rows,chs))
            #lw=100*255*lw/np.max(lw)
            #lb=100*255*lb/np.max(lb)
            cv2.imshow("Input Image",beeld)
            cv2.imshow("white1",255-10*lw.astype('uint8'))
            cv2.imshow("white2",255-10*lb.astype('uint8'))
           

            cv2.waitKey()
            #plt.show()
            





