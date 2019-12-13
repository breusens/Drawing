import os
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from quaternioDFT import filters
my_list = os.listdir('../ffhq-dataset/images1024x1024')
x=0
y=0
N=0
scale_percent = 5
# exp(x)/(1+exp(x))  =z
# x=log(z/(1-z))
NBlock=32
lf=35

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

            

            rows, cols = b1.shape

            finalimage=np.zeros((rows,cols,3))
            Weight=np.array((0.5+np.arange(NBlock))/NBlock)
            Weight=np.concatenate((Weight,np.flip(Weight)))
            Weight=np.outer(Weight,Weight)
            crow,ccol = rows//2 , cols//2    
            Maskx=0*b1
            Masky=0*b1
            Maskx[:,:ccol]=-1
            Masky[:,:crow]=-1
            Mask=0.5*(Maskx+Masky)


            f = np.fft.fft2(b1)
            fshift = np.fft.fftshift(f)
            rows, cols = b1.shape
            crow,ccol = rows//2 , cols//2
            fshift=(1+Mask)*fshift
            fshift[crow-lf:crow+lf, ccol-lf:ccol+lf] = 0
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back1 = np.real(img_back)
            enveloppe1 =np.absolute(img_back)

            f = np.fft.fft2(b2)
            fshift = np.fft.fftshift(f)
            rows, cols = b1.shape
            crow,ccol = rows//2 , cols//2
            fshift=(1+Mask)*fshift
            fshift[crow-lf:crow+lf, ccol-lf:ccol+lf] = 0
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back2 = np.real(img_back)
            enveloppe2 =np.absolute(img_back)

            f = np.fft.fft2(b3)
            fshift = np.fft.fftshift(f)
            rows, cols = b1.shape
            crow,ccol = rows//2 , cols//2
            fshift=(1+Mask)*fshift
            fshift[crow-lf:crow+lf, ccol-lf:ccol+lf] = 0
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back3 = np.real(img_back)
            enveloppe3 =np.absolute(img_back)   


            beeldout=beeld.copy()

            beeldout[:,:,0]=np.exp(img_back1)/(1+np.exp(img_back1))*255
            beeldout[:,:,1]=np.exp(img_back2)/(1+np.exp(img_back2))*255
            beeldout[:,:,2]=np.exp(img_back3)/(1+np.exp(img_back3))*255


            cv2.imshow("Input Image",beeld)
            cv2.imshow("spectrum magnitude",beeldout)
            dx=np.diff(b1,axis=1)[:-1,:]
            I=dx==0
            Y= np.arctan(np.diff(b1,axis=0)[:,:-1]/np.diff(b1,axis=1)[:-1,:])
            Y[I]=np.pi/2
            changeO1=np.diff(b1,axis=0)[:,:-1]*np.sin(Y)+np.diff(b1,axis=1)[:-1,:]*np.cos(Y)
            changeN1=np.diff(img_back1,axis=0)[:,:-1]*np.sin(Y)+np.diff(img_back1,axis=1)[:-1,:]*np.cos(Y)
            dx=np.diff(b2,axis=1)[:-1,:]
            I=dx==0
            Y= np.arctan(np.diff(b2,axis=0)[:,:-1]/np.diff(b2,axis=1)[:-1,:])
            Y[I]=np.pi/2
            changeO2=np.diff(b2,axis=0)[:,:-1]*np.sin(Y)+np.diff(b2,axis=1)[:-1,:]*np.cos(Y)
            changeN2=np.diff(img_back2,axis=0)[:,:-1]*np.sin(Y)+np.diff(img_back2,axis=1)[:-1,:]*np.cos(Y)
            dx=np.diff(b3,axis=1)[:-1,:]
            I=dx==0
            Y= np.arctan(np.diff(b3,axis=0)[:,:-1]/np.diff(b3,axis=1)[:-1,:])
            Y[I]=np.pi/2
            changeO3=np.diff(b3,axis=0)[:,:-1]*np.sin(Y)+np.diff(b3,axis=1)[:-1,:]*np.cos(Y)
            changeN3=np.diff(img_back3,axis=0)[:,:-1]*np.sin(Y)+np.diff(img_back3,axis=1)[:-1,:]*np.cos(Y)

            X1=np.sqrt((changeO1-changeN1)*(changeO1-changeN1))  
            Y1=-X1
            X2=np.sqrt((changeO2-changeN2)*(changeO2-changeN2))  
            Y2=-X2 
            X3=np.sqrt((changeO3-changeN3)*(changeO3-changeN3))  
            Y3=-X3

            X=np.zeros((1023,1023,3))
            Y=np.zeros((1023,1023,3))

            X[:,:,0]=X1
            X[:,:,1]=X2
            X[:,:,2]=X3

            Y[:,:,0]=Y1
            Y[:,:,1]=Y2
            Y[:,:,2]=Y3


            scale1=np.min(X)
            scale2=np.max(X)
            X=255*(X-scale1)/(scale2-scale1)
            scale1=np.min(Y)
            scale2=np.max(Y)
            Y=255*(Y-scale1)/(scale2-scale1)
            cv2.imshow("bad pointsW",X.astype('uint8'))
            cv2.imshow("bad pointsB",Y.astype('uint8'))
            cv2.waitKey()

            enveloppe1=enveloppe1





sd=x/N-np.outer(y/N,y/N)
U,S,V = np.linalg.svd(sd)

ZCAMatrix=np.dot(U,np.dot(np.diag(1.0/np.sqrt(S+0.000001)),U.T))
cv2.namedWindow('original', cv2.WINDOW_NORMAL)
cv2.namedWindow('new', cv2.WINDOW_NORMAL)
for imagefile in imagelist:
        imf=f+'/'+imagefile
        beeld = cv2.imread(imf, cv2.IMREAD_UNCHANGED)
        if not beeld is None:
            width = int(beeld.shape[1] * scale_percent / 100)
            height = int(beeld.shape[0] * scale_percent / 100)
            dim = (width, height)
            beeld = cv2.resize(beeld, dim, interpolation = cv2.INTER_AREA)
            cv2.namedWindow('original', cv2.WINDOW_NORMAL)
            cv2.namedWindow('new', cv2.WINDOW_NORMAL)
            cv2.imshow("original",beeld)
            rows,cols,colours=beeld.shape
            img_size=rows*cols*colours
            img_1D_vector=np.minimum(np.maximum(beeld.reshape(img_size),1),254)/255
            img_1D_vector=np.log(img_1D_vector/(1-img_1D_vector))-y/N
            beeld=np.dot(ZCAMatrix,img_1D_vector)+y/N
            beeld=255*np.exp(beeld)/(1+np.exp(beeld))
            beeld=beeld.astype(int)
            beeld=beeld.reshape((rows,cols,colours))
            cv2.imshow("new",beeld)
            

            
