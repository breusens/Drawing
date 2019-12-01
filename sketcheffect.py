import os
import numpy as np 
import matplotlib.pyplot as plt
import cv2

my_list = os.listdir('../ffhq-dataset/images1024x1024')
x=0
y=0
N=0
scale_percent = 5
# exp(x)/(1+exp(x))  =z
# x=log(z/(1-z))
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

            b1=np.maximum((b1+1)/258,1/258)
            b2=np.maximum((b2+1)/258,1/258)
            b3=np.maximum((b3+1)/258,1/258)

            b1=np.log(b1/(1-b1))
            b2=np.log(b2/(1-b2))
            b3=np.log(b3/(1-b3))

            f = np.fft.fft2(b1)
            fshift = np.fft.fftshift(f)
            rows, cols = b1.shape
            crow,ccol = rows//2 , cols//2
            fshift[crow-lf:crow+lf, ccol-lf:ccol+lf] = 0
            #fshift[:crow-35,:] = 0
            #fshift[crow+35:,:] = 0
            #fshift[:,ccol+35:] = 0
            #fshift[:,:ccol-35] = 0
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back1 = np.real(img_back)

            f = np.fft.fft2(b2)
            fshift = np.fft.fftshift(f)
            rows, cols = b1.shape
            crow,ccol = rows//2 , cols//2
            fshift[crow-lf:crow+lf, ccol-lf:ccol+lf] = 0
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back2 = np.real(img_back)

            f = np.fft.fft2(b3)
            fshift = np.fft.fftshift(f)
            rows, cols = b1.shape
            crow,ccol = rows//2 , cols//2
            fshift[crow-lf:crow+lf, ccol-lf:ccol+lf] = 0
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back3 = np.real(img_back)

            beeldout=beeld.copy()
            beeldout[:,:,0]=np.exp(img_back1)/(1+np.exp(img_back1))*255
            beeldout[:,:,1]=np.exp(img_back2)/(1+np.exp(img_back2))*255
            beeldout[:,:,2]=np.exp(img_back3)/(1+np.exp(img_back3))*255

            cv2.imshow("Input Image",beeld)
            cv2.imshow("spectrum magnitude",beeldout)
            cv2.waitKey()


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
            

            
