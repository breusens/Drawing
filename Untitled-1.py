import os
import numpy as np 
import matplotlib.pyplot as plt
import cv2

my_list = os.listdir('../ffhq-dataset/images1024x1024')
x=0
y=0
N=0
scale_percent = 5

for folder in my_list:
    print(folder)
    f='../ffhq-dataset/images1024x1024/'+folder
    imagelist=os.listdir(f)
    for imagefile in imagelist:
        imf=f+'/'+imagefile
        beeld = cv2.imread(imf, cv2.IMREAD_UNCHANGED)
        if not beeld is None:
            width = int(beeld.shape[1] * scale_percent / 100)
            height = int(beeld.shape[0] * scale_percent / 100)
            dim = (width, height)
            beeld = cv2.resize(beeld, dim, interpolation = cv2.INTER_AREA)
            rows,cols,colours=beeld.shape
            img_size=rows*cols*colours
            img_1D_vector=np.minimum(np.maximum(beeld.reshape(img_size),1),254)/255
            img_1D_vector=np.log(img_1D_vector/(1-img_1D_vector))
            y=y+img_1D_vector
            x=x+np.outer(img_1D_vector,img_1D_vector)
            N=N+1

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
            cv2.waitKey(0)

            
