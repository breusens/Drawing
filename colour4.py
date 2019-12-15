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
            for rb in range(rows//NBlock+1):
                for cb in range(cols//NBlock+1):
                    
                    thissl1=slice(np.maximum(rb,0)*NBlock,np.minimum(rb+1,rows//NBlock)*NBlock)
                    thissl2=slice(np.maximum(cb,0)*NBlock,np.minimum(cb+1,cols//NBlock)*NBlock)

                    sl1=slice(np.maximum(rb-1,0)*NBlock,np.minimum(rb+2,rows//NBlock)*NBlock)
                    sl2=slice(np.maximum(cb-1,0)*NBlock,np.minimum(cb+2,cols//NBlock)*NBlock)
                    rsl1=slice(NBlock,2*NBlock)
                    rsl2=slice(NBlock,2*NBlock)
                    
                    TWeight=Weight.copy()
                    if (rb==0):
                        TWeight=np.delete(TWeight,slice(NBlock),0)
                        sl1=slice(0,2*NBlock)
                        rsl1=slice(0,NBlock)
                    if (cb==0):
                        TWeight=np.delete(TWeight,slice(NBlock),1)
                        sl2=slice(0,2*NBlock)
                        rsl2=slice(0,NBlock)
                    if (rb==rows//NBlock):
                        TWeight=np.delete(TWeight,slice(NBlock,2*NBlock),0)
                        sl1=slice((rb-1)*NBlock,rows)
                        rsl1=slice(NBlock+1,2*NBlock)
                    if (cb==cols//NBlock):
                        TWeight=np.delete(TWeight,slice(NBlock,2*NBlock),1) 
                        sl2=slice((cb-1)*NBlock,cols)
                        rsl2=slice(NBlock+1,2*NBlock)


                    Intermediate1=b1[sl1,sl2].copy()
                    Intermediate2=b2[sl1,sl2].copy()
                    Intermediate3=b3[sl1,sl2].copy()
                    #Intermediate1=np.maximum((Intermediate1+1)/258,1/258)
                    #Intermediate2=np.maximum((Intermediate2+1)/258,1/258)
                    #Intermediate3=np.maximum((Intermediate3+1)/258,1/258)
                    #Intermediate1=np.log(Intermediate1/(1-Intermediate1))
                    #Intermediate2=np.log(Intermediate2/(1-Intermediate2))
                    #Intermediate3=np.log(Intermediate3/(1-Intermediate3))

                    f = np.fft.fft2(Intermediate1)
                    fshift = np.fft.fftshift(f)
                    rowsI, colsI = Intermediate1.shape
                    crow,ccol = rowsI//2 , colsI//2
                    fshift[crow-lf:crow+lf, ccol-lf:ccol+lf] = 0
                    #fshift[:crow-35,:] = 0
                    #fshift[crow+35:,:] = 0
                    #fshift[:,ccol+35:] = 0
                    #fshift[:,:ccol-35] = 0
                    f_ishift = np.fft.ifftshift(fshift)
                    img_back = np.fft.ifft2(f_ishift)
                    img_back1 = np.real(img_back)

                    f = np.fft.fft2(Intermediate2)
                    fshift = np.fft.fftshift(f)
                    rowsI, colsI = Intermediate1.shape
                    crow,ccol = rowsI//2 , colsI//2
                    fshift[crow-lf:crow+lf, ccol-lf:ccol+lf] = 0
                    #fshift[:crow-35,:] = 0
                    #fshift[crow+35:,:] = 0
                    #fshift[:,ccol+35:] = 0
                    #fshift[:,:ccol-35] = 0
                    f_ishift = np.fft.ifftshift(fshift)
                    img_back = np.fft.ifft2(f_ishift)
                    img_back2 = np.real(img_back)

                    f = np.fft.fft2(Intermediate3)
                    fshift = np.fft.fftshift(f)
                    rowsI, colsI = Intermediate1.shape
                    crow,ccol = rowsI//2 , colsI//2
                    fshift[crow-lf:crow+lf, ccol-lf:ccol+lf] = 0
                    #fshift[:crow-35,:] = 0
                    #fshift[crow+35:,:] = 0
                    #fshift[:,ccol+35:] = 0
                    #fshift[:,:ccol-35] = 0
                    f_ishift = np.fft.ifftshift(fshift)
                    img_back = np.fft.ifft2(f_ishift)
                    img_back3 = np.real(img_back)

                    img_back1=img_back1[rsl1,rsl2]
                    img_back2=img_back2[rsl1,rsl2]
                    img_back3=img_back3[rsl1,rsl2]

                    
                    finalimage[thissl1,thissl2,0]= img_back1
                    finalimage[thissl1,thissl2,1]= img_back2
                    finalimage[thissl1,thissl2,2]= img_back3

           

            

            beeldout=np.exp(finalimage)/(1+np.exp(finalimage))*255

            img_back1=finalimage[:,:,0]
            img_back2=finalimage[:,:,1]
            img_back3=finalimage[:,:,2]


            cv2.imshow("Input Image",beeld)
            cv2.imshow("spectrum magnitude",beeldout)
            b1s=b1
            dx=np.diff(b1s,axis=1)[:-1,:]
            I=dx==0
            Y= np.arctan(np.diff(b1s,axis=0)[:,:-1]/np.diff(b1s,axis=1)[:-1,:])
            Y[I]=np.pi/2
            changeO1=np.diff(b1,axis=0)[:,:-1]*np.sin(Y)+np.diff(b1,axis=1)[:-1,:]*np.cos(Y)
            changeN1=np.diff(img_back1,axis=0)[:,:-1]*np.sin(Y)+np.diff(img_back1,axis=1)[:-1,:]*np.cos(Y)
            b2s=b2
            dx=np.diff(b2s,axis=1)[:-1,:]
            I=dx==0
            Y= np.arctan(np.diff(b2s,axis=0)[:,:-1]/np.diff(b2s,axis=1)[:-1,:])
            Y[I]=np.pi/2
            changeO2=np.diff(b2,axis=0)[:,:-1]*np.sin(Y)+np.diff(b2,axis=1)[:-1,:]*np.cos(Y)
            changeN2=np.diff(img_back2,axis=0)[:,:-1]*np.sin(Y)+np.diff(img_back2,axis=1)[:-1,:]*np.cos(Y)
            b3s=b3
            dx=np.diff(b3s,axis=1)[:-1,:]
            I=dx==0
            Y= np.arctan(np.diff(b3s,axis=0)[:,:-1]/np.diff(b3s,axis=1)[:-1,:])
            Y[I]=np.pi/2
            changeO3=np.diff(b3,axis=0)[:,:-1]*np.sin(Y)+np.diff(b3,axis=1)[:-1,:]*np.cos(Y)
            changeN3=np.diff(img_back3,axis=0)[:,:-1]*np.sin(Y)+np.diff(img_back3,axis=1)[:-1,:]*np.cos(Y)

            X1=np.abs((changeO1-changeN1))
            Y1=-X1
            X2=np.abs((changeO2-changeN2))
            Y2=-X2 
            X3=np.abs((changeO3-changeN3))  
            Y3=-X3

            X=np.zeros((1023,1023,3))
            Y=np.zeros((1023,1023,3))

            X[:,:,0]=X1
            X[:,:,1]=X2
            X[:,:,2]=X3

            Y[:,:,0]=Y1
            Y[:,:,1]=Y2
            Y[:,:,2]=Y3

            X=X/(1+X)*255
            Y=255-X
            
            cv2.imshow("bad pointsW",X.astype('uint8'))
            cv2.imshow("bad pointsB",Y.astype('uint8'))
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
            

            
