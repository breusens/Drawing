from glob import glob                                                           
import cv2 
import pyguetzli
import matplotlib.pyplot as plt
from bilateral_approximation import bilateral_approximation


pngs = glob('./*.png')

for j in pngs:
    img = cv2.imread(j)
    plt.plot(img[500,:,0])
    lb=img
    for i in range(10):
        lb[:,:,0]=bilateral_approximation(lb[:,:,0], img[:,:,0], 6, 4)
        lb[:,:,1]=bilateral_approximation(lb[:,:,1], img[:,:,1], 6, 4)
        lb[:,:,2]=bilateral_approximation(lb[:,:,2], img[:,:,2], 6, 4)
                
    plt.plot(lb[500,:,0])
    cv2.imshow('new',lb)
    cv2.waitKey()
    plt.show()
    

    