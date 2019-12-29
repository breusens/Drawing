
from skimage.filters import denoise_tv_chambolle
import cv2 
import matplotlib.pyplot as plt
from glob import glob

pngs = glob('./*.png')

for j in pngs:
    img = cv2.imread(j)
    tv_denoised = flt.tv_denoise(img, weight=10)
    cv2.imshow(tv_denoised)
    cv2.waitKey()
    plt.plot(img[500,:,0])
    plt.plot(tv_denoised[500,:,0])
    plt.show()