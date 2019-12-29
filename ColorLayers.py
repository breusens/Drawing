import numpy as np
X=cv2.imread()
r,phi,theta=CartToPolar(np.reshape(X,(cols*rows,chs)))
            #plt.hist2d(phi,theta)