from cv2 import *
import numpy as np
import matplotlib.pyplot as plt
import fgrad

# Suppresses the non-maximum values in the image
# 1. For each pixel, find the direction D* in (0, 45, 90, 135) that is closest 
#    to the orientation D at that pixel
# 2. If the edge strength F(x,y) is smaller than at least one of its neighbors along D*,
#    set I(x,y) = 0, else set I(x,y) = F(x,y)
def suppress(img, D):
    dstar = convertScaleAbs(D)
    m, n = img.shape
    padded = copyMakeBorder(dstar, 1, 1, 1, 1, BORDER_CONSTANT)
    sup = np.zeros((m,n))
    for i in range(1, m-1):
        for j in range(1, n-1):
            s, r = 255, 255
            if(padded[i,j] == 0):
                s = img[i, j + 1]
                r = img[i, j - 1]
            elif(padded[i,j] == 1):
                s = img[i + 1, j - 1]
                r = img[i - 1, j + 1]
            elif(padded[i,j] == 2):
                s = img[i + 1, j]
                r = img[i - 1, j]
            elif(padded[i,j] == 3):
                s = img[i - 1, j - 1]
                r = img[i + 1, j + 1]
            
            if((img[i,j] > s) and (img[i,j] > r)):
                sup[i,j] = img[i,j]
            else:
                sup[i,j] = 0
    return sup

def main():
    img = imread("lena.png", 0)
    F, D = fgrad.gradient(img)
    sup = suppress(F, D)
    f = plt.figure(figsize=(5,5))
    plt.imshow(sup, cmap='gray')
    plt.show()
if __name__ == "__main__":
    main()