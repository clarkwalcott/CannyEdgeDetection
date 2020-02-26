from cv2 import *
import numpy as np
import matplotlib.pyplot as plt
import fgrad, nmaxsup, thresh

def hysteresis(img, weak, strong):
    x, y = img.shape
    padded = copyMakeBorder(img, 1, 1, 1, 1, BORDER_CONSTANT)
    for i in range(1, x+1):
        for j in range(1, y+1):
            if(padded[i,j] == weak):
                if((padded[i-1,j-1] == strong) or (padded[i-1, j] == strong) or 
                (padded[i-1, j+1] == strong) or (padded[i, j-1] == strong) or 
                (padded[i, j+1] == strong) or (padded[i+1, j-1] == strong) or 
                (padded[i+1, j] == strong) or (padded[i+1, j+1] == strong)):
                    img[i-1, j-1] = strong
                else:
                    img[i-1, j-1] = 0
    return img

def start(img):
    F, D = fgrad.gradient(img)
    sup = nmaxsup.suppress(F, D)
    T = thresh.thresh(sup)
    result = hysteresis(T[0], T[1], T[2])
    fig, axs = plt.subplots(ncols=3, figsize=(20,20))
    axs[0].set_title('After Suppression')
    axs[0].imshow(sup, cmap='gray')
    axs[0].axis('off')
    axs[1].set_title('After Thresholding')
    axs[1].imshow(T[0], cmap='gray')
    axs[1].axis('off')
    axs[2].set_title('After Hysteresis')
    axs[2].imshow(result, cmap='gray')
    axs[2].axis('off')    
    plt.show()

def main():
    img1 = imread("images/lena.png", 0)
    img2 = imread("images/edge.tif", 0)
    img3 = imread("images/building.tif", 0)
    start(img1)
    start(img2)
    start(img3)
if __name__ == "__main__":
    main()
