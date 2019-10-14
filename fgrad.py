from cv2 import *
import numpy as np
import matplotlib.pyplot as plt

# Computes the filtered gradient on a given image
# 1. Load an image
# 2. Convolve the image with a Gaussian
# 3. Find the x and y components of the gradient Fx and Fy at each point.
# 4. Compute the edge strength F (magnitude of gradient) and edge orientation D = arctan(Fy/Fx) at each pixel
# (Steps 2 and 3 may be combined by convolving the image with the derivative of a Gaussian)
def gradient(img):
    blur = GaussianBlur(img, (5,5), 0)
    kernelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    kernely = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    sobelx = filter2D(blur, CV_64F, kernel=kernelx)
    sobely = filter2D(blur, CV_64F, kernel=kernely)
    
    F = np.hypot(sobelx, sobely, casting="same_kind")
    D = np.arctan2(sobely, sobelx)
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
    axs[0,0].set_title('Fx')
    axs[0,0].imshow(sobelx, cmap='gray')
    axs[0,0].axis('off')
    axs[0,1].set_title('Fy')
    axs[0,1].imshow(sobely, cmap='gray')
    axs[0,1].axis('off')
    axs[1,0].set_title('F')
    axs[1,0].imshow(F, cmap='gray')
    axs[1,0].axis('off')    
    axs[1,1].set_title('D')
    axs[1,1].imshow(D, cmap='gray')
    axs[1,1].axis('off')        
    plt.show()
    # print(D)
    return F, D

def main():
    img = imread("lena.png", 0)
    F, D = gradient(img)
    
if __name__ == "__main__":
    main()