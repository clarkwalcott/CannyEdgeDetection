from cv2 import *
import numpy as np

# Computes the filtered gradient on a given image
# 1. Load an image
# 2. Convolve the image with a Gaussian
# 3. Find the x and y components of the gradient Fx and Fy at each point.
# 4. Compute the edge strength F (magnitude of gradient) and edge orientation D = arctan(Fy/Fx) at each pixel
# (Steps 2 and 3 may be combined by convolving the image with the derivative of a Gaussian)



def main():
    img = imread("lena.png", 0)
    
    blur = GaussianBlur(img, (5,5), 0)
    kernelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
    kernely = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32)
    
    sobelx = filter2D(blur, -1, kernelx)
    sobely = filter2D(blur, -1, kernely)

    F = np.hypot(sobelx, sobely)
    F = F / F.max() * 255
    D = np.arctan(sobely, sobelx)
    print(F)
    imshow('Gradient', F)
    waitKey(0)
    destroyAllWindows()
if __name__ == "__main__":
    main()