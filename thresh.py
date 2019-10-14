from cv2 import *
import numpy as np
import matplotlib.pyplot as plt
import fgrad, nmaxsup

def thresh(img):
    T_h = img.max()*.08
    T_l = img.max()*.04
    w, s = 50, 255

    result = np.zeros(img.shape)
    si, sj = np.where(img > T_h)
    wi, wj = np.where((img < T_h) & (img > T_l))
    
    result[si, sj] = s
    result[wi, wj] = w

    return (result, w, s)

def main():
    img = imread("lena.png", 0)
    F, D = fgrad.gradient(img)
    sup = nmaxsup.suppress(F, D)
    T = thresh(sup)
    f = plt.figure(figsize=(5,5))
    plt.imshow(T[0], cmap='gray')
    plt.show()
if __name__ == "__main__":
    main()