import cv2
import sys
import numpy as np
if len(sys.argv) > 1:
    filename = sys.argv[1]
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    mask = np.zeros_like(img)
    def remove_pixel(i, j):
        img[i][j] = (0, 0, 0, 255)
        mask[i][j] = (255, 255, 255, 255)
 
    for i in range(0, img.shape[0], 2):
        for j in range(0, img.shape[1], 2):
            remove_pixel(i, j)
    print (img.shape)
    cv2.imwrite("mask.png", mask)
    cv2.imwrite("img.png", img)
