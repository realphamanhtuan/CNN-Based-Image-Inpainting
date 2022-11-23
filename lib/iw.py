import cv2
import sys
import numpy as np
if len(sys.argv) > 1:
    filename = sys.argv[1]
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    padding = 0.05
    padding = int(img.shape[1] * padding)
    shape = (img.shape[0] + 2 * padding, img.shape[1] + 2 * padding, 4)
    out = np.multiply(np.ones(shape), 255)
    mask = np.multiply(np.ones(shape), 255)
    
    count = 0
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            out[i + padding][j + padding] = img[i][j]
            mask[i + padding][j + padding] = (0, 0, 0, 0)
            count += 1
    print (img.shape, " -> ", out.shape)
    print ("original proportion:", count, "/", shape[0] * shape[1], " = ", count / shape[0] / shape[1])
    cv2.imwrite("mask.png", mask)
    cv2.imwrite("img.png", out)
