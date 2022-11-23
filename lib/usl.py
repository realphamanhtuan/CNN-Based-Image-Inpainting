import cv2
import sys
import numpy as np
if len(sys.argv) > 1:
    filename = sys.argv[1]
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    factor = 2
    shape = (int(img.shape[0] * factor), int(img.shape[1] * factor), 4)
    out = np.zeros(shape)
    mask = np.multiply(np.ones(shape), 255)
    
    count = 0
    i = 0
    while i < shape[0]:
        j = 0
        while j < shape[1]:
            out[int(i)][int(j)] = img[int(i / factor)][int(j / factor)]
            mask[int(i)][int(j)] = (0, 0, 0, 0)
            count += 1
            j += factor
        i += factor

    print (img.shape, " -> ", out.shape)
    print ("original proportion:", count, "/", shape[0] * shape[1], " = ", count / shape[0] / shape[1])
    cv2.imwrite("mask.png", mask)
    cv2.imwrite("img.png", out)
