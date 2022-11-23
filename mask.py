from random import randint
import numpy as np
import cv2
from math import *
import os

root = "/home/tuan/inpaint/workers"

def transform_mask(whiteHole, out_mask_path, raw_mask_path):
    img = cv2.imread(raw_mask_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    holePixelCount = 0
    for i in range (0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if np.array_equal(img[i, j], [0, 0, 0, 255]):
                img[i, j] = (0, 0, 0, 0) if whiteHole else (255, 255, 255, 255)
            else:
                img[i, j] = (255, 255, 255, 255) if whiteHole else (0, 0, 0, 255)
                holePixelCount += 1

    cv2.imwrite(out_mask_path, img)
    return holePixelCount / (img.shape[2] * img.shape[1])

if __name__ == "__main__":
    os.makedirs("dataset/irregular_mask2")
    for i in range(0, 32):
        filename = str(i).zfill(5) + ".png"
        print(filename)
        transform_mask(False, "dataset/irregular_mask2/" + filename, raw_mask_path="dataset/irregular_mask1/" + filename)
