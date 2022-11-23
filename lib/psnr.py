import math
import os
import sys
import cv2
import numpy as np

def psnr(originalPath, contrastPath):
    original = cv2.imread(originalPath)
    contrast = cv2.imread(contrastPath)
    mse = np.mean((original - contrast) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR

if __name__ == "__main__":
    original = cv2.imread(sys.argv[1])
    contrast = cv2.imread(sys.argv[2], 1)
    print(f"PSNR value is {psnr(original, contrast)} dB")
