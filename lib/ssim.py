from skimage.metrics import structural_similarity
import cv2
import numpy as np

def ssim(originalPath, contrastPath):
    before = cv2.imread(originalPath)
    after = cv2.imread(contrastPath)

    # Convert images to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    return score

if __name__ == "__main__":
    import sys
    print(ssim(sys.argv[1], sys.argv[2]))
