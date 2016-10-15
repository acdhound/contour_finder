import sys

import cv2
import numpy as np

from contourfinder.core.contfind import MorphContourFinder

if __name__ != "__main__" or len(sys.argv) < 4:
    print "invalid arguments"
    exit(1)

threshold = float(sys.argv[2])
kSize = int(sys.argv[3])

img = cv2.imread(sys.argv[1], 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((kSize, kSize), np.uint8)
imgCont = MorphContourFinder.withThresholdAndKernel(threshold, kernel).getContour(img)
cv2.imwrite('image_contour.jpg', imgCont)

cv2.waitKey(0)
cv2.destroyAllWindows()