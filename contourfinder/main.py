import sys

import cv2

from contourfinder.core.contfind import MorphContourFinder, CannyContourFinder, ClosingMorphContourFinder

if __name__ != "__main__" or len(sys.argv) < 6:
    print "invalid arguments"
    exit(1)

imgPathName = str(sys.argv[1])
threshold = float(sys.argv[2])
kSize = int(sys.argv[3])
cannyThreshold1 = int(sys.argv[4])
cannyThreshold2 = int(sys.argv[5])

img = cv2.imread(imgPathName, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('image_contour_morph_inside.jpg',
            MorphContourFinder.withThreshold(threshold, True).getContour(img))
cv2.imwrite('image_contour_morph_inside_closing.jpg',
            ClosingMorphContourFinder.withThresholdAndKSize(threshold, kSize, True).getContour(img))
cv2.imwrite('image_contour_morph_outside.jpg',
            MorphContourFinder.withThreshold(threshold, False).getContour(img))
cv2.imwrite('image_contour_morph_outside_closing.jpg',
            ClosingMorphContourFinder.withThresholdAndKSize(threshold, kSize, False).getContour(img))
cv2.imwrite('image_contour_canny.jpg',
            CannyContourFinder.withThresholds(cannyThreshold1, cannyThreshold2).getContour(img))

cv2.waitKey(0)
cv2.destroyAllWindows()