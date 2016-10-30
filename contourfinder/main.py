import sys

import cv2

from contourfinder.core.edge import MorphEdgeDetector, CannyEdgeDetector, ClosingMorphEdgeDetector

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
            MorphEdgeDetector.withThreshold(threshold, True).getEdges(img))
cv2.imwrite('image_contour_morph_inside_closing.jpg',
            ClosingMorphEdgeDetector.withThresholdAndKSize(threshold, kSize, True).getEdges(img))
cv2.imwrite('image_contour_morph_outside.jpg',
            MorphEdgeDetector.withThreshold(threshold, False).getEdges(img))
cv2.imwrite('image_contour_morph_outside_closing.jpg',
            ClosingMorphEdgeDetector.withThresholdAndKSize(threshold, kSize, False).getEdges(img))
cv2.imwrite('image_contour_canny.jpg',
            CannyEdgeDetector.withThresholds(cannyThreshold1, cannyThreshold2).getEdges(img))

cv2.waitKey(0)
cv2.destroyAllWindows()