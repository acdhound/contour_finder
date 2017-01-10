import sys

import cv2

from contourfinder.core.detect import EdgeDetectorFactory

if __name__ != '__main__' or len(sys.argv) < 3:
    print 'invalid arguments'
    exit(1)

img = cv2.cvtColor(cv2.imread(str(sys.argv[1]), 1), cv2.COLOR_BGR2GRAY)
edgeDetectorFactory = EdgeDetectorFactory()

if str(sys.argv[2]) == 'canny':
    threshold1 = int(sys.argv[3])
    threshold2 = int(sys.argv[4])
    cv2.imwrite('edges_canny.bmp',
                edgeDetectorFactory.createCannyEdgeDetector(threshold1, threshold2).getEdges(img))
elif str(sys.argv[2]) == "morph":
    threshold = float(sys.argv[3])
    kSize = int(sys.argv[4])
    cv2.imwrite('edges_morph_outside.bmp',
                edgeDetectorFactory.createClosingMorphEdgeDetector(threshold, kSize, False).getEdges(img))
    cv2.imwrite('edges_morph_inside.bmp',
                edgeDetectorFactory.createClosingMorphEdgeDetector(threshold, kSize, True).getEdges(img))
else:
    raise Exception('unknown method: ' + str(sys.argv[2]))