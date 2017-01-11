import sys
import cv2
import os
from contourfinder.core.detect import EdgeDetectorFactory
from contourfinder.compare.compare import EdgeComparator


def read_gray_img(path):
    return cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2GRAY)


if __name__ != '__main__' or len(sys.argv) < 4:
    print 'invalid arguments'
    exit(1)

if str(sys.argv[1]) == 'quality':
    actual = read_gray_img(str(sys.argv[2]))
    expected = read_gray_img(str(sys.argv[3]))
    print 'Result: {0}'.format(EdgeComparator().compareFast(actual, expected)[0])
    exit(0)

imgPath = str(sys.argv[1])
img = read_gray_img(imgPath)
imgName = os.path.basename(imgPath)
edgeDetectorFactory = EdgeDetectorFactory()

if str(sys.argv[2]) == 'canny':
    threshold1 = int(sys.argv[3])
    threshold2 = int(sys.argv[4])
    cv2.imwrite(imgName + '__edges_canny.bmp',
                edgeDetectorFactory.createCannyEdgeDetector(threshold1, threshold2).getEdges(img))
elif str(sys.argv[2]) == "morph":
    threshold = float(sys.argv[3])
    kSize = int(sys.argv[4])
    cv2.imwrite(imgName + '__edges_morph_outside.bmp',
                edgeDetectorFactory.createClosingMorphEdgeDetector(threshold, kSize, False).getEdges(img))
    cv2.imwrite(imgName + '__edges_morph_inside.bmp',
                edgeDetectorFactory.createClosingMorphEdgeDetector(threshold, kSize, True).getEdges(img))
else:
    raise Exception('unknown method: ' + str(sys.argv[2]))