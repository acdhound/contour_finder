import cv2
from contourfinder.core.detect import EdgeDetectorFactory
from contourfinder.core.match import BinaryEdgeMatcher


img1 = cv2.cvtColor(cv2.imread('../resources/img/piece2.png', 1), cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread('../resources/img/piece3.png', 1), cv2.COLOR_BGR2GRAY)

detector = EdgeDetectorFactory().createClosingMorphEdgeDetector(100, 10)
matcher = BinaryEdgeMatcher(detector)
result = matcher.matchEdges(img1, img2)

print result[0], result[1]
cv2.imshow('gap line', result[2])
cv2.waitKey(0)
