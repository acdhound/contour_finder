import cv2
import sys
from edges.detect import EdgeDetectorFactory


img_path = str(sys.argv[1])
threshold = int(sys.argv[2])
kernel_size = int(sys.argv[3])
inside = int(sys.argv[4]) == 1

edge_detector = EdgeDetectorFactory().createClosingMorphEdgeDetector(threshold, kernel_size, inside)
img = cv2.cvtColor(cv2.imread(img_path, 1), cv2.COLOR_BGR2GRAY)
contour = edge_detector.getEdges(img)
cv2.imshow('contour', contour)
cv2.waitKey(0)
cv2.imwrite('contour.png', contour)

exit(0)