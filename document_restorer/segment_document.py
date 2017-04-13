import cv2
import sys
from edges.detect import EdgeDetectorFactory
from operations.segment import ImageSegmenter


img_path = str(sys.argv[1])
threshold = int(sys.argv[2])
kernel_size = int(sys.argv[3])
inside = int(sys.argv[4]) == 1

img_colored = cv2.imread(img_path)
img = cv2.cvtColor(img_colored, cv2.COLOR_BGR2GRAY)

edge_detector = EdgeDetectorFactory().createClosingMorphEdgeDetector(threshold, kernel_size, inside)
segmenter = ImageSegmenter(edge_detector)
segments = segmenter.segment(img)
for s in segments:
    cv2.rectangle(img_colored, s.topLeft(), s.bottomRight(), (0, 255, 0), 1)

cv2.imwrite('segmented.png', img_colored)
cv2.imshow('segmented', img_colored)
cv2.waitKey(0)

exit(0)