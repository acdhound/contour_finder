import sys

import cv2
import numpy as np

from restore.collect import FragmentsCollector
from edges.detect import EdgeDetectorFactory

img_path = str(sys.argv[1])
threshold = int(sys.argv[2])
kernel_size = int(sys.argv[3])
inside = int(sys.argv[4]) == 1

img_colored = cv2.imread(img_path)
img = cv2.cvtColor(img_colored, cv2.COLOR_BGR2GRAY)

edge_detector = EdgeDetectorFactory().createClosingMorphEdgeDetector(threshold, kernel_size, inside)
collector = FragmentsCollector(edge_detector)
fragments = collector.collectFragments(img)
for f in fragments:
    # f_img = cv2.cvtColor(np.copy(f.img), cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(f_img, [f.contour], -1, (0, 255, 0), 1)
    # cv2.imshow('fragment', f_img)
    # cv2.waitKey(0)

    cv2.rectangle(img_colored, f.source_rect.topLeft(), f.source_rect.bottomRight(), (0, 0, 255), 1)
    cv2.drawContours(img_colored, [f.source_contour], -1, (0, 255, 0), 1)

cv2.imwrite('segmented.png', img_colored)
cv2.imshow('segmented', img_colored)
cv2.waitKey(0)

exit(0)