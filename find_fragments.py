import sys

import cv2

from core.restore.collect import FragmentsCollector
from core.operations.imgbin import MorphCloseBinarizer

img_path = str(sys.argv[1])
threshold = int(sys.argv[2])
kernel_size = int(sys.argv[3])

img_colored = cv2.imread(img_path)
img = cv2.cvtColor(img_colored, cv2.COLOR_BGR2GRAY)

binarizer = MorphCloseBinarizer(threshold, kernel_size)
collector = FragmentsCollector(binarizer)
fragments = collector.collectFragments(img)
img_colored = cv2.normalize(img_colored, img_colored, 60, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
for f in fragments:
    cv2.rectangle(img_colored, f.source_rect.topLeft(), f.source_rect.bottomRight(), (0, 0, 255), 1)
    cv2.drawContours(img_colored, [f.source_contour], -1, (0, 255, 0), 1)
    cv2.putText(img_colored, str(fragments.index(f)),
                (10, f.source_rect.y + f.source_rect.h/2 + 8), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255))

cv2.imwrite('fragments.png', img_colored)
cv2.imshow('fragments', img_colored)
cv2.waitKey(0)

exit(0)