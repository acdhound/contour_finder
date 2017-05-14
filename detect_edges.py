import os
import sys

import cv2

from core.edges.compare import EdgeComparatorFactory
from core.edges.detect import EdgeDetectorFactory

edgeComparator = EdgeComparatorFactory().create()


def read_gray_img(path):
    return cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2GRAY)


def save_quality_index(actual_edges, expected_edges, file_name='result'):
    result = edgeComparator.compare(actual_edges, expected_edges)[0]
    f = open(file_name + '.txt', 'w')
    f.write(str(result))
    f.close()


def save_parameters(file_name, param_values):
    f = open(file_name + '.txt', 'w')
    f.write(str(param_values))
    f.close()


imgPath = str(sys.argv[1])
img = read_gray_img(imgPath)
imgName = os.path.basename(imgPath)
edgeDetectorFactory = EdgeDetectorFactory()

if str(sys.argv[2]) == 'canny':
    threshold1 = int(sys.argv[3])
    threshold2 = int(sys.argv[4])
    save_parameters(imgName + '__parameters_canny', {'threshold1': threshold1, 'threshold2': threshold2})

    canny_edges = edgeDetectorFactory.createCannyEdgeDetector(threshold1, threshold2).getEdges(img)
    cv2.imwrite(imgName + '__edges_canny.bmp', canny_edges)

    expected_edge_area = read_gray_img(str(sys.argv[5]))
    save_quality_index(canny_edges, expected_edge_area, imgName + '__result_canny')

    exit(0)
elif str(sys.argv[2]) == "morph":
    threshold = float(sys.argv[3])
    kSize = int(sys.argv[4])
    save_parameters(imgName + '__parameters_morph', {'threshold': threshold, 'kSize': kSize})

    morph_edges_outside = edgeDetectorFactory.createClosingMorphEdgeDetector(threshold, kSize, False).getEdges(img)
    cv2.imwrite(imgName + '__edges_morph_outside.bmp', morph_edges_outside)

    morph_edges_inside = edgeDetectorFactory.createClosingMorphEdgeDetector(threshold, kSize, True).getEdges(img)
    cv2.imwrite(imgName + '__edges_morph_inside.bmp', morph_edges_inside)

    expected_edge_area = read_gray_img(str(sys.argv[5]))
    save_quality_index(morph_edges_outside, expected_edge_area, imgName + '__result_morph_outside')
    save_quality_index(morph_edges_inside, expected_edge_area, imgName + '__result_morph_inside')

    exit(0)
else:
    raise Exception('unknown method: ' + str(sys.argv[2]))