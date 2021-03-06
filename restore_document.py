import cv2
import numpy as np
import sys

from core.operations.imgbin import MorphCloseBinarizer
from core.restore.match import VerticalShiftFragmentsConnector, HarrisFragmentsContentMatcher, SobelFragmentsContentMatcher
from core.restore.sequence import find_sequence, find_most_probable_sequence, restore_document
from core.restore.collect import FragmentsCollector

img_path = str(sys.argv[1])
threshold = int(sys.argv[2])
kernel_size = int(sys.argv[3])
method = str(sys.argv[4])
if len(sys.argv) < 6:
    alpha, beta = 0.8, 0.2
else:
    alpha, beta = float(sys.argv[5]), float(sys.argv[6])

binarizer = MorphCloseBinarizer(threshold, kernel_size)
connector = VerticalShiftFragmentsConnector()
if method == 'sobel':
    content_matcher = SobelFragmentsContentMatcher()
elif method == 'harris':
    content_matcher = HarrisFragmentsContentMatcher()
else:
    raise Exception('Unknown method ' + method)


def match_pieces(piece1, piece2):
    connection = connector.connectFragments(piece1, piece2)
    content_match_result_1 = content_matcher.matchFragmentsContent(connection.stuck_fragments, connection.gap_line)
    return content_match_result_1, connection


def save_normalized(mat, name):
    img = np.zeros(mat.shape, np.uint8)
    img = cv2.normalize(mat, img, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imwrite(name, img)

print 'collecting fragments...'
fragments = FragmentsCollector(binarizer).collectFragments(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))

print 'matching fragments...'
results_content = np.zeros([len(fragments), len(fragments), 1])
results_edges = np.zeros([len(fragments), len(fragments), 1])
connections = {}
for i in range(0, len(fragments)):
    for j in range(0, len(fragments)):
        if (i, j) == (26, 15):
            content_matcher.write_result_to = 'match_{0}_{1}.png'.format(i + 1, j + 1)
            need_print_result = True
        else:
            content_matcher.write_result_to = None
        result = match_pieces(fragments[i], fragments[j])
        results_content[i, j] = -1 * result[0]
        results_edges[i, j] = result[1].adjacency
        connections['{0}-{1}'.format(i, j)] = result[1]

save_normalized(results_edges, 'edges_results.bmp')
save_normalized(results_content, 'content_results.bmp')

results_edges = cv2.normalize(results_edges, results_edges, 1.00, 0.00)
results_content = cv2.normalize(results_content, results_content, 1.00, 0.00)
values = results_edges * alpha + results_content * beta
save_normalized(values, 'values.bmp')

print 'restoring fragments sequence...'
sequence = find_sequence(values)
print 'maximum cells method result: ' + str(sequence)
print 'probability tree method result: ' + str(find_most_probable_sequence(values))

restored = restore_document(sequence, connections)
cv2.imshow('restored', restored)
cv2.waitKey(0)
cv2.imwrite('restored_document.png', restored)
exit(0)
