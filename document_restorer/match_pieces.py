import cv2
import numpy as np

from document_restorer.edges.detect import EdgeDetectorFactory
from document_restorer.restore.match import VerticalShiftFragmentsConnector
from document_restorer.restore.match import HarrisFragmentsContentMatcher, SobelFragmentsContentMatcher
from document_restorer.restore.sequence import find_sequence, find_most_probable_sequence, restore_document
from restore.collect import FragmentsCollector

detector = EdgeDetectorFactory().createClosingMorphEdgeDetector(160, 9, True, 1)
connector = VerticalShiftFragmentsConnector(detector, [0])
content_matcher = HarrisFragmentsContentMatcher()


def match_pieces(piece1, piece2):
    connection = connector.connectFragments(piece1, piece2)
    content_match_result_1 = content_matcher.matchFragmentsContent(connection.stuck_fragments, connection.gap_line)
    return content_match_result_1, connection


def save_normalized(mat, name):
    img = np.zeros(mat.shape, np.uint8)
    img = cv2.normalize(mat, img, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imwrite(name, img)

fragments = FragmentsCollector(detector).collectFragments(
    cv2.cvtColor(cv2.imread('../resources/img/documents/4/fragments.bmp', 1), cv2.COLOR_BGR2GRAY)
)

print 'matching fragments...'
results_content = np.zeros([len(fragments), len(fragments), 1])
results_edges = np.zeros([len(fragments), len(fragments), 1])
connections = {}
for i in range(0, len(fragments)):
    for j in range(0, len(fragments)):
        # if (i, j) in [(0, 1), (1, 2), (1, 13), (2, 3), (3, 4), (4, 5), (1, 5), (2, 6), (3, 5), (4, 1)]:
        #     content_matcher.write_result_to = 'match_{0}_{1}.png'.format(i + 1, j + 1)
        #     need_print_result = True
        # else:
        #     content_matcher.write_result_to = None
        result = match_pieces(fragments[i], fragments[j])
        results_content[i, j] = result[0]
        results_edges[i, j] = result[1].adjacency
        connections['{0}-{1}'.format(i, j)] = result[1]

save_normalized(results_edges, 'edges_results.bmp')
save_normalized(results_content, 'content_results.bmp')

results_edges = cv2.normalize(results_edges, results_edges, 1.00, 0.00)
results_content = cv2.normalize(results_content, results_content, 1.00, 0.00)
results_content = np.full(results_content.shape, 1.00) - results_content
values = results_content * 0.2 + results_edges * 0.8
save_normalized(values, 'values.bmp')

print 'restoring fragments sequence...'
sequence = find_sequence(values)
print 'maximum cells method result: ' + str(sequence)
print 'probability tree method result: ' + str(find_most_probable_sequence(values))

cv2.imwrite('restored_document.bmp', restore_document(sequence, connections))
exit(0)
