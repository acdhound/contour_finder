from unittest import TestCase

import numpy as np

from document_restorer.edges.compare import EdgeComparatorFactory


class TestEdgeComparator(TestCase):

    def createSquareEdges(self, thickness):
        img = np.zeros([40, 40, 1], np.uint8)
        img[20:30, 20:30] = np.ones([10, 10, 1]) * 255
        img[20+thickness:30-thickness, 20+thickness:30-thickness] = np.zeros([10-2*thickness, 10-2*thickness, 1])
        return img

    def test_no_edge_expected(self):
        sq = self.createSquareEdges(1)
        empty = np.zeros([40, 40, 1], np.uint8)
        result = EdgeComparatorFactory().create().compare(sq, empty)
        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[1], 0.0)

    def test_similar_edge_expected(self):
        sq1 = self.createSquareEdges(1)
        sq2 = self.createSquareEdges(1)
        result = EdgeComparatorFactory().create().compare(sq1, sq2)
        self.assertEqual(result[0], 1.0)
        self.assertEqual(result[1], 1.0)

    def test_thicker_edge_expected(self):
        sq1 = self.createSquareEdges(1)
        sq2 = self.createSquareEdges(4)
        result = EdgeComparatorFactory().create().compare(sq1, sq2)
        self.assertEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], 0.3, None, None, 0.1)

    def test_thinner_edge_expected(self):
        sq1 = self.createSquareEdges(4)
        sq2 = self.createSquareEdges(1)
        result = EdgeComparatorFactory().create().compare(sq1, sq2)
        self.assertAlmostEqual(result[0], 0.3, None, None, 0.1)
        self.assertEqual(result[1], 1.0)
