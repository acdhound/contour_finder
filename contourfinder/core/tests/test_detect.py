from unittest import TestCase
import numpy as np
from ..detect import MorphEdgeDetector, ClosingMorphEdgeDetector


class TestMorphEdgeDetector(TestCase):

    def createSquare(self):
        img = np.zeros([100, 100, 1], np.uint8)
        img[50:80, 50:80] = np.ones([30, 30, 1]) * 200
        return img

    def assertPointValue(self, img, x, y, value):
        self.assertEqual(img[y, x], value,
                         "incorrect value of pixel ({0},{1}): expected: {2}, actual: {3}"
                            .format(x, y, value, img[y, x]))

    def assertSquareContour(self, img, x0, y0, d, value):
        for y in range(y0, y0 + d):
            self.assertPointValue(img, x0, y, value)
            self.assertPointValue(img, x0 + d, y, value)
        for x in range(x0, x0 + d):
            self.assertPointValue(img, x, y0, value)
            self.assertPointValue(img, x, y0 + d, value)

    def test_square_inside(self):
        self.contFinder = MorphEdgeDetector.withThreshold(127, True)
        img = self.createSquare()
        contImg = self.contFinder.getEdges(img)
        self.assertSquareContour(contImg, 50, 50, 29, 255)

    def test_square_outside(self):
        self.contFinder = MorphEdgeDetector.withThreshold(127, False)
        img = self.createSquare()
        contImg = self.contFinder.getEdges(img)
        self.assertSquareContour(contImg, 49, 49, 31, 255)

    def test_square_with_hole(self):
        self.contFinder = MorphEdgeDetector.withThreshold(127, True)
        img = self.createSquare()
        img[60:65, 60:65] = np.zeros([5, 5, 1])
        contImg = self.contFinder.getEdges(img)
        self.assertSquareContour(contImg, 50, 50, 29, 255)
        self.assertSquareContour(contImg, 59, 59, 6, 255)

    def test_closing_square_with_hole(self):
        self.contFinder = ClosingMorphEdgeDetector.withThresholdAndKSize(127, 11, True)
        img = self.createSquare()
        img[60:65, 60:65] = np.zeros([5, 5, 1])
        contImg = self.contFinder.getEdges(img)
        self.assertSquareContour(contImg, 50, 50, 29, 255)
        self.assertSquareContour(contImg, 59, 59, 6, 0)
