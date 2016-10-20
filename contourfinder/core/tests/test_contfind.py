from unittest import TestCase
import numpy as np
from ..contfind import MorphContourFinder


class TestMorphContourFinder(TestCase):

    def createSquare(self):
        img = np.zeros([100, 100, 1], np.uint8)
        img[50:80, 50:80] = np.ones([30, 30, 1]) * 200
        return img

    def assertPointValue(self, img, x, y, value):
        self.assertEqual(img[y, x], value, "unexpected value of pixel ({0},{1})".format(x, y))

    def assertSquareContour(self, img, x0, y0, d, value):
        for y in range(y0, y0 + d):
            self.assertPointValue(img, x0, y, 255)
            self.assertPointValue(img, x0 + d, y, 255)
        for x in range(x0, x0 + d):
            self.assertPointValue(img, x, y0, 255)
            self.assertPointValue(img, x, y0 + d, 255)

    def test_square_inside(self):
        self.contFinder = MorphContourFinder.withThresholdInside(127)

        img = self.createSquare()
        contImg = self.contFinder.getContour(img)

        self.assertSquareContour(contImg, 50, 50, 29, 255)

    def test_square_outside(self):
        self.contFinder = MorphContourFinder.withThresholdOutside(127)

        img = self.createSquare()
        contImg = self.contFinder.getContour(img)

        self.assertSquareContour(contImg, 49, 49, 31, 255)
