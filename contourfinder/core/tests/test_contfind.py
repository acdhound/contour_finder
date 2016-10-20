from unittest import TestCase
import numpy as np
from ..contfind import MorphContourFinder


class TestMorphContourFinder(TestCase):

    def __init__(self, methodName='runTest'):
        super(TestMorphContourFinder, self).__init__(methodName)
        self.contFinder = MorphContourFinder.withThresholdAndKernelInside(127, np.ones((3, 3), np.uint8))

    def test_square(self):
        img = np.zeros([100, 100, 1], np.uint8)
        img[50:80, 50:80] = np.ones([30, 30, 1]) * 200

        contImg = self.contFinder.getContour(img)

        self.assertEqual(contImg[0, 0], 0)
        self.assertEqual(contImg[50, 50], 255)
        self.assertEqual(contImg[79, 79], 255)
        self.assertEqual(contImg[50, 79], 255)
        self.assertEqual(contImg[79, 50], 255)
        self.assertEqual(contImg[65, 65], 0)
