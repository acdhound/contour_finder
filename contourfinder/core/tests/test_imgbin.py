from unittest import TestCase

import numpy as np

from contourfinder.core.imgbin import SimpleBinarizer


class TestSimpleBinarizer(TestCase):

    def __init__(self, methodName='runTest'):
        super(TestSimpleBinarizer, self).__init__(methodName)
        self.binarizer = SimpleBinarizer(127)

    def test_linear_gradient(self):
        img = np.zeros([100, 100, 1], np.uint8)
        for y in range(0, 99):
            for x in range(0, 99):
                img[y][x] = (y / 100.0) * 255.0

        binImg = self.binarizer.binarize(img)
        errMsg = 'binirized gradient expected to be {0} at point ({1}, {2})'
        for y in range(0, 40, 5):
            self.assertEqual(binImg[y][0], 0, errMsg.format('black', 0, y))
        for y in range(60, 99, 5):
            self.assertEqual(binImg[y][0], 255, errMsg.format('white', 0, y))
