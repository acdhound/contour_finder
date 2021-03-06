from unittest import TestCase
from ..util import copy_to
import numpy as np


class TestCopyTo(TestCase):

    def __init__(self, methodName='run_test'):
        super(TestCopyTo, self).__init__(methodName)

    def test_copy_larger_rect_to_top_left(self):
        rect1 = np.ones([5, 5, 1], np.uint8)
        rect2 = np.zeros([10, 10, 1], np.uint8)
        result = copy_to(rect2, rect1, 0, 0)
        self.assertEqual(np.count_nonzero(result), 0)

    def test_copy_smaller_rect_to_top_left(self):
        rect1 = np.zeros([5, 5, 1], np.uint8)
        rect2 = np.ones([10, 10, 1], np.uint8)
        result = copy_to(rect1, rect2, 0, 0)
        self.assertEqual(np.count_nonzero(result[0:5, 0:5]), 0)
        self.assertEqual(np.count_nonzero(result), 75)

    def test_copy_smaller_rect_inside(self):
        rect1 = np.zeros([5, 5, 1], np.uint8)
        rect2 = np.ones([10, 10, 1], np.uint8)
        result = copy_to(rect1, rect2, 2, 3)
        self.assertEqual(np.count_nonzero(result[3:8, 2:7]), 0)
        self.assertEqual(np.count_nonzero(result), 75)

    def test_copy_rect_with_cropping_at_right_bottom(self):
        rect1 = np.ones([5, 5, 1], np.uint8)
        rect2 = np.zeros([5, 5, 1], np.uint8)
        result = copy_to(rect2, rect1, 2, 2)
        self.assertEqual(np.count_nonzero(result[2:, 2:]), 0)
        self.assertEqual(np.count_nonzero(result), 16)

    def test_copy_rect_with_cropping_at_left_top(self):
        rect1 = np.ones([5, 5, 1], np.uint8)
        rect2 = np.zeros([5, 5, 1], np.uint8)
        result = copy_to(rect2, rect1, -2, -1)
        self.assertEqual(np.count_nonzero(result[:4, :3]), 0)
        self.assertEqual(np.count_nonzero(result), 13)

    def test_copy_rect_outside_at_left_top(self):
        rect1 = np.ones([5, 5, 1], np.uint8)
        rect2 = np.zeros([5, 5, 1], np.uint8)
        self.assertRaises(Exception, lambda: copy_to(rect2, rect1, -5, -5))

    def test_copy_rect_outside_at_right_bottom(self):
        rect1 = np.ones([5, 5, 1], np.uint8)
        rect2 = np.zeros([5, 5, 1], np.uint8)
        self.assertRaises(Exception, lambda: copy_to(rect2, rect1, 5, 5))
