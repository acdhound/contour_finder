import cv2
import numpy as np


class BaseEdgeComparator(object):
    """Compare expected and actual edges. Comparison result is a tuple consisting of:
        - portion of actual edge points that coincide with expected
        - portion of expected edge points that coincide with actual"""

    def compare(self, actual, expected):
        pass

    def _actualCoincidentPortion(self, coincident, actual, expected):
        if actual == 0:
            return 1.0 if expected == 0 else 0.0
        return float(coincident) / float(actual)

    def _expectedCoincidentPortion(self, coincident, actual, expected):
        if expected == 0:
            return 1.0 if actual == 0 else 0.0
        return float(coincident) / float(expected)


class DeprecatedEdgeComparator(BaseEdgeComparator):

    def compare(self, actual, expected):
        coincident_edges_img = cv2.bitwise_and(actual, expected)

        (coinc_cnt, act_cnt, exp_cnt) = (0, 0, 0)
        for i in range(coincident_edges_img.shape[0]):
            for j in range(coincident_edges_img.shape[1]):
                coinc_cnt += (1 if coincident_edges_img[i][j] > 0 else 0)
                act_cnt += (1 if actual[i][j] > 0 else 0)
                exp_cnt += (1 if expected[i][j] > 0 else 0)

        return (self._actualCoincidentPortion(coinc_cnt, act_cnt, exp_cnt),
                self._expectedCoincidentPortion(coinc_cnt, act_cnt, exp_cnt))


class EdgeComparator(BaseEdgeComparator):

    def compare(self, actual, expected):
        coincident_edges_img = cv2.bitwise_and(actual, expected)

        coinc_cnt = self.__nonzeroPxCount(coincident_edges_img)
        act_cnt = self.__nonzeroPxCount(actual)
        exp_cnt = self.__nonzeroPxCount(expected)

        return (self._actualCoincidentPortion(coinc_cnt, act_cnt, exp_cnt),
                self._expectedCoincidentPortion(coinc_cnt, act_cnt, exp_cnt))

    def __nonzeroPxCount(self, img):
        return np.nonzero(img)[0].shape[0]


class EdgeComparatorFactory(object):

    def create(self):
        return EdgeComparator()
