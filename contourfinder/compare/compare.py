import cv2


class EdgeComparator(object):
    """Compare expected and actual edges. Comparison result is a tuple consisting of:
        - portion of actual edge points that coincide with expected
        - portion of expected edge points that coincide with actual"""

    def compare(self, actual, expected):
        coincident_edges_img = cv2.bitwise_and(actual, expected)

        (coinc_cnt, act_cnt, exp_cnt) = (0, 0, 0)
        for i in range(coincident_edges_img.shape[0]):
            for j in range(coincident_edges_img.shape[1]):
                coinc_cnt += (1 if coincident_edges_img[i][j] > 0 else 0)
                act_cnt += (1 if actual[i][j] > 0 else 0)
                exp_cnt += (1 if expected[i][j] > 0 else 0)

        return (self.actualCoincidentPortion(coinc_cnt, act_cnt, exp_cnt),
                self.expectedCoincidentPortion(coinc_cnt, act_cnt, exp_cnt))

    def actualCoincidentPortion(self, coincident, actual, expected):
        if actual == 0:
            return 1.0 if expected == 0 else 0.0
        return float(coincident) / float(actual)

    def expectedCoincidentPortion(self, coincident, actual, expected):
        if expected == 0:
            return 1.0 if actual == 0 else 0.0
        return float(coincident) / float(expected)
