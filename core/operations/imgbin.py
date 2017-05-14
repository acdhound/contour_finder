import cv2
import numpy as np


class Binaizer(object):
    """Converts an image to binary."""

    def binarize(selfself, img):
        pass


class ThresholdBinarizer(Binaizer):
    """Uses simple rule to binarize every pixel of an image: if intensity is less than threshold - set 0, else - set max value.
    Calculation of threshold and maximum value must be implemented in subclasses. Accepts only single channel 8bit images."""

    def _calcThreshold(self, img):
        pass

    def _calcMaxValue(self, img):
        pass

    def binarize(self, img):
        if (len(img.shape) > 2 and img.shape[2] > 1) or img.dtype != np.uint8:
            raise Exception('single UINT8 channel image expected')
        return cv2.threshold(img, self._calcThreshold(img), self._calcMaxValue(img), cv2.THRESH_BINARY)[1]


class SimpleBinarizer(ThresholdBinarizer):
    """Converts image to black-white with specified threshold value."""

    def __init__(self, threshold):
        self.__threshold = threshold

    def _calcThreshold(self, img):
        return self.__threshold

    def _calcMaxValue(self, img):
        return 255


class MorphCloseBinarizer(SimpleBinarizer):

    def __init__(self, threshold, ksize):
        super(MorphCloseBinarizer, self).__init__(threshold)
        self.__kernel = np.ones((ksize, ksize), np.uint8)

    def binarize(self, img):
        bin = super(MorphCloseBinarizer, self).binarize(img)
        return cv2.morphologyEx(bin, cv2.MORPH_CLOSE, self.__kernel)
