import cv2
import numpy as np


class AbstractBinarizer:
    """Convert image to binary"""

    def isImgValid(self, img):
        pass

    def calcThreshold(self, img):
        pass

    def calcMaxValue(self, img):
        pass

    def binarize(self, img):
        self.isImgValid(img)
        return cv2.threshold(img, self.calcThreshold(img), self.calcMaxValue(img), cv2.THRESH_BINARY)[1]


class SimpleBinarizer(AbstractBinarizer):
    """Simply binarize by specified threshold. Accepts only 1 byte grey images"""

    def __init__(self, threshold):
        self.threshold = threshold

    def isImgValid(self, img):
        if (len(img.shape) > 2 and img.shape[2] > 1) or img.dtype != np.uint8:
            raise TypeError('single UINT8 channel image expected')

    def calcThreshold(self, img):
        return self.threshold

    def calcMaxValue(self, img):
        return 255



