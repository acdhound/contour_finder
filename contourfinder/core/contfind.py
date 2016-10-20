import cv2
from imgbin import SimpleBinarizer
import numpy as np


class AbstractContourFinder:
    """Returns contour as a binary image"""

    def getContour(self, img):
        pass


class MorphContourFinder(AbstractContourFinder):
    """Applies the specified binarization algorithm to an image
    and calculates contour using morphological transformations"""

    def __init__(self, binarizer, kernel=np.ones((3, 3), np.uint8), inside=True):
        self.binarizer = binarizer
        self.kernel = kernel
        self.inside = inside

    def getContour(self, img):
        binImg = self.binarizer.binarize(img)
        if self.inside:
            erosedBinImg = cv2.erode(binImg, self.kernel, iterations=1)
            return cv2.bitwise_and(cv2.bitwise_not(erosedBinImg), binImg)
        else:
            dilatedBinImg = cv2.dilate(binImg, self.kernel, iterations=1)
            return cv2.bitwise_and(dilatedBinImg, cv2.bitwise_not(binImg))

    @staticmethod
    def withThresholdInside(threshold):
        return MorphContourFinder(SimpleBinarizer(threshold), inside=True)

    @staticmethod
    def withThresholdOutside(threshold):
        return MorphContourFinder(SimpleBinarizer(threshold), inside=False)


class CannyContourFinder(AbstractContourFinder):
    """Calculates contour using Canny edge detection algorithm"""

    def __init__(self, low, high, ksize=3, presice=False):
        self.low = low
        self.high = high
        self.ksize = ksize
        self.presice = presice

    def getContour(self, img):
        return cv2.Canny(image=img, threshold1=self.low, threshold2=self.high,
            apertureSize=self.ksize, L2gradient=self.presice)

    @staticmethod
    def withThresholds(low, high):
        return CannyContourFinder(low, high)
