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

    def __init__(self, binarizer, inside=True):
        self.binarizer = binarizer
        self.inside = inside

    def getBinContour(self, binImg):
        kernel = np.ones((3, 3), np.uint8)
        if self.inside:
            erosedBinImg = cv2.erode(binImg, kernel, iterations=1)
            return cv2.bitwise_and(cv2.bitwise_not(erosedBinImg), binImg)
        else:
            dilatedBinImg = cv2.dilate(binImg, kernel, iterations=1)
            return cv2.bitwise_and(dilatedBinImg, cv2.bitwise_not(binImg))

    def getContour(self, img):
        binImg = self.binarizer.binarize(img)
        return self.getBinContour(binImg)

    @staticmethod
    def withThreshold(threshold, inside=True):
        return MorphContourFinder(SimpleBinarizer(threshold), inside)


class ClosingMorphContourFinder(MorphContourFinder):
    """Applies closing morphological operation with specified kernel
    to binary image before finding contours"""

    def __init__(self, kernel, binarizer, inside=True):
        MorphContourFinder.__init__(self, binarizer, inside)
        self.kernel = kernel

    def getBinContour(self, binImg):
        closedBinImg = cv2.morphologyEx(binImg, cv2.MORPH_CLOSE, self.kernel)
        return MorphContourFinder.getBinContour(self, closedBinImg)

    @staticmethod
    def withThresholdAndKSize(threshold, ksize, inside=True):
        return ClosingMorphContourFinder(
            np.ones((ksize, ksize), np.uint8), SimpleBinarizer(threshold), inside)


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
