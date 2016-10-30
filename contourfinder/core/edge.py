import cv2
from imgbin import SimpleBinarizer
import numpy as np


class AbstractEdgeDetector:
    """Returns edges as a binary image"""

    def getEdges(self, img):
        pass


class MorphEdgeDetector(AbstractEdgeDetector):
    """Applies the specified binarization algorithm to an image
    and finds edge points using morphological transformations"""

    def __init__(self, binarizer, inside=True):
        self.binarizer = binarizer
        self.inside = inside

    def getBinEdges(self, binImg):
        kernel = np.ones((3, 3), np.uint8)
        if self.inside:
            erosedBinImg = cv2.erode(binImg, kernel, iterations=1)
            return cv2.bitwise_and(cv2.bitwise_not(erosedBinImg), binImg)
        else:
            dilatedBinImg = cv2.dilate(binImg, kernel, iterations=1)
            return cv2.bitwise_and(dilatedBinImg, cv2.bitwise_not(binImg))

    def getEdges(self, img):
        binImg = self.binarizer.binarize(img)
        return self.getBinEdges(binImg)

    @staticmethod
    def withThreshold(threshold, inside=True):
        return MorphEdgeDetector(SimpleBinarizer(threshold), inside)


class ClosingMorphEdgeDetector(MorphEdgeDetector):
    """Applies closing morphological operation with specified kernel
    to binary image before finding contours"""

    def __init__(self, kernel, binarizer, inside=True):
        MorphEdgeDetector.__init__(self, binarizer, inside)
        self.kernel = kernel

    def getBinEdges(self, binImg):
        closedBinImg = cv2.morphologyEx(binImg, cv2.MORPH_CLOSE, self.kernel)
        return MorphEdgeDetector.getBinEdges(self, closedBinImg)

    @staticmethod
    def withThresholdAndKSize(threshold, ksize, inside=True):
        return ClosingMorphEdgeDetector(
            np.ones((ksize, ksize), np.uint8), SimpleBinarizer(threshold), inside)


class CannyEdgeDetector(AbstractEdgeDetector):
    """Calculates contour using Canny edge detection algorithm"""

    def __init__(self, low, high, ksize=3, presice=False):
        self.low = low
        self.high = high
        self.ksize = ksize
        self.presice = presice

    def getEdges(self, img):
        return cv2.Canny(image=img, threshold1=self.low, threshold2=self.high,
            apertureSize=self.ksize, L2gradient=self.presice)

    @staticmethod
    def withThresholds(low, high):
        return CannyEdgeDetector(low, high)
