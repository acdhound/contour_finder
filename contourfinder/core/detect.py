import cv2
from imgbin import SimpleBinarizer
import numpy as np


class EdgeDetector(object):
    """Returns edges as a binary image"""

    def getEdges(self, img):
        pass


class MorphEdgeDetector(EdgeDetector):
    """Applies the specified binarization algorithm to an image
    and finds edge points using morphological transformations"""

    def __init__(self, binarizer, inside=True):
        self.__binarizer = binarizer
        self.inside = inside

    def _getBinEdges(self, binImg):
        kernel = np.ones((3, 3), np.uint8)
        # maybe use cross kernel instead of box?
        # kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        if self.inside:
            erosedBinImg = cv2.erode(binImg, kernel, iterations=1)
            return cv2.bitwise_and(cv2.bitwise_not(erosedBinImg), binImg)
        else:
            dilatedBinImg = cv2.dilate(binImg, kernel, iterations=1)
            return cv2.bitwise_and(dilatedBinImg, cv2.bitwise_not(binImg))

    def getEdges(self, img):
        # cv2.imwrite("source.bmp", img)
        binImg = self.__binarizer.binarize(img)
        # cv2.imwrite("bin.bmp", binImg)
        return self._getBinEdges(binImg)


class ClosingMorphEdgeDetector(MorphEdgeDetector):
    """Applies closing morphological operation with specified kernel
    to binary image before finding contours"""

    def __init__(self, kernel, binarizer, inside=True):
        super(ClosingMorphEdgeDetector, self).__init__(binarizer, inside)
        self.kernel = kernel

    def _getBinEdges(self, binImg):
        closedBinImg = cv2.morphologyEx(binImg, cv2.MORPH_CLOSE, self.kernel)
        # cv2.imwrite("closed.bmp", closedBinImg)
        return MorphEdgeDetector._getBinEdges(self, closedBinImg)


class CannyEdgeDetector(EdgeDetector):
    """Calculates contour using Canny edge detection algorithm"""

    def __init__(self, low, high, ksize=3, presice=False):
        self.low = low
        self.high = high
        self.ksize = ksize
        self.presice = presice

    def getEdges(self, img):
        return cv2.Canny(image=img, threshold1=self.low, threshold2=self.high,
            apertureSize=self.ksize, L2gradient=self.presice)


class EdgeDetectorFactory(object):

    def createMorphEdgeDetector(self, threshold, inside=True):
        return MorphEdgeDetector(SimpleBinarizer(threshold), inside)

    def createClosingMorphEdgeDetector(self, threshold, ksize, inside=True):
        return ClosingMorphEdgeDetector(
            np.ones((ksize, ksize), np.uint8), SimpleBinarizer(threshold), inside)

    def createCannyEdgeDetector(self, low, high):
        return CannyEdgeDetector(low, high)