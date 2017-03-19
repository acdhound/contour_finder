import cv2
import numpy as np

from document_restorer.operations.imgbin import SimpleBinarizer


class EdgeDetector(object):
    """Returns edges as a binary image"""

    def getEdges(self, img):
        pass

    def getArea(self, img):
        pass


class MorphEdgeDetector(EdgeDetector):
    """Applies the specified binarization algorithm to an image
    and finds edge points using morphological transformations"""

    def __init__(self, binarizer, inside=True, thickness=1):
        self.__binarizer = binarizer
        self.inside = inside
        self.thickness = thickness

    def getEdges(self, img):
        area = self.getArea(img)
        kernel = np.ones((self.thickness, self.thickness), np.uint8)
        # maybe use cross kernel instead of box?
        # kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        if self.inside:
            erosedArea = cv2.erode(area, kernel, iterations=1)
            return cv2.bitwise_and(cv2.bitwise_not(erosedArea), area)
        else:
            dilatedArea = cv2.dilate(area, kernel, iterations=1)
            return cv2.bitwise_and(dilatedArea, cv2.bitwise_not(area))

    def getArea(self, img):
        return self.__binarizer.binarize(img)


class ClosingMorphEdgeDetector(MorphEdgeDetector):
    """Applies closing morphological operation with specified kernel
    to binary image before finding contours"""

    def __init__(self, kernel, binarizer, inside=True, thickness=1):
        super(ClosingMorphEdgeDetector, self).__init__(binarizer, inside, thickness)
        self.kernel = kernel

    def getArea(self, img):
        area = MorphEdgeDetector.getArea(self, img)
        return cv2.morphologyEx(area, cv2.MORPH_CLOSE, self.kernel)


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

    def getArea(self, img):
        raise NotImplementedError('method not supported')


class EdgeDetectorFactory(object):

    def createMorphEdgeDetector(self, threshold, inside=True):
        return MorphEdgeDetector(SimpleBinarizer(threshold), inside)

    def createClosingMorphEdgeDetector(self, threshold, ksize, inside=True, thickness=1):
        return ClosingMorphEdgeDetector(
            np.ones((ksize, ksize), np.uint8), SimpleBinarizer(threshold), inside, thickness)

    def createCannyEdgeDetector(self, low, high):
        return CannyEdgeDetector(low, high)