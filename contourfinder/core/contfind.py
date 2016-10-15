import cv2
from imgbin import SimpleBinarizer


class AbstractContourFinder:
    """Returns contour as a binary image"""

    def getContour(self, img):
        pass


class MorphContourFinder(AbstractContourFinder):
    """Applies the specified binarization algorithm to an image
    and calculates contour using morphological transformations"""

    def __init__(self, binarizer, kernel):
        self.binarizer = binarizer
        self.kernel = kernel

    def getContour(self, img):
        binImg = self.binarizer.binarize(img)
        erosedBinImg = cv2.erode(binImg, self.kernel, iterations = 1)
        return cv2.bitwise_and(cv2.bitwise_not(erosedBinImg), binImg)

    @staticmethod
    def withThresholdAndKernel(threshold, kernel):
        return MorphContourFinder(SimpleBinarizer(threshold), kernel)
