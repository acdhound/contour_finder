import cv2
from imgbin import SimpleBinarizer


class AbstractContourFinder:
    """Returns contour as a binary image"""

    def getContour(self, img):
        pass


class MorphContourFinder(AbstractContourFinder):
    """Applies the specified binarization algorithm to an image
    and calculates contour using morphological transformations"""

    def __init__(self, binarizer, kernel, inside):
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
    def withThresholdAndKernelInside(threshold, kernel):
        return MorphContourFinder(SimpleBinarizer(threshold), kernel, 1)

    @staticmethod
    def withThresholdAndKernelOutside(threshold, kernel):
        return MorphContourFinder(SimpleBinarizer(threshold), kernel, 0)
