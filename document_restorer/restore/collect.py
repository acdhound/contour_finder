import cv2
import numpy as np


class Rect(object):

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def topLeft(self):
        return self.x, self.y

    def bottomRight(self):
        return self.x + self.w, self.y + self.h

    def ofImage(self, img):
        return img[self.y:self.y+self.h, self.x:self.x+self.w]


class Fragment(object):

    def __init__(self, img, contour, area, source_rect=None, source_contour=None):
        self.img = img
        self.contour = contour
        self.area = area
        self.source_rect = source_rect
        self.source_contour = source_contour


class FragmentsCollector(object):

    def __init__(self, edge_detector):
        self.edge_detector = edge_detector

    def collectFragments(self, img):
        binary = self.edge_detector.getArea(img)
        contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

        fragments = []
        for cont in contours:
            x, y, w, h = cv2.boundingRect(cont)
            rect = Rect(x, y, w, h)
            if rect.w < img.shape[1] / 2 or rect.h < img.shape[0] / 100:
                continue
            arr = np.full(cont.shape, [-x, -y], cont.dtype)
            contour = cont + arr
            fragments.append(Fragment(rect.ofImage(img), contour, rect.ofImage(binary), rect, cont))
        return fragments
