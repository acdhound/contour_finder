import cv2


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


class ImageSegmenter(object):

    def __init__(self, edge_detector):
        self.edge_detector = edge_detector

    def segment(self, img):
        binary_closed = self.edge_detector.getArea(img)
        contours = cv2.findContours(binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        rectangles = []
        for cont in contours:
            x, y, w, h = cv2.boundingRect(cont)
            rect_candidate = Rect(x, y, w, h)
            if rect_candidate.w < img.shape[1] / 2 or rect_candidate.h < img.shape[0] / 100:
                continue
            rectangles.append(rect_candidate)
        return rectangles
