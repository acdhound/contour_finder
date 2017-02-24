import numpy as np
import cv2
from ..operations.util import count_nonzero, copy_to


class FragmentsConnector(object):
    """
        Tries to connect two fragments of the paper text document assuming the document was broken
    into horizontal strips. Analyses fragments' edges and calculates numerical coefficient representing
    fragments' adjacency.
    """

    def connectFragments(self, top_fragment, bottom_fragment):
        pass


class FragmentsConnection(object):

    def __init__(self, adjacency, stuck_fragments, gap_line):
        self.adjacency = adjacency
        self.stuck_fragments = stuck_fragments
        self.gap_line = gap_line


class VerticalShiftFragmentsConnector(FragmentsConnector):

    def __init__(self, edge_detector):
        self.edge_detector = edge_detector

    def connectFragments(self, top_fragment, bottom_fragment):
        if top_fragment.shape[1] != bottom_fragment.shape[1]:
            raise Exception("Pieces of the same width expected")

        top_edge = self.edge_detector.getEdges(top_fragment)
        bottom_edge = self.edge_detector.getEdges(bottom_fragment)

        top_edge_bottom = self.__get_bottom_half(top_edge)
        bottom_edge_top = self.__get_top_half(bottom_edge)
        im1 = np.zeros([top_edge_bottom.shape[0] + bottom_edge_top.shape[0], top_edge.shape[1]], np.uint8)
        copy_to(top_edge_bottom, im1, 0, 0)

        max_adjacency, delta_for_max, gap_lines = 0, 0, None

        for delta in range(-bottom_edge_top.shape[0] + 1, top_edge_bottom.shape[0]):
            im2 = np.zeros([im1.shape[0], im1.shape[1]], np.uint8)
            copy_to(bottom_edge_top, im2, 0, delta)
            common_edge = cv2.bitwise_and(im1, im2)
            adjacency = float(count_nonzero(common_edge)) / float(common_edge.shape[1])
            if adjacency > max_adjacency:
                max_adjacency = adjacency
                delta_for_max = delta
                gap_lines = [im1, im2]
            elif gap_lines is None:
                gap_lines = [im1, im2]

        top_fragment_bottom = self.__get_bottom_half(top_fragment)
        bottom_fragment_top = self.__get_top_half(bottom_fragment)
        im1 = np.zeros([im1.shape[0], im1.shape[1]], np.uint8)
        im2 = np.copy(im1)
        copy_to(top_fragment_bottom, im1, 0, 0)
        copy_to(bottom_fragment_top, im2, 0, delta_for_max)

        return self.__stick_over(max_adjacency, im2, gap_lines[1], im1, gap_lines[0])

    def __get_bottom_half(self, img):
        return img[img.shape[0] / 2:, :]

    def __get_top_half(self, img):
        return img[:img.shape[0] / 2, :]

    def __stick_over(self, adjacency, fragment_above, edge_above, fragment_below, edge_below):
        above_area = self.edge_detector.getArea(fragment_above)
        above_area_pixels = np.nonzero(above_area)
        stuck_fragments = np.copy(fragment_below)
        stuck_fragments[above_area_pixels] = fragment_above[above_area_pixels]
        gap_line = cv2.bitwise_or(np.copy(edge_above), cv2.bitwise_and(edge_below, cv2.bitwise_not(above_area)))
        return FragmentsConnection(adjacency, stuck_fragments, gap_line)


class FragmentsContentMatcher(object):

    def matchFragmentsContent(self, stuck_fragments, gap_line):
        pass


class HarrisFragmentsContentMatcher(FragmentsContentMatcher):

    def __init__(self):
        self.write_result_to = None

    def matchFragmentsContent(self, stuck_fragments, gap_line):
        gap_line_pixels = np.nonzero(gap_line)
        fragments_harris = cv2.cornerHarris(stuck_fragments, blockSize=2, ksize=3, k=0.04)
        harris_values = fragments_harris[gap_line_pixels]
        if self.write_result_to is not None:
            img = np.zeros(fragments_harris.shape, fragments_harris.dtype)
            img[gap_line_pixels] = harris_values
            img = cv2.normalize(img, img, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            cv2.imwrite(self.write_result_to, img)
            cv2.imwrite('gap_' + self.write_result_to, gap_line)
        harris_sum = np.sum(np.abs(harris_values))
        return float(harris_sum) / float(harris_values.shape[0])
