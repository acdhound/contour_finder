import numpy as np
import cv2
from core.operations.util import copy_to, to_norm_cv2_8bit_gray, calculate_avg_in_rows


class FragmentsConnector(object):
    """
        Tries to connect two fragments of the paper text document assuming the document was broken
    into horizontal strips. Analyses fragments' edges and calculates numerical coefficient representing
    fragments' adjacency.
    """

    def connectFragments(self, top_fragment, bottom_fragment):
        pass


class FragmentsConnection(object):

    def __init__(self, adjacency, stuck_fragments, gap_line, offset):
        self.adjacency = adjacency
        self.stuck_fragments = stuck_fragments
        self.gap_line = gap_line
        self.offset = offset


class VerticalShiftFragmentsConnector(FragmentsConnector):

    def __init__(self, x_range=None):
        if x_range is None:
            self.x_range = [0]
        else:
            self.x_range = x_range

    def connectFragments(self, top_fragment, bottom_fragment):
        top_edge = top_fragment.drawContour()
        bottom_edge = bottom_fragment.drawContour()

        top_edge_bottom = self.__get_bottom_half(top_edge)
        bottom_edge_top = self.__get_top_half(bottom_edge)
        im1 = np.zeros([top_edge_bottom.shape[0] + bottom_edge_top.shape[0],
                        max(top_edge.shape[1], bottom_edge.shape[1])], np.uint8)
        copy_to(top_edge_bottom, im1, 0, 0)

        max_adjacency, delta_for_max, gap_edges = 0, [], None

        for delta_x in self.x_range:
            for delta_y in range(-bottom_edge_top.shape[0] + 1, top_edge_bottom.shape[0]):
                im2 = np.zeros([im1.shape[0], im1.shape[1]], np.uint8)
                copy_to(bottom_edge_top, im2, delta_x, delta_y)
                common_edge = cv2.bitwise_and(im1, im2)
                adjacency = float(np.count_nonzero(common_edge)) / float(common_edge.shape[1])
                if adjacency > max_adjacency:
                    max_adjacency = adjacency
                    delta_for_max = [delta_x, delta_y]
                    gap_edges = [im1, im2]
                elif gap_edges is None:
                    gap_edges = [im1, im2]

        return self.__stick_halves(max_adjacency, top_fragment, bottom_fragment, delta_for_max, gap_edges)

    def __get_bottom_half(self, img):
        return img[img.shape[0] / 2:, :]

    def __get_top_half(self, img):
        return img[:img.shape[0] / 2, :]

    def __stick_halves(self, adjacency, top_f, bot_f, offset, gap_edges):
        top_half = self.__get_bottom_half(top_f.img)
        bot_half = self.__get_top_half(bot_f.img)

        result = np.zeros([top_half.shape[0] + bot_half.shape[0], max(top_half.shape[1], bot_half.shape[1])],
                          top_half.dtype)
        copy_to(bot_half, result, offset[0], offset[1])
        top_half_area = self.__get_bottom_half(top_f.area)
        top_half_area_pixels = np.nonzero(top_half_area)
        result[top_half_area_pixels] = top_half[top_half_area_pixels]

        out_of_top = np.zeros(gap_edges[0].shape, gap_edges[0].dtype)
        copy_to(top_half_area, out_of_top, 0, 0)
        out_of_top = cv2.bitwise_not(out_of_top)
        gap_line = cv2.bitwise_or(np.copy(gap_edges[0]), cv2.bitwise_and(gap_edges[1], out_of_top))

        return FragmentsConnection(adjacency,
                                   result[0:offset[1] + bot_half.shape[0], :],
                                   gap_line[0:offset[1] + bot_half.shape[0], :],
                                   offset)


class FragmentsContentMatcher(object):

    def matchFragmentsContent(self, stuck_fragments, gap_line):
        pass


class HarrisFragmentsContentMatcher(FragmentsContentMatcher):

    def __init__(self):
        self.write_result_to = None

    def matchFragmentsContent(self, stuck_fragments, gap_line):
        gap_line_pixels = np.nonzero(gap_line)
        harris_stuck = cv2.cornerHarris(stuck_fragments, blockSize=2, ksize=3, k=0.04)
        harris_values = harris_stuck[gap_line_pixels]
        if self.write_result_to is not None:
            harris_gap = np.zeros(harris_stuck.shape, harris_stuck.dtype)
            harris_gap[gap_line_pixels] = harris_values
            cv2.imwrite('harris_stuck_' + self.write_result_to, to_norm_cv2_8bit_gray(harris_stuck))
            cv2.imwrite('harris_gap_' + self.write_result_to, to_norm_cv2_8bit_gray(harris_gap))
            cv2.imwrite('gap_' + self.write_result_to, gap_line)
            cv2.imwrite('stuck_' + self.write_result_to, stuck_fragments)
            cv2.imwrite('avg_stuck_' + self.write_result_to,
                        to_norm_cv2_8bit_gray(calculate_avg_in_rows(stuck_fragments)))
            cv2.imwrite('avg_harris_stuck_' + self.write_result_to,
                        to_norm_cv2_8bit_gray(calculate_avg_in_rows(harris_stuck)))
        harris_sum = np.sum(np.abs(harris_values))
        return float(harris_sum) / float(harris_values.shape[0])


class SobelFragmentsContentMatcher(FragmentsContentMatcher):

    def __init__(self):
        self.write_result_to = None

    def matchFragmentsContent(self, stuck_fragments, gap_line):
        sobel_x_stuck = cv2.Sobel(stuck_fragments, cv2.CV_8U, 0, 1, ksize=1)
        gap_line_pixels = np.nonzero(gap_line)
        sobel_x_values = sobel_x_stuck[gap_line_pixels]
        if self.write_result_to is not None:
            sobel_x_gap = np.zeros(sobel_x_stuck.shape, sobel_x_stuck.dtype)
            sobel_x_gap[gap_line_pixels] = sobel_x_values
            cv2.imwrite('sobel_gap_' + self.write_result_to, sobel_x_gap)
            cv2.imwrite('sobel_stuck_' + self.write_result_to, sobel_x_stuck)
        return float(np.sum(sobel_x_values)) / float(sobel_x_values.shape[0])
