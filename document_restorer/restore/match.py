import numpy as np
import cv2


class EdgeMatcher(object):
    """
        Matches edges of two images representing pieces of a paper with text broken into horizontal strips.
    Edges of pieces are shifted relative to each other vertically to find position where they most look like
    two adjacent pieces of rotten paper put together.
        The result is a tuple of three elements. First is the number of pixels from the bottom of first image to the top
    of second image for which edges are most adjacent. Second is the value of numeric coefficient representing edges'
    adjacency when shifted by this number. Third is the image representing supposed gap line between two pieces.
    """

    def matchEdges(self, top_piece, bottom_piece):
        pass


class BinaryEdgeMatcher(EdgeMatcher):
    """
        Extracts binary edges from pieces and calculates adjacency coefficient by applying logical AND and counting
    number of non-zero pixels.
    """

    def __init__(self, edge_detector):
        self.edge_detector = edge_detector

    def matchEdges(self, top_piece, bottom_piece):
        if top_piece.shape[1] != bottom_piece.shape[1]:
            raise Exception("Pieces of the same width expected")

        top_edge = self.edge_detector.getEdges(top_piece)
        bottom_edge = self.edge_detector.getEdges(bottom_piece)
        width = top_piece.shape[1]

        half_top = top_edge.shape[0] / 2
        half_bottom = bottom_edge.shape[0] / 2
        im1 = np.zeros([half_top + half_bottom, width], np.uint8)
        if half_top == top_edge.shape[0] - half_top:
            im1[0:half_top, 0:width] = top_edge[half_top:, 0:width]
        else:
            im1[0:half_top, 0:width] = top_edge[half_top + 1:, 0:width]

        max_nonzero, delta_for_max, im_product_for_max = 0, 0, None

        for delta in range(-half_bottom + 1, half_top + 1):
            im2 = np.zeros([half_top + half_bottom, width], np.uint8)
            if delta >= 0:
                im2[delta:delta + half_bottom, 0:width] = bottom_edge[0:half_bottom, 0:width]
            else:
                im2[0:half_bottom + delta, 0:width] = bottom_edge[-delta:half_bottom, 0:width]

            im_product = cv2.bitwise_and(im1, im2)
            nonzero = np.nonzero(im_product)[0].shape[0]
            if nonzero > max_nonzero:
                max_nonzero = nonzero
                delta_for_max = delta
                im_product_for_max = im_product
            elif im_product_for_max is None:
                im_product_for_max = im_product

        return delta_for_max, max_nonzero, im_product_for_max


class ContentMatcher(object):

    def matchContent(self, top_piece, bottom_piece, gap_line):
        pass