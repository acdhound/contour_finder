import numpy as np
import cv2
from ..operations.util import count_nonzero, copy_to


class EdgeMatcher(object):
    """
        Matches edges of two images representing pieces of a paper with text broken into horizontal strips.
    Edges of pieces are shifted relative to each other vertically to find position where they most look like
    two adjacent pieces of rotten paper put together.
        The result is a tuple of three elements. First is the value of numeric coefficient representing edges'
    adjacency, second is the image representing two pieces stuck together and third is corresponding gap line
    between pieces.
    """

    def matchEdges(self, top_piece, bottom_piece):
        pass


def get_bottom_half(img):
    return img[img.shape[0] / 2:, :]


def get_top_half(img):
    return img[:img.shape[0] / 2, :]


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

        top_edge_bottom = get_bottom_half(top_edge)
        bottom_edge_top = get_top_half(bottom_edge)
        im1 = np.zeros([top_edge_bottom.shape[0] + bottom_edge_top.shape[0], top_edge.shape[1]], np.uint8)
        copy_to(top_edge_bottom, im1, 0, 0)

        max_nonzero, delta_for_max, im_product_for_max = 0, 0, None

        for delta in range(-bottom_edge_top.shape[0] + 1, top_edge_bottom.shape[0]):
            im2 = np.zeros([im1.shape[0], im1.shape[1]], np.uint8)
            copy_to(bottom_edge_top, im2, 0, delta)

            im_product = cv2.bitwise_and(im1, im2)
            nonzero = count_nonzero(im_product)
            if nonzero > max_nonzero:
                max_nonzero = nonzero
                delta_for_max = delta
                im_product_for_max = im_product
            elif im_product_for_max is None:
                im_product_for_max = im_product

        top_piece_bottom = get_bottom_half(top_piece)
        bottom_piece_top = get_top_half(bottom_piece)
        im1 = np.zeros([im1.shape[0], im1.shape[1]], np.uint8)
        im2 = np.copy(im1)
        copy_to(top_piece_bottom, im1, 0, 0)
        copy_to(bottom_piece_top, im2, 0, delta_for_max)
        stuck_pieces = cv2.max(im1, im2)

        return max_nonzero, stuck_pieces, im_product_for_max


class ContentMatcher(object):

    def matchContent(self, stuck_pieces, gap_line):
        pass
