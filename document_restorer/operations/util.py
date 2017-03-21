import numpy as np
import cv2


def copy_to(src, dst, x0, y0):
    def src_axis_range(a0, src_size, dst_size):
        if (a0 >= dst_size) or (a0 < 0 and -a0 >= src_size):
            raise Exception("can't copy to area out of destination image")
        if a0 >= 0:
            return 0, min(src_size, dst_size - a0)
        else:
            return -a0, min(src_size, -a0 + dst_size)

    def dst_axis_range(a0, dst_size, size_to_copy):
        rng = (a0, dst_size) if a0 > 0 else (0, dst_size)
        diff = size_to_copy - (rng[1] - rng[0])
        if diff < 0:
            return rng[0], rng[1] + diff
        return rng

    src_x = src_axis_range(x0, src.shape[1], dst.shape[1])
    src_y = src_axis_range(y0, src.shape[0], dst.shape[0])
    dst_x = dst_axis_range(x0, dst.shape[1], src_x[1] - src_x[0])
    dst_y = dst_axis_range(y0, dst.shape[0], src_y[1] - src_y[0])

    dst[dst_y[0]:dst_y[1], dst_x[0]:dst_x[1]] = src[src_y[0]:src_y[1], src_x[0]:src_x[1]]
    return dst


def find_max(arr):
    return np.unravel_index(np.nanargmax(arr), arr.shape)


def find_min(arr):
    return np.unravel_index(np.nanargmin(arr), arr.shape)


def calculate_avg_in_rows(img):
    avg = np.zeros([img.shape[0], 50], img.dtype)
    for i in range(0, img.shape[0]):
        value = np.sum(img[i, :]) / img.shape[1]
        avg[i, :] = np.full([1, avg.shape[1]], value, dtype=avg.dtype)
    return avg


def to_norm_cv2_8bit_gray(img):
    return cv2.normalize(img, img, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

