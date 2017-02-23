import numpy as np


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

    result = np.copy(dst)
    result[dst_y[0]:dst_y[1], dst_x[0]:dst_x[1]] = src[src_y[0]:src_y[1], src_x[0]:src_x[1]]
    return result


def count_nonzero(arr):
    return np.nonzero(arr)[0].shape[0]
