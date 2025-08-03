import ctypes

import numpy as np

from random_walk_package import Point2DArrayPtr, point2d_arr_free
from random_walk_package import get_walk_points
from random_walk_package.wrapper import dll

dll.gpu_brownian_walk.argtypes = [ctypes.POINTER(ctypes.c_float),  # kernel
                                  ctypes.c_uint32,  # S
                                  ctypes.c_uint32,  # T
                                  ctypes.c_uint32,  # W
                                  ctypes.c_uint32,  # H
                                  ctypes.c_uint32,  # start_x
                                  ctypes.c_uint32,  # start_y
                                  ctypes.c_uint32,  # end_x
                                  ctypes.c_uint32]  # end_y
dll.gpu_brownian_walk.restype = Point2DArrayPtr


def brownian_walk_gpu(kernel_np: np.ndarray, T, H, W, S, start_x, start_y, end_x, end_y):
    assert kernel_np.dtype == np.float32
    kernel_ptr = kernel_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    result = dll.gpu_brownian_walk(kernel_ptr,
                                   ctypes.c_uint32(S),
                                   ctypes.c_uint32(T),
                                   ctypes.c_uint32(W),
                                   ctypes.c_uint32(H),
                                   ctypes.c_uint32(start_x),
                                   ctypes.c_uint32(start_y),
                                   ctypes.c_uint32(end_x),
                                   ctypes.c_uint32(end_y))

    result_np = get_walk_points(result)
    point2d_arr_free(result)
    return result_np
