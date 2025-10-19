import os
import sys
from ctypes import *

from random_walk_package.bindings.data_structures.point2D import create_point2d_array, get_walk_points
from random_walk_package.bindings.data_structures.types import Tensor, Matrix, Point2DArray
from random_walk_package.wrapper import dll

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# New functions from header
dll.brownian_init.argtypes = [
    POINTER(Matrix),  # kernel
    c_ssize_t,  # W
    c_ssize_t,  # H
    c_ssize_t,  # T
    c_ssize_t,  # start_x
    c_ssize_t,  # start_y
]
dll.brownian_init.restype = POINTER(Tensor)

dll.brownian_backtrace.argtypes = [
    POINTER(Tensor),  # DP
    POINTER(Matrix),  # Kernel
    c_ssize_t, c_ssize_t  # end coordinates
]
dll.brownian_backtrace.restype = POINTER(Point2DArray)

dll.brownian_multi_step.argtypes = [
    c_ssize_t,  # W
    c_ssize_t,  # H
    c_ssize_t,  # T
    POINTER(Matrix),  # kernel
    POINTER(Point2DArray),  # steps
]
dll.brownian_multi_step.restype = POINTER(Point2DArray)


# Wrapper
def brownian_walk_init(kernel, width, height, time, start_x=None, start_y=None):
    return dll.brownian_init(kernel, width, height, time, start_x, start_y)


def brownian_backtrace(dp_matrix, kernel, end_x, end_y):
    walk_c = dll.brownian_backtrace(dp_matrix, kernel, end_x, end_y)
    walk_np = get_walk_points(walk_c)
    dll.point2d_array_free(walk_c)
    return walk_np


def brownian_backtrace_multiple(kernel, points, time, width, height):
    array_ptr = create_point2d_array(points)
    multistep_walk = dll.brownian_multi_step(width, height, time, kernel, array_ptr)
    walk_np = get_walk_points(multistep_walk)
    dll.point2d_array_free(multistep_walk)
    dll.point2d_array_free(array_ptr)
    return walk_np
