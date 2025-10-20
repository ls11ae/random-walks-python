from random_walk_package import get_walk_points, create_point2d_array
from random_walk_package.bindings.data_structures.terrain import *
from random_walk_package.wrapper import dll

dll.correlated_init.argtypes = [
    c_ssize_t,  # W
    c_ssize_t,  # H
    POINTER(Tensor),  # kernel
    c_ssize_t,  # T
    c_ssize_t,  # start_x
    c_ssize_t,  # start_y
    c_bool,  # use_serialization
    c_char_p,  # output_folder
]
dll.correlated_init.restype = POINTER(POINTER(Tensor))

dll.correlated_backtrace.argtypes = [
    c_bool,  # use_serialization
    POINTER(POINTER(Tensor)),  # DP_Matrix
    c_char_p,  # dp_folder
    c_ssize_t,  # T
    POINTER(Tensor),  # kernel
    c_ssize_t,  # end_x
    c_ssize_t,  # end_y
    c_ssize_t,  # dir
]
dll.correlated_backtrace.restype = POINTER(Point2DArray)

dll.correlated_multi_step.argtypes = [
    c_ssize_t,  # W
    c_ssize_t,  # H
    c_char_p,  # dp_folder
    c_ssize_t,  # T
    POINTER(Tensor),  # kernel
    POINTER(Point2DArray),  # steps
    c_ssize_t,  # dir
    c_bool,  # use_serialization
]
dll.correlated_multi_step.restype = POINTER(Point2DArray)


def correlated_walk_init(kernel, width=100, height=100, time=50, start_x=None, start_y=None, use_serialization=False,
                         output_folder=None):
    if kernel is None:
        raise ValueError("Kernel is None.")
    return dll.correlated_init(width, height, kernel, time, start_x, start_y, use_serialization,
                               output_folder.encode('utf-8'))


def correlated_backtrace(dp_mat, T, kernels, end_x, end_y, direction=0, use_serialization=False, dp_folder=None):
    if dp_mat is None and use_serialization is False:
        raise ValueError("DP matrix is None.")
    if kernels is None:
        raise ValueError("Kernels are None.")
    if end_x is None or end_y is None:
        raise ValueError("End point is None.")
    if direction is None:
        direction = 0
    walk_ptr = dll.correlated_backtrace(use_serialization, dp_mat,
                                        dp_folder.encode('utf-8') if dp_folder is not None else None, T, kernels, end_x,
                                        end_y,
                                        direction)
    if walk_ptr is None:
        raise ValueError("Walk failed to backtrace. Maybe try again with higher T?")
    walk_np = get_walk_points(walk_ptr)
    dll.point2d_array_free(walk_ptr)
    return walk_np


def correlated_multi_step(W, H, T, kernels, steps, direction=0, use_serialization=False, dp_folder=None):
    if use_serialization and dp_folder is None:
        raise ValueError("DP folder is None.")
    if kernels is None:
        raise ValueError("Kernels are None.")
    if steps is None:
        raise ValueError("Steps are None.")
    if dp_folder is not None:
        dp_folder = dp_folder.encode('utf-8')

    steps_c = create_point2d_array(steps)
    walk_c = dll.correlated_multi_step(W, H, dp_folder, T, kernels, steps_c, direction, use_serialization)
    if walk_c is None:
        raise ValueError("Walk failed to backtrace. Maybe try again with higher T?")
    walk_np = get_walk_points(walk_c)
    print("free walk c")
    dll.point2d_array_free(walk_c)
    print("free steps c")
    dll.point2d_array_free(steps_c)
    print("free walk np")
    return walk_np
