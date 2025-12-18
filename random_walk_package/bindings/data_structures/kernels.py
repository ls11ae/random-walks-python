from ctypes import *

import numpy as np

from random_walk_package import Matrix, TensorPtr, matrix_free, MatrixPtr
from random_walk_package.wrapper import dll

dll.kernel_from_array.argtypes = [POINTER(c_double), c_ssize_t, c_ssize_t]
dll.kernel_from_array.restype = POINTER(Matrix)

dll.generate_kernels_from_matrix.argtypes = [POINTER(Matrix), c_ssize_t]
dll.generate_kernels_from_matrix.restype = TensorPtr

dll.generate_correlated_kernels.argtypes = [c_ssize_t, c_ssize_t]
dll.generate_correlated_kernels.restype = TensorPtr

dll.generate_chi_kernel.argtypes = [
    c_ssize_t,  # size
    c_ssize_t,  # subsample_size
    c_int,  # k
    c_int  # d
]
dll.generate_chi_kernel.restype = POINTER(Matrix)


def kernel_from_array(array: np.ndarray, width: int, height: int) -> MatrixPtr:
    return dll.kernel_from_array(array.ctypes.data_as(POINTER(c_double)), width, height)


def generate_correlated_kernels(width: int, dirs: int) -> TensorPtr:
    return dll.generate_correlated_kernels(dirs, width)


def clip_kernel(Z, radius):
    cx = Z.shape[0] // 2
    cy = Z.shape[1] // 2
    return Z[cx - radius:cx + radius + 1,
    cy - radius:cy + radius + 1]


def normalize_kernel(Z):
    s = Z.sum()
    if s > 0:
        return Z / s
    return Z


def correlated_kernels_from_matrix(array: np.ndarray, width: int, height: int, directions: int) -> TensorPtr:
    matrix_c = kernel_from_array(array, width, height)
    result = dll.generate_kernels_from_matrix(matrix_c, directions)
    matrix_free(matrix_c)
    return result
