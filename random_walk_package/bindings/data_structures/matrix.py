import matplotlib.pyplot as plt

from random_walk_package.bindings.data_structures.types import *
from random_walk_package.wrapper import dll

# Generate a new matrix
dll.matrix_new.restype = MatrixPtr
dll.matrix_new.argtypes = [c_ssize_t, c_ssize_t]

dll.matrix_free.argtypes = [MatrixPtr]
dll.matrix_free.restype = None

# I/O and serialization
dll.matrix_print.argtypes = [MatrixPtr]
dll.matrix_print.restype = None

dll.matrix_to_string.restype = c_char_p
dll.matrix_to_string.argtypes = [MatrixPtr]

dll.matrix_save.restype = c_ssize_t
dll.matrix_save.argtypes = [MatrixPtr, c_char_p]

dll.matrix_load.restype = MatrixPtr
dll.matrix_load.argtypes = [c_char_p]

# Normalization functions
dll.matrix_normalize.argtypes = [MatrixPtr, c_double]
dll.matrix_normalize.restype = None

dll.matrix_normalize_L1.argtypes = [MatrixPtr]
dll.matrix_normalize_L1.restype = None

# Additional operations from C code
dll.matrix_combind.restype = MatrixPtr
dll.matrix_combind.argtypes = [MatrixPtr, MatrixPtr]

dll.matrix_combind_inplace.restype = c_int
dll.matrix_combind_inplace.argtypes = [MatrixPtr, MatrixPtr]

dll.matrix_upsample_bilinear.restype = MatrixPtr
dll.matrix_upsample_bilinear.argtypes = [MatrixPtr, c_ssize_t, c_ssize_t]

dll.matrix_rotate.restype = MatrixPtr
dll.matrix_rotate.argtypes = [MatrixPtr, c_double]

dll.matrix_rotate_center.restype = MatrixPtr
dll.matrix_rotate_center.argtypes = [MatrixPtr, c_double]

dll.matrix_generator_gaussian_pdf.restype = MatrixPtr
dll.matrix_generator_gaussian_pdf.argtypes = [
    c_ssize_t,  # width
    c_ssize_t,  # height
    c_double,  # sigma
    c_double,  # scale
    c_ssize_t,  # x_offset
    c_ssize_t  # y_offset
]


def matrix_free(matrix_ptr):
    if matrix_ptr is not None:
        dll.matrix_free(matrix_ptr)


# Convert result to numpy array
def matrix_to_numpy(matrix_ptr):
    mat = matrix_ptr.contents
    arr = np.ctypeslib.as_array(mat.data, shape=(mat.height, mat.width))
    return arr.copy()


def matrix_new(width, height) -> Matrix:
    matrix = dll.matrix_new(width, height)
    return matrix


def create_gaussian_kernel(width, height, sigma, scale=1, x_offset=0, y_offset=0) -> MatrixPtr:
    kernel = dll.matrix_generator_gaussian_pdf(width, height, sigma, scale, x_offset, y_offset)
    return kernel


def create_correlated_chi_kernels(directions, step_size):
    kernel = dll.generate_kernels(directions, step_size * 2 + 1)
    return kernel


def plot_kernel(matrix):
    plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.show()


import numpy as np


def matrix_generator_gaussian_pdf(width, height, sigma, x_offset=0, y_offset=0):
    assert sigma > 0, "Sigma must be positive"
    sigma = max(sigma, 2.0)

    width_half = width // 2
    height_half = height // 2

    x_offset += width_half
    y_offset += height_half

    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)

    distance_squared = (xx - x_offset) ** 2 + (yy - y_offset) ** 2
    gaussian = np.exp(-distance_squared / (2.0 * sigma ** 2))
    gaussian /= gaussian.sum()

    return gaussian.astype(np.float32)
