from ctypes import *

import numpy as np

from random_walk_package import Matrix
from random_walk_package.wrapper import dll

dll.kernel_from_array.argtypes = [POINTER(c_double), c_ssize_t, c_ssize_t]
dll.kernel_from_array.restype = POINTER(Matrix)


def kernel_from_array(array: np.ndarray, width: int, height: int) -> Matrix:
    return dll.kernel_from_array(array.ctypes.data_as(POINTER(c_double)), width, height)
