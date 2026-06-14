from ctypes import POINTER

import numpy as np

from randomwalks.bindings.data_structures.Matrix import MatrixHandle
from randomwalks.bindings.data_structures.Tensor import TensorHandle
from randomwalks.bindings.data_structures.types import *
from randomwalks.wrapper import dll


class KernelFactory:
    dll.kernel_from_array.argtypes = [POINTER(c_double), c_ssize_t, c_ssize_t]
    dll.kernel_from_array.restype = MatrixPtr
    from_array_ptr = dll.kernel_from_array

    dll.generate_kernels_from_matrix.argtypes = [MatrixPtr, c_ssize_t]
    dll.generate_kernels_from_matrix.restype = TensorPtr
    from_matrix_ptr = dll.generate_kernels_from_matrix

    dll.generate_correlated_kernels.argtypes = [c_ssize_t, c_ssize_t, c_double, c_double]
    dll.generate_correlated_kernels.restype = TensorPtr
    correlated_ptr = dll.generate_correlated_kernels

    dll.generate_chi_kernel.argtypes = [c_ssize_t, c_ssize_t, c_int, c_int]
    dll.generate_chi_kernel.restype = MatrixPtr
    chi_ptr = dll.generate_chi_kernel

    dll.generate_directed_matrix.argtypes = [c_ssize_t, c_float, c_ssize_t, c_ssize_t]
    dll.generate_directed_matrix.restype = MatrixPtr
    directed_ptr = dll.generate_directed_matrix

    dll.generate_kernel.argtypes = [KernelParametersPtr]
    dll.generate_kernel.restype = TensorPtr
    kernel_ptr = dll.generate_kernel

    dll.generate_kernel_from_set.argtypes = [KernelParametersPtr, c_int, TensorSetPtr, c_bool]
    dll.generate_kernel_from_set.restype = TensorPtr
    kernel_from_set_ptr = dll.generate_kernel_from_set

    @classmethod
    def matrix_from_array(cls, array: np.ndarray):
        array = np.ascontiguousarray(array, dtype=np.float64)
        ptr = cls.from_array_ptr(array.ctypes.data_as(POINTER(c_double)), array.shape[1], array.shape[0])
        return MatrixHandle.from_ptr(ptr, owned=True)

    @classmethod
    def correlated(cls, width: int, directions: int, *,
                   angle_diffusivity: float = 0.3,
                   length_diffusivity: float = 1.0):
        ptr = cls.correlated_ptr(directions, width, angle_diffusivity, length_diffusivity)
        return TensorHandle.from_ptr(ptr, owned=True)

    @classmethod
    def correlated_from_matrix(cls, array: np.ndarray, directions: int):
        matrix = cls.matrix_from_array(array)
        try:
            ptr = cls.from_matrix_ptr(matrix.ptr, directions)
            return TensorHandle.from_ptr(ptr, owned=True)
        finally:
            matrix.free()

    @classmethod
    def chi(cls, size: int, subsample_size: int, k: int, d: int):
        ptr = cls.chi_ptr(size, subsample_size, k, d)
        return MatrixHandle.from_ptr(ptr, owned=True)

    @classmethod
    def directed(cls, step_size: int, angle_diffusivity: float, bias_x: int, bias_y: int):
        ptr = cls.directed_ptr(step_size, angle_diffusivity, bias_x, bias_y)
        return MatrixHandle.from_ptr(ptr, owned=True)

    @classmethod
    def from_parameters(cls, params):
        if isinstance(params, KernelParameters):
            params = byref(params)
        ptr = cls.kernel_ptr(params)
        return TensorHandle.from_ptr(ptr, owned=True)
