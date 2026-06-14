from randomwalks.bindings.data_structures.Point2D import Point2DArrayHandle
from randomwalks.bindings.data_structures.Tensor import TensorHandle
from randomwalks.bindings.data_structures.types import *
from randomwalks.wrapper import dll


class BrownianWalkBinding:
    dll.brownian_init.argtypes = [
        MatrixPtr,
        c_ssize_t,
        c_ssize_t,
        c_ssize_t,
        c_ssize_t,
        c_ssize_t,
    ]
    dll.brownian_init.restype = TensorPtr
    init_ptr = dll.brownian_init

    dll.brownian_backtrace.argtypes = [
        TensorPtr,
        MatrixPtr,
        c_ssize_t,
        c_ssize_t,
    ]
    dll.brownian_backtrace.restype = Point2DArrayPtr
    backtrace_ptr = dll.brownian_backtrace

    @classmethod
    def generate(cls, kernel, width, height, time, start_x, start_y):
        ptr = cls.init_ptr(_matrix_ptr(kernel), width, height, time, start_x, start_y)
        return TensorHandle.from_ptr(ptr, owned=True)

    @classmethod
    def backtrace(cls, dp_matrix, kernel, end_x, end_y):
        walk = Point2DArrayHandle.from_ptr(
            cls.backtrace_ptr(_tensor_ptr(dp_matrix), _matrix_ptr(kernel), end_x, end_y),
            owned=True,
        )
        return walk.to_numpy()


def _matrix_ptr(matrix):
    return matrix.ptr if hasattr(matrix, "ptr") else matrix


def _tensor_ptr(tensor):
    return tensor.ptr if hasattr(tensor, "ptr") else tensor
