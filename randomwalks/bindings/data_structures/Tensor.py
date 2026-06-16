import ctypes

import numpy as np

from randomwalks.bindings.data_structures.types import *
from randomwalks.wrapper import dll


class TensorHandle:
    dll.tensor_new.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
    dll.tensor_new.restype = TensorPtr
    new = dll.tensor_new

    dll.tensor_free.argtypes = [TensorPtr]
    dll.tensor_free.restype = None
    free_ptr = dll.tensor_free

    dll.tensor_copy.argtypes = [TensorPtr]
    dll.tensor_copy.restype = TensorPtr
    copy_ptr = dll.tensor_copy

    dll.tensor_clone.argtypes = [TensorPtr]
    dll.tensor_clone.restype = TensorPtr
    clone_ptr = dll.tensor_clone

    dll.tensor_fill.argtypes = [TensorPtr, ctypes.c_float]
    dll.tensor_fill.restype = None
    fill_ptr = dll.tensor_fill

    dll.tensor_normalize.argtypes = [TensorPtr]
    dll.tensor_normalize.restype = None
    normalize_ptr = dll.tensor_normalize

    dll.tensor_sum.argtypes = [TensorPtr]
    dll.tensor_sum.restype = ctypes.c_double
    sum_ptr = dll.tensor_sum

    dll.tensor_in_bounds.argtypes = [TensorPtr, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
    dll.tensor_in_bounds.restype = c_int
    in_bounds_ptr = dll.tensor_in_bounds

    dll.tensor_save.argtypes = [TensorPtr, ctypes.c_char_p]
    dll.tensor_save.restype = ctypes.c_size_t
    save_ptr = dll.tensor_save

    dll.tensor_load.argtypes = [ctypes.c_char_p]
    dll.tensor_load.restype = TensorPtr
    load_ptr = dll.tensor_load

    def __init__(self, width=None, height=None, depth=None, *, ptr=None, owned=True):
        if ptr is None:
            if width is None or height is None or depth is None:
                raise ValueError("width, height, and depth are required when ptr is not provided")
            ptr = self.new(width, height, depth)
        if not ptr:
            raise RuntimeError("Failed to allocate Tensor")
        self._ptr = ptr
        self._owned = owned

    @classmethod
    def from_ptr(cls, ptr, *, owned=False):
        return cls(ptr=ptr, owned=owned)

    @classmethod
    def load(cls, foldername):
        ptr = cls.load_ptr(foldername.encode("utf-8"))
        return cls.from_ptr(ptr, owned=True)

    @property
    def ptr(self):
        return self._ptr

    @property
    def contents(self):
        return self._ptr.contents

    def fill(self, value):
        self.fill_ptr(self._ptr, value)

    def normalize(self):
        self.normalize_ptr(self._ptr)

    def sum(self):
        return self.sum_ptr(self._ptr)

    def save(self, foldername):
        return self.save_ptr(self._ptr, foldername.encode("utf-8"))

    def free(self):
        if self._owned and self._ptr:
            self.free_ptr(self._ptr)
        self._ptr = None

    def __bool__(self):
        return bool(self._ptr)

    def __del__(self):
        self.free()


class Tensor4DHandle:
    dll.tensor4D_free.argtypes = [ctypes.POINTER(TensorPtr), ctypes.c_ssize_t]
    dll.tensor4D_free.restype = None
    free_4d = dll.tensor4D_free

    def __init__(self, T=None, ptr=None, owned=True):
        self._ptr = ptr
        self._owned = owned
        self._T = T

    @property
    def ptr(self):
        return self._ptr

    @property
    def T(self):
        return self._T

    @classmethod
    def from_ptr(cls, ptr, T):
        return cls(T=T, ptr=ptr)

    def to_numpy_sum(self, width, height, *, average=True):
        if not self._ptr:
            raise ValueError("NULL Tensor4D pointer")

        acc = np.zeros((height, width), dtype=np.float64)
        for t in range(self._T):
            tensor_ptr = self._ptr[t]
            if not tensor_ptr:
                continue

            tensor = tensor_ptr.contents
            for direction in range(tensor.len):
                matrix_ptr = tensor.data[direction]
                if not matrix_ptr:
                    continue
                matrix = matrix_ptr.contents
                values = np.ctypeslib.as_array(matrix.points, shape=(height, width))
                acc += values

        if average and self._T:
            acc /= self._T
        return acc

    def free(self):
        print("freeing 4D tensor")
        if self._owned and self._ptr:
            self.free_4d(self._ptr, self._T)
        self._ptr = None

    def __bool__(self):
        return bool(self._ptr)

    def __del__(self):
        self.free()
