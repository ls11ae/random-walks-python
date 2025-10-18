import ctypes

from random_walk_package.bindings.data_structures.types import *
from random_walk_package.wrapper import dll

dll.tensor_set_new.argtypes = [ctypes.c_ssize_t, ]
dll.tensor_set_new.restype = TensorSetPtr

# Core Tensor operations
dll.tensor_new.restype = TensorPtr
dll.tensor_new.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]

dll.tensor_free.argtypes = [TensorPtr]
dll.tensor_free.restype = None

dll.tensor_fill.argtypes = [TensorPtr, ctypes.c_double]
dll.tensor_fill.restype = None

# Accessors and basic manipulation
dll.tensor_in_bounds.argtypes = [TensorPtr, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
dll.tensor_in_bounds.restype = c_int

dll.tensor_save.argtypes = [TensorPtr, ctypes.c_char_p]
dll.tensor_save.restype = ctypes.c_size_t

dll.tensor_load.argtypes = [ctypes.c_char_p]
dll.tensor_in_bounds.restype = TensorPtr

dll.tensor4D_free.argtypes = [ctypes.POINTER(TensorPtr), ctypes.c_ssize_t]
dll.tensor4D_free.restype = None


def tensor_new(width, height, depth):
    tensor = dll.tensor_new(width, height, depth)
    return tensor


def tensor_free(tensor_ptr: TensorPtr):
    dll.tensor_free(tensor_ptr)


def tensor4D_free(tensor_ptrs: POINTER(TensorPtr), length: int):
    dll.tensor4D_free(tensor_ptrs, length)
