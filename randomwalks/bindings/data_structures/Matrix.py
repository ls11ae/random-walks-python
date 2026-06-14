import numpy as np

from randomwalks.bindings.data_structures.types import *
from randomwalks.wrapper import dll


class MatrixHandle:
    dll.matrix_new.argtypes = [c_ssize_t, c_ssize_t]
    dll.matrix_new.restype = MatrixPtr
    new = dll.matrix_new

    dll.matrix_free.argtypes = [MatrixPtr]
    dll.matrix_free.restype = None
    free_ptr = dll.matrix_free

    dll.matrix_copy.argtypes = [MatrixPtr]
    dll.matrix_copy.restype = MatrixPtr
    copy_ptr = dll.matrix_copy

    dll.matrix_clone.argtypes = [MatrixPtr]
    dll.matrix_clone.restype = MatrixPtr
    clone_ptr = dll.matrix_clone

    dll.matrix_fill.argtypes = [MatrixPtr, c_double]
    dll.matrix_fill.restype = None
    fill_ptr = dll.matrix_fill

    dll.matrix_factor_inplace.argtypes = [MatrixPtr, c_double]
    dll.matrix_factor_inplace.restype = None
    factor_inplace = dll.matrix_factor_inplace

    dll.matrix_print.argtypes = [MatrixPtr]
    dll.matrix_print.restype = None
    print_ptr = dll.matrix_print

    dll.matrix_print_to_file.argtypes = [MatrixPtr, c_char_p]
    dll.matrix_print_to_file.restype = None
    print_to_file = dll.matrix_print_to_file

    dll.matrix_to_string.argtypes = [MatrixPtr]
    dll.matrix_to_string.restype = c_char_p
    to_string_ptr = dll.matrix_to_string

    dll.matrix_save.argtypes = [MatrixPtr, c_char_p]
    dll.matrix_save.restype = c_size_t
    save_ptr = dll.matrix_save

    dll.matrix_load.argtypes = [c_char_p]
    dll.matrix_load.restype = MatrixPtr
    load_ptr = dll.matrix_load

    dll.matrix_normalize_L1.argtypes = [MatrixPtr]
    dll.matrix_normalize_L1.restype = None
    normalize_l1 = dll.matrix_normalize_L1

    dll.matrix_in_bounds.argtypes = [MatrixPtr, c_size_t, c_size_t]
    dll.matrix_in_bounds.restype = c_int
    in_bounds_ptr = dll.matrix_in_bounds

    dll.matrix_generator_gaussian_pdf.argtypes = [
        c_ssize_t,
        c_ssize_t,
        c_double,
        c_ssize_t,
        c_ssize_t,
    ]
    dll.matrix_generator_gaussian_pdf.restype = MatrixPtr
    gaussian_pdf = dll.matrix_generator_gaussian_pdf

    def __init__(self, width=None, height=None, *, ptr=None, owned=True):
        if ptr is None:
            if width is None or height is None:
                raise ValueError("width and height are required when ptr is not provided")
            ptr = self.new(width, height)
        if not ptr:
            raise RuntimeError("Failed to allocate Matrix")
        self._ptr = ptr
        self._owned = owned

    @classmethod
    def from_ptr(cls, ptr, *, owned=False):
        return cls(ptr=ptr, owned=owned)

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "MatrixHandle":
        assert arr is not None
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("Matrix must be a 2D numpy array")
        # NumPy shape: rows, cols
        height, width = arr.shape
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        handle = cls(width=width, height=height)
        matrix = handle._ptr.contents
        expected_len = width * height
        memmove(
            matrix.points,
            arr.ctypes.data,
            expected_len * sizeof(c_double)
        )
        return handle

    @classmethod
    def load(cls, filename):
        ptr = cls.load_ptr(filename.encode("utf-8"))
        return cls.from_ptr(ptr, owned=True)

    @classmethod
    def gaussian(cls, width, height, sigma, *, x_offset=0, y_offset=0):
        ptr = cls.gaussian_pdf(width, height, sigma, x_offset, y_offset)
        if not ptr:
            raise RuntimeError("Failed to allocate Gaussian Matrix")
        return cls.from_ptr(ptr, owned=True)

    @property
    def ptr(self):
        return self._ptr

    @property
    def contents(self):
        return self._ptr.contents

    @staticmethod
    def __matrix_to_numpy(matrix_ptr):
        matrix_ptr = _matrix_ptr(matrix_ptr)
        mat = matrix_ptr.contents
        arr = np.ctypeslib.as_array(mat.points, shape=(mat.height, mat.width))
        return arr.copy()

    def to_numpy(self):
        return MatrixHandle.__matrix_to_numpy(self._ptr)

    def save(self, filename):
        return self.save_ptr(self._ptr, filename.encode("utf-8"))

    def fill(self, value):
        self.fill_ptr(self._ptr, value)

    def free(self):
        if self._owned and self._ptr:
            self.free_ptr(self._ptr)
        self._ptr = None

    def __bool__(self):
        return bool(self._ptr)

    def __del__(self):
        self.free()


def _matrix_ptr(matrix):
    return matrix.ptr if hasattr(matrix, "ptr") else matrix
