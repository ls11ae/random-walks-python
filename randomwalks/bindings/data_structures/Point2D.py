from typing import Tuple, List

import numpy as np

from randomwalks.bindings.data_structures.types import *
from randomwalks.wrapper import dll


class Point2DHandle:
    dll.point_2d_new.argtypes = [c_ssize_t, c_ssize_t]
    dll.point_2d_new.restype = Point2DPtr
    new = dll.point_2d_new

    dll.point_2d_free.argtypes = [Point2DPtr]
    dll.point_2d_free.restype = None
    free_ptr = dll.point_2d_free

    def __init__(self, x=None, y=None, *, ptr=None, owned=True):
        if ptr is None:
            if x is None or y is None:
                raise ValueError("x and y are required when ptr is not provided")
            ptr = self.new(x, y)
        if not ptr:
            raise RuntimeError("Failed to allocate Point2D")
        self._ptr = ptr
        self._owned = owned

    @classmethod
    def from_ptr(cls, ptr, *, owned=False):
        return cls(ptr=ptr, owned=owned)

    @classmethod
    def from_points(cls, points):
        return cls(points=list(points), owned=True)

    @property
    def ptr(self):
        return self._ptr

    @property
    def contents(self):
        return self._ptr.contents

    def free(self):
        if self._owned and self._ptr:
            self.free_ptr(self._ptr)
        self._ptr = None

    def __bool__(self):
        return bool(self._ptr)

    def __del__(self):
        self.free()


class Point2DArrayHandle:
    dll.point_2d_array_new.argtypes = [Point2DPtr, c_size_t]
    dll.point_2d_array_new.restype = Point2DArrayPtr
    new = dll.point_2d_array_new

    dll.point_2d_array_new_empty.argtypes = [c_size_t]
    dll.point_2d_array_new_empty.restype = Point2DArrayPtr
    new_empty = dll.point_2d_array_new_empty

    dll.point2d_array_print.argtypes = [Point2DArrayPtr]
    dll.point2d_array_print.restype = None
    print_ptr = dll.point2d_array_print

    dll.point2d_array_free.argtypes = [Point2DArrayPtr]
    dll.point2d_array_free.restype = None
    free_ptr = dll.point2d_array_free

    def __init__(self, points: List[Tuple[int, int]] = None, *, length=None, ptr=None, owned=True):
        if ptr is None:
            if points is None:
                if length is None:
                    raise ValueError("points or length is required when ptr is not provided")
                ptr = self.new_empty(length)
            else:
                point_array = _point_array_buffer(points)
                ptr = self.new(point_array, len(points))
        if not ptr:
            raise RuntimeError("Failed to allocate Point2DArray")
        self._ptr = ptr
        self._owned = owned

    @staticmethod
    def __get_walk_points(walk) -> np.ndarray:
        walk = _point_array_ptr(walk)
        if not walk:
            raise ValueError("NULL walk pointer")

        walk_data = walk.contents
        if not walk_data.points:
            raise ValueError("NULL points array in walk")

        points = []
        for i in range(walk_data.length):
            point = walk_data.points[i]
            points.append((point.x, point.y))

        return np.array(points, dtype=np.int64)

    @classmethod
    def from_ptr(cls, ptr, *, owned=False):
        return cls(ptr=ptr, owned=owned)

    @property
    def ptr(self):
        return self._ptr

    @property
    def contents(self):
        return self._ptr.contents

    def to_numpy(self):
        return Point2DArrayHandle.__get_walk_points(self._ptr)

    def free(self):
        if self._owned and self._ptr:
            self.free_ptr(self._ptr)
        self._ptr = None

    def __bool__(self):
        return bool(self._ptr)

    def __del__(self):
        self.free()


def _point_array_buffer(steps):
    point_array = (Point2D * len(steps))()
    for i, (x, y) in enumerate(steps):
        point_array[i].x = int(x)
        point_array[i].y = int(y)
    return point_array


def _point_array_ptr(array_ptr):
    return array_ptr.ptr if hasattr(array_ptr, "ptr") else array_ptr


def _point2d_array_new(length):
    return Point2DArrayHandle.new_empty(length)


def _point2d_arr_free(array_ptr):
    if hasattr(array_ptr, "free"):
        array_ptr.free()
        return
    array_ptr = _point_array_ptr(array_ptr)
    if array_ptr is not None:
        Point2DArrayHandle.free_ptr(array_ptr)


def _point2d_array_grid_new(height, width, times):
    grid = Point2DArrayGrid()
    grid.height = height
    grid.width = width
    grid.times = times

    data = (POINTER(POINTER(Point2DArray)) * height)()
    row_buffers = []
    for y in range(height):
        row = (POINTER(Point2DArray) * width)()
        row_buffers.append(row)
        data[y] = row
        for x in range(width):
            data[y][x] = _point2d_array_new(times)

    grid.data = data
    grid._buffers = (data, row_buffers)
    return grid
