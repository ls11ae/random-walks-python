import numpy as np

from random_walk_package.bindings.data_structures.types import *
from random_walk_package.wrapper import dll

# Load DLL functions
dll.point_2d_new.argtypes = [c_size_t, c_size_t]
dll.point_2d_new.restype = POINTER(Point2D)

dll.point_2d_free.argtypes = [POINTER(Point2D)]
dll.point_2d_free.restype = None

dll.point_2d_array_new.argtypes = [POINTER(Point2D), c_size_t]
dll.point_2d_array_new.restype = Point2DArrayPtr

dll.point2d_array_print.argtypes = [Point2DArrayPtr]
dll.point2d_array_print.restype = None

dll.point2d_array_free.argtypes = [Point2DArrayPtr]
dll.point2d_array_free.restype = None

dll.point_2d_array_grid_new.argtypes = [c_size_t, c_size_t, c_size_t]
dll.point_2d_array_grid_new.restype = Point2DArrayGridPtr

dll.point_2d_array_grid_free.argtypes = [Point2DArrayGridPtr]
dll.point_2d_array_grid_free.restype = None

dll.point_2d_array_new_empty.argtypes = [c_size_t]
dll.point_2d_array_new_empty.restype = Point2DArrayPtr

# Wrap load_weather_grid
dll.load_weather_grid.argtypes = [c_char_p, c_int, c_int, c_int]
dll.load_weather_grid.restype = Point2DArrayGridPtr


def point2d_array_new(length):
    return dll.point_2d_array_new_empty(length)


def point2d_arr_free(array_ptr):
    dll.point2d_array_free(array_ptr)


def point2d_array_grid(width, height, times):
    return dll.point_2d_array_grid_new(width, height, times)


def load_weather_grid(filename_base: str, grid_y: int, grid_x: int, times: int):
    """
    Loads a weather grid from files with the given base filename.
    Args:
        filename_base (str): Base filename (without extension)
        grid_y (int): Number of grid rows
        grid_x (int): Number of grid columns
        times (int): Number of time steps
    Returns:
        Point2DArrayGridPtr: Pointer to the loaded grid
    """
    return dll.load_weather_grid(filename_base.encode('utf-8'), grid_y, grid_x, times)


def point2d_array_grid_new(height, width, times):
    # Allocate grid structure
    grid = Point2DArrayGrid()
    grid.height = height
    grid.width = width
    grid.times = times

    # Allocate 2D array of Point2DArray pointers
    data = (POINTER(POINTER(Point2DArray)) * height)()
    for y in range(height):
        data[y] = (POINTER(Point2DArray) * width)()
        for x in range(width):
            # Initialize with empty arrays
            data[y][x] = point2d_array_new(times)

    grid.data = data
    return grid


def get_walk_points(walk):
    if not walk:
        raise ValueError("NULL walk pointer")

    walk_data = walk.contents
    if not walk_data.points:  # Check if points array is valid
        raise ValueError("NULL points array in walk")

    length = walk_data.length
    print(f"Debug: walk.length = {length}")  # Log length

    points = []
    for i in range(length):
        if i >= length:  # Extra safety check
            break
        point = walk_data.points[i]
        points.append((point.x, point.y))

    # (Optional) Disable free temporarily to test
    # dll.point2d_array_free(walk)
    result = np.array(points, dtype=np.int64)
    return result


def get_point2d(x, y):
    ptr = dll.point_2d_new(x, y)
    return ptr.contents


def create_point2d_array(steps) -> Point2DArray:  # type: ignore
    point_array = (Point2D * len(steps))()
    for i, (x, y) in enumerate(steps):
        point_array[i].x = x
        point_array[i].y = y

    array = Point2DArray()
    array.points = point_array
    array.length = len(steps)
    # Retain reference to prevent garbage collection
    array._buffer = point_array
    return array
