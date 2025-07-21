# mixed_walk.py
import os

from random_walk_package.bindings.data_structures.types import *
from random_walk_package.wrapper import dll

dll.mixed_walk.argtypes = [
    ctypes.c_ssize_t,  # W
    ctypes.c_ssize_t,  # H
    TerrainMapPtr,  # spatial_map
    TensorMapPtr,  # tensor_map
    ctypes.POINTER(Tensor),  # c_kernel
    ctypes.c_ssize_t,  # T
    Point2DArrayPtr]
dll.mixed_walk.restype = Point2DArrayPtr

dll.m_walk.argtypes = [
    ctypes.c_ssize_t,  # W
    ctypes.c_ssize_t,  # H
    TerrainMapPtr,  # spatial_map
    TensorMapPtr,  # tensor_map
    ctypes.c_ssize_t,  # T
    ctypes.c_ssize_t,  # start x
    ctypes.c_ssize_t,
    ctypes.c_bool,
    ctypes.c_bool,
    ctypes.c_char_p]
dll.m_walk.restype = ctypes.POINTER(TensorPtr)

dll.tensor_set_free.argtypes = [TensorSetPtr]
dll.tensor_set_free.restype = None

dll.m_walk_backtrace.argtypes = [
    ctypes.POINTER(TensorPtr),
    ctypes.c_ssize_t,
    TensorMapPtr,
    TerrainMapPtr,
    ctypes.c_ssize_t,
    ctypes.c_ssize_t,
    ctypes.c_ssize_t,
    ctypes.c_bool,
    ctypes.c_char_p,
    ctypes.c_char_p
]
dll.m_walk_backtrace.restype = Point2DArrayPtr

dll.mixed_walk_time.argtypes = [
    ctypes.c_ssize_t,  # W
    ctypes.c_ssize_t,  # H
    TerrainMapPtr,  # terrain
    KernelsMap4DPtr,  # tensormap,
    ctypes.c_ssize_t,  # T
    ctypes.c_ssize_t,  # start:x
    ctypes.c_ssize_t  # start:y
]
dll.mixed_walk_time.restype = ctypes.POINTER(TensorPtr)

dll.backtrace_time_walk.argtypes = [
    ctypes.POINTER(TensorPtr),  # dp
    ctypes.c_ssize_t,  # T
    TerrainMapPtr,  # terrain
    KernelsMap4DPtr,
    ctypes.c_ssize_t,  # end_x
    ctypes.c_ssize_t,  # end_y
    ctypes.c_ssize_t  # dir
]
dll.backtrace_time_walk.restype = Point2DArrayPtr

# Python wrappers

# Wrap time_walk_geo
dll.time_walk_geo.argtypes = [
    ctypes.c_ssize_t,  # T
    ctypes.c_char_p,  # csv_path
    ctypes.c_char_p,  # terrain_path
    ctypes.c_char_p,  # walk_path
    ctypes.c_int,  # grid_x
    ctypes.c_int,  # grid_y
    Point2D,  # start
    Point2D  # goal
]
dll.time_walk_geo.restype = Point2DArrayPtr


def time_walk_geo(T, csv_path, terrain_path, walk_path, grid_x, grid_y, start, goal):
    """
    Calls the C function time_walk_geo to perform a time-dependent walk with geospatial data.
    Args:
        T (int): Number of time steps
        csv_path (str): Path to CSV file
        terrain_path (str): Path to terrain file
        grid_x (int): Grid width
        grid_y (int): Grid height
        W (int): Map width
        H (int): Map height
        start (tuple): Start point as (x, y)
        goal (tuple): Goal point as (x, y)
    Returns:
        Point2DArrayPtr: Pointer to the resulting walk
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_project_dir = os.path.join(script_dir, '..')
    resources_dir = os.path.join(base_project_dir, 'resources')

    terrain_path = os.path.join(resources_dir, terrain_path)
    csv_path = os.path.join(resources_dir, csv_path)
    walk_path = os.path.join(resources_dir, walk_path)

    # Convert Python tuples to Point2D structs
    if not (isinstance(start, (tuple, list)) and len(start) == 2):
        raise ValueError("start must be a tuple or list of length 2")
    if not (isinstance(goal, (tuple, list)) and len(goal) == 2):
        raise ValueError("goal must be a tuple or list of length 2")
    start_pt = Point2D(start[0], start[1])
    goal_pt = Point2D(goal[0], goal[1])
    print(csv_path.encode('utf-8'))
    return dll.time_walk_geo(
        ctypes.c_ssize_t(T),
        csv_path.encode('utf-8'),
        terrain_path.encode('utf-8'),
        walk_path.encode('utf-8'),
        ctypes.c_int(grid_x),
        ctypes.c_int(grid_y),
        start_pt,
        goal_pt
    )


# Wrap time_walk_geo_multi

dll.time_walk_geo_multi.argtypes = [
    ctypes.c_ssize_t,  # T
    ctypes.c_char_p,  # csv_path
    ctypes.c_char_p,  # terrain_path
    ctypes.c_char_p,  # walk_path
    ctypes.c_int,  # grid_x
    ctypes.c_int,  # grid_y
    Point2DArrayPtr  # steps
]
dll.time_walk_geo_multi.restype = Point2DArrayPtr

from random_walk_package.bindings.data_structures.point2D import create_point2d_array


def time_walk_geo_multi(T, csv_path, terrain_path, walk_path, grid_x, grid_y, steps):
    """
    Calls the C function time_walk_geo_multi to perform a time-dependent walk with multiple steps.
    Args:
        T (int): Number of time steps
        csv_path (str): Path to CSV file
        terrain_path (str): Path to terrain file
        walk_path (str): Path to output walk file
        grid_x (int): Grid width
        grid_y (int): Grid height
        steps (list of tuple): List of (x, y) tuples
    Returns:
        Point2DArrayPtr: Pointer to the resulting walk
    """
    steps_array = create_point2d_array(steps)
    return dll.time_walk_geo_multi(
        ctypes.c_ssize_t(T),
        csv_path.encode('utf-8'),
        terrain_path.encode('utf-8'),
        walk_path.encode('utf-8'),
        ctypes.c_int(grid_x),
        ctypes.c_int(grid_y),
        steps_array
    )


def mixed_walk(W, H, spatial_map, tensor_map, c_kernel, T, steps):
    return dll.c_walk_backtrace_multiple(T, W, H, c_kernel, spatial_map, tensor_map, steps)


def tensor_set_new(tensors):
    num_tensors = len(tensors)
    if num_tensors == 0:
        raise ValueError("At least one tensor must be provided.")

    tensor_array = (ctypes.POINTER(Tensor) * num_tensors)(*tensors)

    tensor_set = dll.tensor_set_new(num_tensors, tensor_array)
    return tensor_set


def mix_walk(W, H, terrain_map, kernels_map, T, start_x, start_y, serialize: bool, recompute: bool,
             serialize_path: str):
    result = dll.m_walk(
        ctypes.c_ssize_t(W),
        ctypes.c_ssize_t(H),
        terrain_map,
        kernels_map,
        ctypes.c_ssize_t(T),
        ctypes.c_ssize_t(start_x),
        ctypes.c_ssize_t(start_y),
        ctypes.c_bool(serialize),
        ctypes.c_bool(recompute),
        serialize_path.encode('utf-8'))

    return result


def mix_backtrace(DP_Matrix, T, tensor_map, terrain, end_x, end_y, dir, serialize: bool, serialize_path: str,
                  dp_dir: str):
    return dll.m_walk_backtrace(DP_Matrix, T, tensor_map, terrain, end_x, end_y, dir, serialize,
                                serialize_path.encode('utf-8'), dp_dir.encode('utf-8'))


def time_walk_init(W, H, terrain, tensormap, T, start_x, start_y):
    return dll.mixed_walk_time(W, H, terrain, tensormap, T, start_x, start_y)


def time_walk_backtrace(dp, T, terrain, kernels_map, end_x, end_y, init_dir):
    return dll.backtrace_time_walk(dp, T, terrain, kernels_map, end_x, end_y, init_dir)
