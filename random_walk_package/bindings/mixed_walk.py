# mixed_walk.py

from random_walk_package import point2d_arr_free, get_walk_points
from random_walk_package.bindings.data_structures.terrain import *
from random_walk_package.wrapper import dll

dll.m_walk.argtypes = [
    c_ssize_t,  # W
    c_ssize_t,  # H
    TerrainMapPtr,  # spatial_map
    KernelParametersMappingPtr,  # mapping
    TensorMapPtr,  # tensor_map
    c_ssize_t,  # T
    c_ssize_t,  # start x
    c_ssize_t,
    c_bool,
    c_bool,
    c_char_p]
dll.m_walk.restype = POINTER(TensorPtr)

dll.tensor_set_free.argtypes = [TensorSetPtr]
dll.tensor_set_free.restype = None

dll.m_walk_backtrace.argtypes = [
    POINTER(TensorPtr),
    c_ssize_t,
    TensorMapPtr,
    TerrainMapPtr,
    KernelParametersMappingPtr,  # mapping
    c_ssize_t,
    c_ssize_t,
    c_ssize_t,
    c_bool,
    c_char_p,
    c_char_p
]
dll.m_walk_backtrace.restype = Point2DArrayPtr

dll.mixed_walk_time_compact.argtypes = [
    c_ssize_t,  # W
    c_ssize_t,  # H
    TerrainMapPtr,  # terrain
    DirKernelsMapPtr,
    KernelParametersMappingPtr,  # mapping
    KernelParamsXYTPtr,  # terrain and weather parameters
    c_ssize_t,  # T
    c_ssize_t,  # start:x
    c_ssize_t,  # start:y
]
dll.mixed_walk_time_compact.restype = POINTER(TensorPtr)

# Wrap time_walk_geo
dll.time_walk_geo_compact.argtypes = [
    c_ssize_t,  # T
    c_char_p,  # csv_path
    c_char_p,  # terrain_path
    KernelParametersMappingPtr,  # mapping
    c_int,  # grid_x
    c_int,  # grid_y
    TimedLocation,  # start dated location
    TimedLocation,  # goal dated location
    c_bool  # full weather influence
]
dll.time_walk_geo_compact.restype = Point2DArrayPtr

dll.time_walk_custom.argtypes = [c_ssize_t, KernelParametersMappingPtr, TerrainMapPtr, TimedLocation, TimedLocation]
dll.time_walk_custom.restype = Point2DArrayPtr


def time_walk_geo(T, csv_path, terrain_path, grid_x, grid_y, start, goal, mapping=None, full_weather_influence=False):
    """
    Calls the C function time_walk_geo to perform a time-dependent walk with geospatial data.

    Args:
        T (int): Number of time steps.
        csv_path (str): Path to CSV file.
        terrain_path (str): Path to terrain file.
        grid_x (int): Grid width.
        grid_y (int): Grid height.
        start (tuple[int, int]): Start point as (x, y).
        goal (tuple[int, int]): Goal point as (x, y).
        mapping: Optional KernelParametersMapping; if None, defaults to create_mixed_kernel_parameters(MEDIUM, 7).
        full_weather_influence (bool): Whether to use full weather influence.
    Returns:
        Point2DArrayPtr: Pointer to the resulting walk.
    """
    _script_dir = os.path.dirname(os.path.realpath(__file__))
    base_project_dir = os.path.join(_script_dir, '..')
    resources_dir = os.path.join(base_project_dir, 'resources')

    terrain_path = os.path.join(resources_dir, terrain_path)
    csv_path = os.path.join(resources_dir, csv_path)

    if mapping is None:
        mapping = create_mixed_kernel_parameters(MEDIUM, 7)

    return dll.time_walk_geo_compact(
        c_ssize_t(T),
        csv_path.encode('utf-8'),
        terrain_path.encode('utf-8'),
        mapping,
        c_int(grid_x),
        c_int(grid_y),
        start,
        goal,
        c_bool(full_weather_influence)
    )


def tensor_set_new(tensors):
    num_tensors = len(tensors)
    if num_tensors == 0:
        raise ValueError("At least one tensor must be provided.")

    tensor_array = (POINTER(Tensor) * num_tensors)(*tensors)

    tensor_set = dll.tensor_set_new(num_tensors, tensor_array)
    return tensor_set


def mix_walk(W, H, terrain_map, kernels_map, T, start_x, start_y, serialize: bool, recompute: bool,
             serialize_path: str, mapping=None):
    if mapping is None:
        mapping = create_mixed_kernel_parameters(MEDIUM, 7)

    result = dll.m_walk(
        c_ssize_t(W),
        c_ssize_t(H),
        terrain_map,
        mapping,
        kernels_map,
        c_ssize_t(T),
        c_ssize_t(start_x),
        c_ssize_t(start_y),
        c_bool(serialize),
        c_bool(recompute),
        serialize_path.encode('utf-8') if serialize else None)

    return result


def mix_backtrace_c(DP_Matrix, T, tensor_map, terrain, end_x, end_y, serialize: bool = False, serialize_path: str = "",
                    dp_dir: str = "", mapping=None):
    walk_c = dll.m_walk_backtrace(DP_Matrix, T, tensor_map, terrain, mapping, end_x, end_y, 0, serialize,
                                  serialize_path.encode('utf-8'), dp_dir.encode('utf-8'))
    return walk_c


def mix_backtrace(DP_Matrix, T, tensor_map, terrain, end_x, end_y, serialize: bool = False, serialize_path: str = "",
                  dp_dir: str = "", mapping=None):
    if mapping is None:
        mapping = create_mixed_kernel_parameters(MEDIUM, 7)
    walk_c = dll.m_walk_backtrace(DP_Matrix, T, tensor_map, terrain, mapping, end_x, end_y, 0, serialize,
                                  serialize_path.encode('utf-8'), dp_dir.encode('utf-8'))
    if walk_c is None:
        raise ValueError("Walk failed to backtrace. Maybe try again with higher T?")
    walk_np = get_walk_points(walk_c)
    point2d_arr_free(walk_c)
    return walk_np


def time_walk_init(W, H, terrain, tensormap, T, start_x, start_y, use_serialized=False, serialization_path='',
                   mapping=None):
    if mapping is None:
        mapping = create_mixed_kernel_parameters(MEDIUM, 7)
    return dll.mixed_walk_time(
        c_ssize_t(W), c_ssize_t(H),
        terrain,
        mapping,
        tensormap,
        c_ssize_t(T),
        c_ssize_t(start_x),
        c_ssize_t(start_y),
        c_bool(use_serialized),
        serialization_path.encode('utf-8')
    )


def time_walk_backtrace(dp, T, terrain, kernels_map, end_x, end_y, init_dir, use_serialized=False,
                        serialization_path='', mapping=None):
    if mapping is None:
        mapping = create_mixed_kernel_parameters(MEDIUM, 7)
    return dll.backtrace_time_walk(
        dp,
        c_ssize_t(T),
        terrain,
        mapping,
        kernels_map,
        c_ssize_t(end_x),
        c_ssize_t(end_y),
        c_ssize_t(init_dir),
        c_bool(use_serialized),
        serialization_path.encode('utf-8')
    )
