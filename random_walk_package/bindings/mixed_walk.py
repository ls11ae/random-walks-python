# mixed_walk.py

from random_walk_package import point2d_arr_free, get_walk_points
from random_walk_package.bindings.data_structures.terrain import *
from random_walk_package.wrapper import dll

dll.m_walk.argtypes = [
    KernelContextPtr,
    c_ssize_t,  # T
    c_ssize_t,  # start x
    c_ssize_t
]
dll.m_walk.restype = POINTER(TensorPtr)

dll.tensor_set_free.argtypes = [TensorSetPtr]
dll.tensor_set_free.restype = None

dll.m_walk_backtrace.argtypes = [
    POINTER(TensorPtr),
    c_ssize_t,
    KernelContextPtr,
    c_ssize_t,
    c_ssize_t
]
dll.m_walk_backtrace.restype = Point2DArrayPtr

dll.mixed_utilization_distribution.argtypes = [
    POINTER(TensorPtr),
    c_ssize_t,
    KernelContextPtr,
    c_ssize_t,
    c_ssize_t
]
dll.mixed_utilization_distribution.restype = POINTER(TensorPtr)

dll.mixed_walk_time_compact.argtypes = [
    c_ssize_t,  # W
    c_ssize_t,  # H
    TerrainMapPtr,  # terrain
    DirKernelsMapPtr,
    KernelParametersMappingPtr,  # mapping
    KernelParametersTerrainWeatherPtr,  # terrain and weather parameters
    c_ssize_t,  # T
    c_ssize_t,  # start:x
    c_ssize_t,  # start:y
]
dll.mixed_walk_time_compact.restype = POINTER(TensorPtr)


def time_walk_geo(T, csv_path, terrain_path, grid_x, grid_y, start, goal, mapping=None, full_weather_influence=False):
    pass


def tensor_set_new(tensors):
    num_tensors = len(tensors)
    if num_tensors == 0:
        raise ValueError("At least one tensor must be provided.")

    tensor_array = (POINTER(Tensor) * num_tensors)(*tensors)

    tensor_set = dll.tensor_set_new(num_tensors, tensor_array)
    return tensor_set


def mix_walk(kernel_context, T, start_x, start_y):
    result = dll.m_walk(
        kernel_context, c_ssize_t(T), c_ssize_t(start_x), c_ssize_t(start_y))

    return result


def mix_backtrace_c(DP_Matrix, T, kernel_context, end_x, end_y):
    walk_c = dll.m_walk_backtrace(DP_Matrix, T, kernel_context, end_x, end_y)
    return walk_c


def mix_backtrace(DP_Matrix, T, kernel_context, end_x, end_y, dp_dir: str = "", mapping=None):
    if mapping is None:
        mapping = create_mixed_kernel_parameters(MEDIUM, 7)
    walk_c = dll.m_walk_backtrace(DP_Matrix, T, kernel_context, end_x, end_y)
    if walk_c is None:
        raise ValueError("Walk failed to backtrace. Maybe try again with higher T?")
    walk_np = get_walk_points(walk_c)
    point2d_arr_free(walk_c)
    return walk_np


def mix_utilization_distribution(DP_Matrix, T, kernel_context, end_x, end_y):
    return dll.mixed_utilization_distribution(DP_Matrix, T, kernel_context, end_x, end_y)


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
