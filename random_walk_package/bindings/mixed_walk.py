# mixed_walk.py
import numpy as np

from random_walk_package.bindings.data_structures.point2D import point2d_arr_free, get_walk_points
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

dll.time_walk_custom.argtypes = [c_size_t, KernelParametersMappingPtr, TerrainMapPtr, c_char_p, EnvWeightProfilePtr,
                                 DateTimeIntervalPtr,
                                 POINTER(Dimensions3D),
                                 TimedLocation, TimedLocation]
dll.time_walk_custom.restype = Point2DArrayPtr

dll.time_walk_env_binary.argtypes = [c_size_t, KernelParametersMappingPtr, TerrainMapPtr, c_char_p, EnvWeightProfilePtr,
                                     TimedLocation, TimedLocation]
dll.time_walk_env_binary.restype = Point2DArrayPtr

dll.state_dep_walk.argtypes = [c_ssize_t, POINTER(c_int), TensorSetPtr, KernelParametersMappingPtr, TerrainMapPtr,
                               c_ssize_t, c_ssize_t, c_ssize_t, c_ssize_t]
dll.state_dep_walk.restype = Point2DArrayPtr

dll.single_state_walk.argtypes = [c_ssize_t, KernelsMap3DPtr, TerrainMapPtr,
                                  c_ssize_t, c_ssize_t, c_ssize_t, c_ssize_t]
dll.single_state_walk.restype = Point2DArrayPtr


def environment_mixed_walk(T, mapping, terrain, csv_path, dimensions, start_date, end_date,
                           start_point, end_point, env_weights):
    """Performs time‑based custom walk using environment data as csv"""
    start_dt = DateTime(start_date.year, start_date.month, start_date.day, start_date.hour)
    end_dt = DateTime(end_date.year, end_date.month, end_date.day, end_date.hour)
    interval_ptr = pointer(DateTimeInterval(start_dt, end_dt))
    dimensions_ptr = pointer(Dimensions3D(*dimensions))
    path_c = csv_path.encode('utf-8')
    tloc_start = TimedLocation(start_dt, Point2D(start_point[0], start_point[1]))
    tloc_end = TimedLocation(end_dt, Point2D(end_point[0], end_point[1]))
    return dll.time_walk_custom(T, mapping, terrain, path_c, env_weights, interval_ptr, dimensions_ptr, tloc_start,
                                tloc_end)


def env_mixed_walk(T, mapping, terrain, csv_path, start_date, end_date,
                   start_point, end_point, env_weights):
    """Performs time‑based custom walk using environment data as binary"""
    start_dt = DateTime(start_date.year, start_date.month, start_date.day, start_date.hour)
    end_dt = DateTime(end_date.year, end_date.month, end_date.day, end_date.hour)
    path_c = csv_path.encode('utf-8')
    tloc_start = TimedLocation(start_dt, Point2D(start_point[0], start_point[1]))
    tloc_end = TimedLocation(end_dt, Point2D(end_point[0], end_point[1]))
    return dll.time_walk_env_binary(T, mapping, terrain, path_c, env_weights, tloc_start, tloc_end)


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


def numpy_to_matrix(Z: np.ndarray) -> Matrix:
    Z = np.ascontiguousarray(Z, dtype=np.float64)

    data_ptr = Z.ctypes.data_as(POINTER(c_double))

    mat = Matrix()
    mat.width = Z.shape[1]
    mat.height = Z.shape[0]
    mat.len = Z.size
    mat.data.points = data_ptr

    return mat


def build_state_tensor(Z: np.ndarray) -> Tensor:
    mat = numpy_to_matrix(Z)

    mats = (POINTER(Matrix) * 1)()
    mats[0] = pointer(mat)

    tensor = Tensor()
    tensor.len = 1
    tensor.data = mats

    tensor._mat_ref = mat  # GC protection
    return tensor


dll.kernels_map_single.argtypes = [TerrainMapPtr, TensorPtr, KernelParametersMappingPtr]
dll.kernels_map_single.restype = KernelsMap3DPtr


def kernels_map_single(terrain, kernel, mapping):
    tensor_ptr = build_state_tensor(kernel)
    return dll.kernels_map_single(terrain, tensor_ptr, mapping)


def single_state_walk(T, kmap, terrain, start_x, start_y, end_x, end_y):
    return dll.single_state_walk(T, kmap, terrain, start_x, start_y, end_x, end_y)


def state_dep_walk(T, state, kernels, mapping, terrain, start_x, start_y, end_x, end_y):
    tensors = (POINTER(Tensor) * 3)()
    Za = kernels[0]
    Zb = kernels[1]
    Zc = kernels[2]

    t0 = build_state_tensor(Za)
    t1 = build_state_tensor(Zb)
    t2 = build_state_tensor(Zc)

    tensors[0] = pointer(t0)
    tensors[1] = pointer(t1)
    tensors[2] = pointer(t2)

    tensor_set = dll.tensor_set_new(3, tensors)
    tensor_set._tensor_refs = [t0, t1, t2]

    timeline = np.full(T, state, dtype=np.int32)
    timeline_ptr = timeline.ctypes.data_as(POINTER(c_int))
    return dll.state_dep_walk(T, timeline_ptr, tensor_set, mapping, terrain, start_x, start_y, end_x, end_y)
