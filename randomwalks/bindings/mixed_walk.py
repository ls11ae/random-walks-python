from randomwalks.bindings.data_structures.Point2D import Point2DArrayHandle
from randomwalks.bindings.data_structures.Tensor import Tensor4DHandle
from randomwalks.bindings.data_structures.types import *
from randomwalks.wrapper import dll


class MixedWalkBinding:
    dll.m_walk.argtypes = [KernelContextPtr, c_ssize_t, c_ssize_t, c_ssize_t]
    dll.m_walk.restype = POINTER(TensorPtr)
    m_walk = dll.m_walk

    dll.m_walk_backtrace.argtypes = [
        POINTER(TensorPtr),
        c_ssize_t,
        KernelContextPtr,
        c_ssize_t,
        c_ssize_t,
    ]
    dll.m_walk_backtrace.restype = Point2DArrayPtr
    m_walk_backtrace = dll.m_walk_backtrace

    dll.mixed_utilization_distribution.argtypes = [
        POINTER(TensorPtr),
        c_ssize_t,
        KernelContextPtr,
        c_ssize_t,
        c_ssize_t,
    ]
    dll.mixed_utilization_distribution.restype = POINTER(TensorPtr)
    mixed_utilization_distribution = dll.mixed_utilization_distribution

    dll.single_state_walk.argtypes = [
        c_ssize_t,
        KernelContextPtr,
        c_ssize_t,
        c_ssize_t,
        c_ssize_t,
        c_ssize_t,
    ]
    dll.single_state_walk.restype = Point2DArrayPtr
    single_state_walk_ptr = dll.single_state_walk

    dll.time_walk_env_binary.argtypes = [
        c_size_t,
        KernelParametersMappingPtr,
        TerrainMapPtr,
        c_char_p,
        EnvWeightProfilePtr,
        TimedLocation,
        TimedLocation,
    ]
    dll.time_walk_env_binary.restype = Point2DArrayPtr
    time_walk_env_binary_ptr = dll.time_walk_env_binary

    @classmethod
    def walk(cls, kernel_context, T, start_x, start_y):
        ptr = cls.m_walk(_context_ptr(kernel_context), c_ssize_t(T), c_ssize_t(start_x), c_ssize_t(start_y))
        return Tensor4DHandle.from_ptr(ptr, T, owned=True)

    @classmethod
    def backtrace(cls, dp_matrix, kernel_context, end_x, end_y):
        dp_ptr = dp_matrix.ptr if hasattr(dp_matrix, "ptr") else dp_matrix
        T = dp_matrix.T if hasattr(dp_matrix, "T") else None
        if T is None:
            raise ValueError("Tensor4DHandle with a T value is required")

        walk_handle = Point2DArrayHandle.from_ptr(
            cls.m_walk_backtrace(dp_ptr, T, _context_ptr(kernel_context), end_x, end_y),
            owned=True,
        )
        return walk_handle.to_numpy()

    @classmethod
    def utilization_distribution(cls, dp_matrix, kernel_context, end_x, end_y):
        dp_ptr = dp_matrix.ptr if hasattr(dp_matrix, "ptr") else dp_matrix
        T = dp_matrix.T if hasattr(dp_matrix, "T") else None
        if T is None:
            raise ValueError("Tensor4DHandle with a T value is required")

        ptr = cls.mixed_utilization_distribution(dp_ptr, T, _context_ptr(kernel_context), end_x, end_y)
        return Tensor4DHandle.from_ptr(ptr, T, owned=True)

    @classmethod
    def single_state_walk(cls, kernel_context, T, start_x, start_y, end_x, end_y):
        ptr = cls.single_state_walk_ptr(
            c_ssize_t(T),
            _context_ptr(kernel_context),
            c_ssize_t(start_x),
            c_ssize_t(start_y),
            c_ssize_t(end_x),
            c_ssize_t(end_y),
        )
        if not ptr:
            return None
        walk_handle = Point2DArrayHandle.from_ptr(
            ptr,
            owned=True,
        )
        try:
            return walk_handle.to_numpy()
        finally:
            walk_handle.free()

    @classmethod
    def time_walk_env_binary(
        cls,
        *,
        T,
        mapping,
        terrain,
        env_binary_path,
        env_weights,
        start_point,
        end_point,
        start_time,
        end_time,
    ):
        ptr = cls.time_walk_env_binary_ptr(
            c_size_t(T),
            _mapping_ptr(mapping),
            _terrain_ptr(terrain),
            env_binary_path.encode("utf-8"),
            _env_weights_ptr(env_weights),
            _timed_location(start_point, start_time),
            _timed_location(end_point, end_time),
        )
        if not ptr:
            return None
        walk_handle = Point2DArrayHandle.from_ptr(
            ptr,
            owned=True,
        )
        try:
            return walk_handle.to_numpy()
        finally:
            walk_handle.free()


def _context_ptr(kernel_context):
    return kernel_context.ptr if hasattr(kernel_context, "ptr") else kernel_context


def _mapping_ptr(mapping):
    return mapping.ptr if hasattr(mapping, "ptr") else mapping


def _terrain_ptr(terrain):
    return terrain.ptr if hasattr(terrain, "ptr") else terrain


def _env_weights_ptr(env_weights):
    return env_weights.ptr if hasattr(env_weights, "ptr") else env_weights


def _timed_location(point, timestamp):
    import pandas as pd

    ts = pd.Timestamp(timestamp)
    return TimedLocation(
        timestamp=DateTime(ts.year, ts.month, ts.day, ts.hour),
        coordinates=Point2D(int(point[0]), int(point[1])),
    )
