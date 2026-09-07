from randomwalks.bindings.data_structures.KernelContext import KernelContextHandle
from randomwalks.bindings.data_structures.Matrix import MatrixHandle
from randomwalks.bindings.data_structures.Point2D import Point2DArrayHandle
from randomwalks.bindings.data_structures.Tensor import Tensor4DHandle
from randomwalks.bindings.data_structures.types import *
from randomwalks.wrapper import dll


class MixedWalkBinding:
    dll.m_walk.argtypes = [KernelContextPtr, c_ssize_t, c_ssize_t, c_ssize_t]
    dll.m_walk.restype = POINTER(TensorPtr)
    m_walk = dll.m_walk

    gpu_m_walk = getattr(dll, "gpu_m_walk", None)
    if gpu_m_walk is not None:
        gpu_m_walk.argtypes = [KernelContextPtr, c_ssize_t, c_ssize_t, c_ssize_t]
        gpu_m_walk.restype = POINTER(TensorPtr)

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

    mixed_utilization_distribution_sum = getattr(
        dll,
        "mixed_utilization_distribution_sum",
        None,
    )
    if mixed_utilization_distribution_sum is not None:
        mixed_utilization_distribution_sum.argtypes = [
            POINTER(TensorPtr),
            c_ssize_t,
            KernelContextPtr,
            c_ssize_t,
            c_ssize_t,
        ]
        mixed_utilization_distribution_sum.restype = MatrixPtr

    gpu_mixed_utilization_distribution = getattr(
        dll,
        "gpu_mixed_utilization_distribution",
        None,
    )
    if gpu_mixed_utilization_distribution is not None:
        gpu_mixed_utilization_distribution.argtypes = [
            POINTER(TensorPtr),
            c_ssize_t,
            KernelContextPtr,
            c_ssize_t,
            c_ssize_t,
        ]
        gpu_mixed_utilization_distribution.restype = POINTER(TensorPtr)

    gpu_mixed_utilization_distribution_sum = getattr(
        dll,
        "gpu_mixed_utilization_distribution_sum",
        None,
    )
    if gpu_mixed_utilization_distribution_sum is not None:
        gpu_mixed_utilization_distribution_sum.argtypes = [
            POINTER(TensorPtr),
            c_ssize_t,
            KernelContextPtr,
            c_ssize_t,
            c_ssize_t,
        ]
        gpu_mixed_utilization_distribution_sum.restype = MatrixPtr

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
    def walk(cls, kernel_context: KernelContextHandle, T, start_x, start_y, *, cuda=False):
        print(f"[randomwalks] m_walk forward density: {'CUDA' if cuda else 'CPU'}", flush=True)
        walk = cls.gpu_m_walk if cuda else cls.m_walk
        if walk is None:
            raise RuntimeError(
                "CUDA mixed-walk support is unavailable; rebuild random_walk with ENABLE_CUDA=ON"
            )
        ptr = walk(kernel_context.ptr, c_ssize_t(T), c_ssize_t(start_x), c_ssize_t(start_y))
        if cuda and not ptr:
            raise RuntimeError("CUDA mixed forward-density calculation failed")
        return Tensor4DHandle.from_ptr(ptr, T, layer_count=T + 1)

    @classmethod
    def backtrace(cls, dp_matrix: Tensor4DHandle, kernel_context: KernelContextHandle, end_x, end_y):
        dp_ptr = dp_matrix.ptr
        T = dp_matrix.T if hasattr(dp_matrix, "T") else None
        if T is None:
            raise ValueError("Tensor4DHandle with a T value is required")
        walk_handle = None
        try:
            walk_handle = Point2DArrayHandle.from_ptr(cls.m_walk_backtrace(dp_ptr, T, kernel_context.ptr, end_x, end_y))
            return walk_handle.to_numpy()
        except Exception as e:
            print(f"Error in backtrace: {e}")
            print(f"Backtrace failed for end_x={end_x}, end_y={end_y}")
            return None
        finally:
            if walk_handle is not None:
                walk_handle.free()

    @classmethod
    def utilization_distribution(
            cls,
            dp_matrix: Tensor4DHandle,
            kernel_context: KernelContextHandle,
            end_x,
            end_y,
            *,
            cuda=False,
    ):
        print(
            f"[randomwalks] mixed utilization distribution: {'CUDA' if cuda else 'CPU'}",
            flush=True,
        )
        dp_ptr = dp_matrix.ptr
        T = dp_matrix.T if hasattr(dp_matrix, "T") else None
        if T is None:
            raise ValueError("Tensor4DHandle with a T value is required")

        calculate = (
            cls.gpu_mixed_utilization_distribution
            if cuda
            else cls.mixed_utilization_distribution
        )
        if calculate is None:
            raise RuntimeError(
                "CUDA mixed-walk UD support is unavailable; rebuild random_walk with ENABLE_CUDA=ON"
            )
        ptr = calculate(dp_ptr, T, kernel_context.ptr, end_x, end_y)
        if cuda and not ptr:
            raise RuntimeError("CUDA mixed utilization-distribution calculation failed")
        return Tensor4DHandle.from_ptr(ptr, T, layer_count=T + 1)

    @classmethod
    def utilization_distribution_sum(
            cls,
            dp_matrix: Tensor4DHandle,
            kernel_context: KernelContextHandle,
            end_x,
            end_y,
            *,
            cuda=False,
    ):
        """Return the averaged 2D UD without retaining backward time layers."""
        print(
            f"[randomwalks] mixed utilization distribution (streaming 2D reduction): "
            f"{'CUDA' if cuda else 'CPU'}",
            flush=True,
        )
        T = dp_matrix.T if hasattr(dp_matrix, "T") else None
        if T is None:
            raise ValueError("Tensor4DHandle with a T value is required")

        calculate = (
            cls.gpu_mixed_utilization_distribution_sum
            if cuda
            else cls.mixed_utilization_distribution_sum
        )
        if calculate is None:
            raise RuntimeError(
                "The memory-bounded mixed-walk UD reducer is unavailable; "
                "rebuild the random_walk native library"
            )
        ptr = calculate(dp_matrix.ptr, T, kernel_context.ptr, end_x, end_y)
        if not ptr:
            raise RuntimeError("Mixed utilization-distribution reduction failed")
        return MatrixHandle.from_ptr(ptr, owned=True)

    @classmethod
    def single_state_walk(cls, kernel_context: KernelContextHandle, T, start_x, start_y, end_x, end_y):
        print("[randomwalks] single-state walk: CPU", flush=True)
        ptr = cls.single_state_walk_ptr(
            c_ssize_t(T),
            kernel_context.ptr,
            c_ssize_t(start_x),
            c_ssize_t(start_y),
            c_ssize_t(end_x),
            c_ssize_t(end_y),
        )
        if not ptr:
            return None
        walk_handle = Point2DArrayHandle.from_ptr(ptr)
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
        walk_handle = Point2DArrayHandle.from_ptr(ptr)
        try:
            return walk_handle.to_numpy()
        finally:
            walk_handle.free()


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
