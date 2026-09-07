from randomwalks.bindings.data_structures.Point2D import Point2DArrayHandle
from randomwalks.bindings.data_structures.Tensor import Tensor4DHandle, TensorHandle
from randomwalks.bindings.data_structures.types import *
from randomwalks.wrapper import dll


class CorrelatedWalkBinding:
    dll.correlated_init.argtypes = [
        c_ssize_t,
        c_ssize_t,
        TensorPtr,
        c_ssize_t,
        c_ssize_t,
        c_ssize_t,
        c_bool,
        c_char_p,
    ]
    dll.correlated_init.restype = POINTER(TensorPtr)
    init_ptr = dll.correlated_init

    dll.correlated_visit.argtypes = [
        c_ssize_t,
        c_ssize_t,
        TensorPtr,
        c_ssize_t,
        c_ssize_t,
        c_ssize_t,
        POINTER(c_bool),
        c_bool,
        c_char_p,
    ]
    dll.correlated_visit.restype = POINTER(TensorPtr)
    visit_ptr = dll.correlated_visit

    dll.correlated_backtrace.argtypes = [
        c_bool,
        POINTER(TensorPtr),
        c_char_p,
        c_ssize_t,
        TensorPtr,
        c_ssize_t,
        c_ssize_t,
        c_ssize_t,
    ]
    dll.correlated_backtrace.restype = Point2DArrayPtr
    backtrace_ptr = dll.correlated_backtrace

    dll.correlated_utilization_distribution.argtypes = [
        c_bool,
        POINTER(TensorPtr),
        c_char_p,
        c_ssize_t,
        TensorPtr,
        c_ssize_t,
        c_ssize_t,
        c_char_p,
    ]
    dll.correlated_utilization_distribution.restype = TensorPtr
    utilization_ptr = dll.correlated_utilization_distribution

    dll.correlated_multi_step.argtypes = [
        c_ssize_t,
        c_ssize_t,
        c_char_p,
        c_ssize_t,
        TensorPtr,
        Point2DArrayPtr,
        c_ssize_t,
        c_bool,
    ]
    dll.correlated_multi_step.restype = Point2DArrayPtr
    multi_step_ptr = dll.correlated_multi_step

    @classmethod
    def generate(cls, kernels, width, height, time, start_x, start_y,
                 *, use_serialization=False, output_folder=None):
        ptr = cls.init_ptr(
            width,
            height,
            _tensor_ptr(kernels),
            time,
            start_x,
            start_y,
            use_serialization,
            _optional_path(output_folder),
        )
        return None if use_serialization else Tensor4DHandle.from_ptr(
            ptr,
            time,
            layer_count=time + 1,
        )

    @classmethod
    def backtrace(cls, dp_matrix, kernels, time, end_x, end_y,
                  *, direction=0, use_serialization=False, dp_folder=None):
        walk = Point2DArrayHandle.from_ptr(
            cls.backtrace_ptr(
                use_serialization,
                _tensor4d_ptr(dp_matrix),
                _optional_path(dp_folder),
                time,
                _tensor_ptr(kernels),
                end_x,
                end_y,
                direction,
            ),
            owned=True,
        )
        return walk.to_numpy()

    @classmethod
    def utilization_distribution(cls, dp_matrix, kernels, time, end_x, end_y,
                                 *, use_serialization=False, dp_folder=None, output_folder=None):
        ptr = cls.utilization_ptr(
            use_serialization,
            _tensor4d_ptr(dp_matrix),
            _optional_path(dp_folder),
            time,
            _tensor_ptr(kernels),
            end_x,
            end_y,
            _optional_path(output_folder),
        )
        return TensorHandle.from_ptr(ptr, owned=True)

    @classmethod
    def multi_step(cls, width, height, time, kernels, steps, *,
                   direction=0, use_serialization=False, dp_folder=None):
        step_handle = Point2DArrayHandle.from_points(steps)
        try:
            walk = Point2DArrayHandle.from_ptr(
                cls.multi_step_ptr(
                    width,
                    height,
                    _optional_path(dp_folder),
                    time,
                    _tensor_ptr(kernels),
                    step_handle.ptr,
                    direction,
                    use_serialization,
                ),
                owned=True,
            )
            return walk.to_numpy()
        finally:
            step_handle.free()


def _optional_path(path):
    return path.encode("utf-8") if path is not None else None


def _tensor_ptr(tensor):
    return tensor.ptr if hasattr(tensor, "ptr") else tensor


def _tensor4d_ptr(tensor):
    return tensor.ptr if hasattr(tensor, "ptr") else tensor
