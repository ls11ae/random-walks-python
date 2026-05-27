from random_walk_package.bindings.data_structures.types import *
from random_walk_package.wrapper import dll

dll.kernel_parameters_create.argtypes = [c_bool,  # is brownian?
                                         c_ssize_t,  # step size
                                         c_ssize_t,  # directions
                                         c_float,  # diffusity
                                         c_float,  # diffusity
                                         c_ssize_t,  # max bias x
                                         c_ssize_t]  # max bias y
dll.kernel_parameters_create.restype = KernelParametersPtr


def create_kernel_parameters(is_brownian: bool, step_size: int, directions: int, len_diffusity: float = 1.0,
                             angle_diff: float = 0.3,
                             max_bias_x: int = 0,
                             max_bias_y: int = 0) -> KernelParametersPtr:
    return dll.kernel_parameters_create(is_brownian, step_size, directions, len_diffusity, angle_diff, max_bias_y,
                                        max_bias_x)
