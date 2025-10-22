from random_walk_package.bindings.data_structures.types import *
from random_walk_package.wrapper import dll

dll.kernel_parameters_create.argtypes = [c_bool,  # is brownian?
                                         c_ssize_t,  # step size
                                         c_ssize_t,  # directions
                                         c_float,  # diffusity
                                         c_ssize_t,  # max bias x
                                         c_ssize_t]  # max bias y
dll.kernel_parameters_create.restype = KernelParametersPtr


def create_kernel_parameters(is_brownian: bool, step_size: int, directions: int, diffusity: float, max_bias_x: int,
                             max_bias_y: int) -> KernelParametersPtr:
    return dll.kernel_parameters_create(is_brownian, step_size, directions, diffusity, max_bias_y, max_bias_x)
