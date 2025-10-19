from random_walk_package.bindings.data_processing.movebank_parser import create_kernel_parameters
from random_walk_package.bindings.data_structures.types import *
from random_walk_package.wrapper import dll

dll.create_default_mixed_mapping.argtypes = [c_int, c_int]
dll.create_default_mixed_mapping.restype = KernelParametersMappingPtr

dll.create_default_brownian_mapping.argtypes = [c_int, c_int]
dll.create_default_brownian_mapping.restype = KernelParametersMappingPtr

dll.create_default_correlated_mapping.argtypes = [c_int, c_int]
dll.create_default_correlated_mapping.restype = KernelParametersMappingPtr

dll.set_landmark_mapping.argtypes = [KernelParametersMappingPtr, c_int, KernelParametersPtr]
dll.set_landmark_mapping.restype = None

dll.landmark_to_index.argtypes = [c_int]
dll.landmark_to_index.restype = c_int

dll.set_forbidden_landmark.argtypes = [KernelParametersMappingPtr, c_int]
dll.set_forbidden_landmark.restype = None

dll.is_forbidden_landmark.argtypes = [c_int, KernelParametersMappingPtr]
dll.is_forbidden_landmark.restype = c_bool

dll.get_parameters_of_terrain.argtypes = [KernelParametersMappingPtr, c_int]
dll.get_parameters_of_terrain.restype = KernelParametersPtr

dll.kernel_parameters_mapping_free.argtypes = [KernelParametersMappingPtr]
dll.kernel_parameters_mapping_free.restype = None


def kernel_mapping_free(kernel_parameters_map: KernelParametersMappingPtr): # pyright: ignore[reportInvalidTypeForm]
    dll.kernel_parameters_mapping_free(kernel_parameters_map)


def create_mixed_kernel_parameters(animal_type: int, base_step_size: int) -> KernelParametersMappingPtr: # pyright: ignore[reportInvalidTypeForm]
    return dll.create_default_mixed_mapping(animal_type, base_step_size)


def create_brownian_kernel_parameters(animal_type: int, base_step_size: int) -> KernelParametersMappingPtr: # pyright: ignore[reportInvalidTypeForm]
    return dll.create_default_brownian_mapping(animal_type, base_step_size)


def create_correlated_kernel_parameters(animal_type: int, base_step_size: int) -> KernelParametersMappingPtr: # pyright: ignore[reportInvalidTypeForm]
    return dll.create_default_correlated_mapping(animal_type, base_step_size)


def set_landmark_mapping(kernel_parameters_map: KernelParametersPtr, landmark: int,
                         is_brownian: bool, step_size: int, directions: int, diffusity: float,
                         max_bias_x: int | None = None, max_bias_y: int | None = None) -> None:
    if is_brownian:
        directions = 1  # Direction must be 1 for Brownian motion
    kernel_parameters = create_kernel_parameters(is_brownian=is_brownian,
                                                 step_size=step_size,
                                                 directions=directions,
                                                 diffusity=diffusity,
                                                 max_bias_x=max_bias_x if max_bias_x is not None else 0,
                                                 max_bias_y=max_bias_y if max_bias_y is not None else 0)
    dll.set_landmark_mapping(kernel_parameters_map, landmark, kernel_parameters)


def set_forbidden_landmark(kernel_parameters_map: KernelParametersPtr, landmark: int):
    dll.set_forbidden_landmark(kernel_parameters_map, landmark)


def is_forbidden_landmark(landmark: int, kernel_parameters_map: KernelParametersMappingPtr) -> bool:
    return dll.is_forbidden_landmark(landmark, kernel_parameters_map)
