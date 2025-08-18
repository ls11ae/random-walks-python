from ctypes import *

from random_walk_package.bindings.data_structures.point2D import *
from random_walk_package.bindings.data_structures.types import *
from random_walk_package.wrapper import dll


dll.extractLocationsFromCSV.argtypes = [c_char_p]
dll.extractLocationsFromCSV.restype = CoordArray

dll.coordinate_array_new.argtypes = [CoordArray, c_size_t]
dll.coordinate_array_new.restype = CoordArray

dll.getNormalizedLocations.argtypes = [CoordArray, c_size_t, c_size_t]
dll.getNormalizedLocations.restype = Point2DArrayPtr

dll.extractSteps.argtypes = [Point2DArrayPtr, c_size_t]
dll.extractSteps.restype = Point2DArrayPtr

dll.coordinate_array_free.argtypes = [CoordArray]
dll.coordinate_array_free.restype = None

dll.kernel_parameters_create.argtypes = [c_bool,    # is brownian?
                                         c_ssize_t, # step size
                                         c_ssize_t, # directions
                                         c_float,   # diffusity
                                         c_ssize_t, # max bias x
                                         c_ssize_t] # max bias y
dll.kernel_parameters_create.restype = KernelParametersPtr

def create_kernel_parameters(is_brownian: bool, step_size: int, directions: int, diffusity: float, max_bias_x: int, max_bias_y: int) -> KernelParametersPtr:
    return dll.kernel_parameters_create(is_brownian, step_size, directions, diffusity, max_bias_y, max_bias_x)

def extract_steps_from_csv(csv_file: str, num_steps, width, height) -> Point2DArrayPtr: # type: ignore
    coords = dll.extractLocationsFromCSV(csv_file.encode('ascii'))
    normalized = dll.getNormalizedLocations(coords, c_size_t(width), c_size_t(height))
    dll.coordinate_array_free(coords)
    extracted = dll.extractSteps(normalized, num_steps)
    dll.point2d_array_free(normalized)
    return extracted


