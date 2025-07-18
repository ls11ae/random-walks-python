from ctypes import c_size_t

from random_walk_package.bindings.data_structures.point2D import *
from random_walk_package.bindings.data_structures.types import *
from random_walk_package.wrapper import dll


dll.extractLocationsFromCSV.argtypes = [ctypes.c_char_p]
dll.extractLocationsFromCSV.restype = CoordArray

dll.coordinate_array_new.argtypes = [CoordArray, ctypes.c_size_t]
dll.coordinate_array_new.restype = CoordArray

dll.getNormalizedLocations.argtypes = [CoordArray, ctypes.c_size_t, ctypes.c_size_t]
dll.getNormalizedLocations.restype = Point2DArrayPtr

dll.extractSteps.argtypes = [Point2DArrayPtr, ctypes.c_size_t]
dll.extractSteps.restype = Point2DArrayPtr

dll.coordinate_array_free.argtypes = [CoordArray]
dll.coordinate_array_free.restype = None


def extract_steps_from_csv(csv_file: str, num_steps, width, height) -> Point2DArrayPtr: # type: ignore
    coords = dll.extractLocationsFromCSV(csv_file.encode('ascii'))
    normalized = dll.getNormalizedLocations(coords, c_size_t(width), c_size_t(height))
    dll.coordinate_array_free(coords)
    extracted = dll.extractSteps(normalized, num_steps)
    dll.point2d_array_free(normalized)
    return extracted


