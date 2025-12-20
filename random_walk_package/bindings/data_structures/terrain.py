import ctypes
import os

import rasterio

from random_walk_package.bindings.data_structures.types import *
from random_walk_package.wrapper import dll, script_dir
from .kernel_terrain_mapping import create_mixed_kernel_parameters
from ..mixed_walk import build_state_tensor

# landcover types
TREE_COVER = 10
SHRUBLAND = 20
GRASSLAND = 30
CROPLAND = 40
BUILT_UP = 50
SPARSE_VEGETATION = 60
SNOW_AND_ICE = 70
WATER = 80
HERBACEOUS_WETLAND = 90
MANGROVES = 95
MOSS_AND_LICHEN = 100

# animal types
AIRBORNE = 0
AMPHIBIAN = 1
LIGHT = 2
MEDIUM = 3
HEAVY = 4

dll.get_terrain_map.argtypes = [ctypes.c_char_p, ctypes.c_char]
dll.get_terrain_map.restype = TerrainMapPtr

dll.terrain_at.argtypes = [ctypes.c_ssize_t, ctypes.c_ssize_t, TerrainMapPtr]
dll.terrain_at.restype = ctypes.c_int

dll.terrain_map_new.argtypes = [ctypes.c_ssize_t, ctypes.c_ssize_t]
dll.terrain_map_new.restype = TerrainMapPtr

dll.terrain_set.argtypes = [TerrainMapPtr, ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_int]
dll.terrain_set.restype = None

dll.tensor_map_terrain.argtypes = [TerrainMapPtr, KernelParametersMappingPtr]
dll.tensor_map_terrain.restype = KernelsMap3DPtr

dll.parse_terrain_map.argtypes = [ctypes.c_char_p, TerrainMapPtr, ctypes.c_char]
dll.parse_terrain_map.restype = ctypes.c_int

dll.tensor_map_terrain_serialize.argtypes = [TerrainMapPtr, KernelParametersMappingPtr, ctypes.c_char_p]
dll.tensor_map_terrain_serialize.restype = None

dll.tensor_at.argtypes = [ctypes.c_char_p, ctypes.c_ssize_t, ctypes.c_ssize_t]
dll.tensor_at.restype = TensorPtr

dll.kernels_map3d_free.argtypes = [KernelsMap3DPtr]
dll.kernels_map3d_free.restype = None

dll.terrain_map_free.argtypes = [TerrainMapPtr]
dll.terrain_map_free.restype = None

dll.create_terrain_map.argtypes = [ctypes.c_char_p, ctypes.c_char]
dll.create_terrain_map.restype = TerrainMapPtr

dll.terrain_single_value.argtypes = [ctypes.c_int, ctypes.c_ssize_t, ctypes.c_ssize_t]
dll.terrain_single_value.restype = TerrainMapPtr


def terrain_single_value(land_type: int, width: int, height: int):
    return dll.terrain_single_value(land_type, width, height)


def kernels_map3d_free(kernels_map3d: KernelsMap3DPtr):
    dll.kernels_map3d_free(kernels_map3d)


def kernels_map4d_free(kernels_map4d: KernelsMap4DPtr):
    dll.kernels_map4d_free(kernels_map4d)


def terrain_map_free(terrain_map: TerrainMapPtr):
    if terrain_map is not None:
        dll.terrain_map_free(terrain_map)


def create_terrain_map(file: str, delim: str) -> TerrainMapPtr:
    file = os.path.join(script_dir, 'resources', file)
    if not os.path.exists(file):
        raise FileNotFoundError(f"File '{file}' does not exist.")
    if len(delim) != 1:
        raise ValueError("Delimiter must be a single character.")
    c_file = file.encode('ascii')
    c_delim = c_char(delim.encode('ascii')[0])
    return dll.create_terrain_map(c_file, c_delim)


def tensor_map_terrain_serialize_time(tensor_set_time, terrain: TerrainMapPtr,
                                      mapping: KernelParametersMappingPtr | None = None, output_path: str = ""):
    file = os.path.join(script_dir, 'resources', output_path)
    c_file = file.encode('ascii')
    if mapping is None:
        mapping = create_mixed_kernel_parameters(animal_type=MEDIUM, base_step_size=7)
    dll.tensor_map_terrain_serialize_time(tensor_set_time, terrain, mapping, c_file)


def load_tensor_at(file, x, y):
    file = os.path.join(script_dir, 'resources', file)
    if not os.path.exists(file):
        raise FileNotFoundError(f"File '{file}' does not exist.")
    c_file = file.encode('ascii')
    return dll.tensor_at(c_file, x, y)


def load_tensor_at_xyt(file, x, y, t):
    file = os.path.join(script_dir, 'resources', file)
    if not os.path.exists(file):
        raise FileNotFoundError(f"File '{file}' does not exist.")
    c_file = file.encode('ascii')
    return dll.tensor_at_xyt(c_file, x, y, t)


def tensor_map_terrain_serialize(terrain: TerrainMapPtr, mapping: KernelParametersMappingPtr, output_path: str) -> None:
    dll.tensor_map_terrain_serialize(terrain, mapping, output_path.encode('utf-8'))


def tensor_map_terrain_time_serialize(terrain: TerrainMapPtr, grid: Point2DArrayGridPtr, output_path: str):
    file = os.path.join(script_dir, 'resources', output_path)
    c_file = file.encode('ascii')
    return dll.tensor_map_terrain_time_serialized(terrain, grid, c_file)


def tensor_map_terrain_biased(terrain: TerrainMapPtr, mapping: KernelParametersMappingPtr,
                              biases: Point2DArrayPtr) -> KernelsMap4DPtr:  # type: ignore
    return dll.tensor_map_terrain_biased(terrain, biases, mapping)


def tensor_map_terrain_biased_grid(terrain: TerrainMapPtr, biases: Point2DArrayGridPtr,
                                   mapping: KernelParametersMappingPtr) -> KernelsMap4DPtr:  # type: ignore
    return dll.tensor_map_terrain_biased_grid(terrain, biases, mapping)


def tensor_map_terrain_biased_grid_serialized(terrain: TerrainMapPtr, biases: Point2DArrayGridPtr,
                                              mapping: KernelParametersMappingPtr, output_path: str) -> None:
    dll.tensor_map_terrain_biased_grid_serialized(terrain, biases, mapping, output_path.encode('utf-8'))


def parse_terrain(file: str, delim: str) -> TerrainMap:
    if len(delim) != 1:
        raise ValueError("Delimiter must be a single character.")

    c_file = file.encode('ascii')
    c_delim = c_char(delim.encode('ascii')[0])

    result = dll.create_terrain_map(c_file, c_delim)
    print(f"Parsed terrain map from {file} with dimensions {result.contents.width}x{result.contents.height}")

    return result


def terrain_at(terrain, x, y):
    return dll.terrain_at(x, y, terrain)


def tensor_map_terrain_biased(terrain, bias_array, kernel_mapping) -> KernelsMap4DPtr:  # type: ignore
    return dll.tensor_map_terrain_biased(terrain, kernel_mapping, bias_array)


def get_tensor_map_mixed(terrain, tensor_mapping) -> TensorMapPtr:  # type: ignore
    if not tensor_mapping:
        raise MemoryError("Failed to create TensorSet in C.")
    return dll.tensor_map_mixed(terrain, tensor_mapping)


def get_tensor_map_terrain(terrain: TerrainMapPtr,
                           mapping: KernelParametersMappingPtr) -> KernelsMap3DPtr:  # type: ignore
    return dll.tensor_map_terrain(terrain, mapping)


def get_terrain_map(file, delim) -> TerrainMapPtr:  # type: ignore
    file = os.path.join(script_dir, 'resources', file)
    if not file:
        raise FileNotFoundError
    if len(delim) != 1:
        raise ValueError("Delimiter must be a single character (e.g., ',')")

    c_delim = c_char(delim.encode('ascii')[0])

    # Call C function
    terrain_ptr = dll.get_terrain_map(file.encode('ascii'), c_delim)
    return terrain_ptr.contents


def landcover_to_discrete_ptr(file_path, res_x, res_y, min_lon, max_lat, max_lon, min_lat,
                              txt_output_path="terrain_movebank.txt") -> TerrainMapPtr | None:  # type: ignore
    try:
        with rasterio.open(file_path) as src:
            landcover_array = src.read(1)
            array_height, array_width = landcover_array.shape

            # Calculate raster indices for the bounding box coordinates
            row_start, col_start = src.index(min_lon, max_lat)
            row_stop, col_stop = src.index(max_lon, min_lat)

            # Ensure start <= stop for rows and columns
            if row_start > row_stop:
                row_start, row_stop = row_stop, row_start
            if col_start > col_stop:
                col_start, col_stop = col_stop, col_start

            # Clamp the indices to valid ranges
            row_start = max(0, min(row_start, array_height - 1))
            row_stop = max(0, min(row_stop, array_height - 1))
            col_start = max(0, min(col_start, array_width - 1))
            col_stop = max(0, min(col_stop, array_width - 1))

            # Calculate the number of rows and columns in the ROI
            roi_rows = row_stop - row_start
            roi_cols = col_stop - col_start

            # Avoid division by zero when resolution is 1
            step_y = roi_rows / (res_y - 1) if res_y > 1 else 0
            step_x = roi_cols / (res_x - 1) if res_x > 1 else 0

            # Generate the grid as a 2D list
            grid = []
            for y_idx in range(res_y):
                r = row_start + int(y_idx * step_y)
                r = max(row_start, min(r, row_stop))
                r = min(r, array_height - 1)

                row = []
                for x_idx in range(res_x):
                    c = col_start + int(x_idx * step_x)
                    c = max(col_start, min(c, col_stop))
                    c = min(c, array_width - 1)

                    row.append(int(landcover_array[r, c]))
                grid.append(row)

            terrain_ptr = dll.terrain_map_new(res_x, res_y)

            for y in range(res_y):
                for x in range(res_x):
                    dll.terrain_set(terrain_ptr, x, y, grid[y][x])

            # Optional: write to .txt
            if txt_output_path:
                with open(txt_output_path, 'w') as f:
                    for row in grid:
                        f.write(' '.join(map(str, row)) + '\n')

            return terrain_ptr
    except rasterio.RasterioIOError as e:
        print(f"Error opening the file: {e}")
        return None
