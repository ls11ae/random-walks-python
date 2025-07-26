import ctypes
import os
from _ctypes import byref
from ctypes import c_char

import rasterio
from .matrix import *
from random_walk_package.wrapper import dll, script_dir


dll.kernels_map_new.argtypes = [TerrainMapPtr, MatrixPtr]
dll.kernels_map_new.restype = KernelsMapPtr

dll.tensor_map_new.argtypes = [TerrainMapPtr, TensorPtr]
dll.tensor_map_new.restype = TensorMapPtr

dll.kernel_at.argtypes = [KernelsMapPtr, ctypes.c_ssize_t, ctypes.c_ssize_t]
dll.kernel_at.restype = Matrix

dll.kernels_map_free.argtypes = [KernelsMapPtr]
dll.kernels_map_free.restype = None

dll.tensor_map_free.argtypes = [TensorMapPtr]
dll.tensor_map_free.restype = None

dll.get_terrain_map.argtypes = [ctypes.c_char_p, ctypes.c_char]
dll.get_terrain_map.restype = TerrainMapPtr

dll.terrain_at.argtypes = [ctypes.c_ssize_t, ctypes.c_ssize_t, TerrainMapPtr]
dll.terrain_at.restype = ctypes.c_int

dll.terrain_map_new.argtypes = [ctypes.c_ssize_t, ctypes.c_ssize_t]
dll.terrain_map_new.restype = TerrainMapPtr

dll.terrain_set.argtypes = [TerrainMapPtr, ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_int]
dll.terrain_set.restype = None

dll.tensor_map_terrain.argtypes = [TerrainMapPtr]
dll.tensor_map_terrain.restype = TensorMapPtr

dll.tensor_map_terrain_biased.argtypes = [TerrainMapPtr, Point2DArrayPtr]
dll.tensor_map_terrain_biased.restype = KernelsMap4DPtr

dll.parse_terrain_map.argtypes = [ctypes.c_char_p, TerrainMapPtr, ctypes.c_char]
dll.parse_terrain_map.restype = ctypes.c_int

dll.tensor_map_terrain_biased_grid.argtypes = [TerrainMapPtr, Point2DArrayGridPtr]
dll.tensor_map_terrain_biased_grid.restype = KernelsMap4DPtr

dll.tensor_map_terrain_biased_grid_serialized.argtypes = [TerrainMapPtr, Point2DArrayGridPtr,ctypes.c_char_p]
dll.tensor_map_terrain_biased_grid_serialized.restype = None

dll.tensor_map_terrain_serialize.argtypes = [TerrainMapPtr, ctypes.c_char_p]
dll.tensor_map_terrain_serialize.restype = None

dll.tensor_at.argtypes = [ctypes.c_char_p, ctypes.c_ssize_t, ctypes.c_ssize_t]
dll.tensor_at.restype = TensorPtr

dll.tensor_at_xyt.argtypes = [ctypes.c_char_p, ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_ssize_t]
dll.tensor_at_xyt.restype = TensorPtr

dll.kernels_map_serialized.argtypes = [TerrainMapPtr, MatrixPtr]
dll.kernels_map_serialized.restype = KernelsMapPtr

dll.kernels_map3d_free.argtypes = [KernelsMap3DPtr]
dll.kernels_map3d_free.restype = None

dll.kernels_map4d_free.argtypes = [KernelsMap4DPtr]
dll.kernels_map4d_free.restype = None

dll.tensor_map_free.argtypes = [ctypes.POINTER(KernelsMap3DPtr), ctypes.c_size_t]
dll.tensor_map_free.restype = None

dll.terrain_map_free.argtypes = [TerrainMapPtr]
dll.terrain_map_free.restype = None

dll.create_terrain_map.argtypes = [ctypes.c_char_p, ctypes.c_char]
dll.create_terrain_map.restype = TerrainMapPtr

dll.tensor_map_terrain_serialize_time.argtypes = [ctypes.c_void_p, TerrainMapPtr, ctypes.c_char_p]
dll.tensor_map_terrain_serialize_time.restype = None

# --- Neue/angepasste Python-Wrapper ---

def kernels_map_serialized(terrain: TerrainMapPtr, kernel: MatrixPtr) -> KernelsMapPtr:
    return dll.kernels_map_serialized(terrain, kernel)

def kernels_map3d_free(kernels_map3d: KernelsMap3DPtr):
    dll.kernels_map3d_free(kernels_map3d)

def kernels_map4d_free(kernels_map4d: KernelsMap4DPtr):
    dll.kernels_map4d_free(kernels_map4d)


def terrain_map_free(terrain_map: TerrainMapPtr):
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

def tensor_map_terrain_serialize_time(tensor_set_time, terrain: TerrainMapPtr, output_path: str):
    file = os.path.join(script_dir, 'resources', output_path)
    c_file = file.encode('ascii')
    dll.tensor_map_terrain_serialize_time(tensor_set_time, terrain, c_file)


def load_tensor_at(file, x, y):
    file = os.path.join(script_dir, 'resources', file)
    if not os.path.exists(file):
        raise FileNotFoundError(f"File '{file}' does not exist.")
    c_file = file.encode('ascii')
    return dll.tensor_at(c_file,x, y)

def load_tensor_at_xyt(file, x, y, t):
    file = os.path.join(script_dir, 'resources', file)
    if not os.path.exists(file):
        raise FileNotFoundError(f"File '{file}' does not exist.")
    c_file = file.encode('ascii')
    return dll.tensor_at_xyt(c_file,x, y, t)

def tensor_map_terrain_serialize(terrain: TerrainMapPtr, output_path: str):
    file = os.path.join(script_dir, 'resources', output_path)
    c_file = file.encode('ascii')
    return dll.tensor_map_terrain_serialize(terrain, c_file)

def tensor_map_terrain_time_serialize(terrain: TerrainMapPtr, grid: Point2DArrayGridPtr, output_path: str):
    file = os.path.join(script_dir, 'resources', output_path)
    c_file = file.encode('ascii')
    return dll.tensor_map_terrain_time_serialized(terrain, grid, c_file)

def tensor_map_terrain_bias(terrain: TerrainMapPtr, bias: Point2DArrayPtr) -> KernelsMap4DPtr: # type: ignore
    return dll.tensor_map_terrain_biased(terrain, bias)

def tensor_map_terrain_bias_grid(terrain: TerrainMapPtr, bias: Point2DArrayGridPtr) -> KernelsMap4DPtr: # type: ignore
    return dll.tensor_map_terrain_biased_grid(terrain, bias)


def parse_terrain(file: str, delim: str) -> TerrainMap:
    file = os.path.join(script_dir, 'resources', file)
    if not os.path.exists(file):
        raise FileNotFoundError(f"File '{file}' does not exist.")

    if len(delim) != 1:
        raise ValueError("Delimiter must be a single character.")

    c_file = file.encode('ascii')
    c_delim = c_char(delim.encode('ascii')[0])

    terrain = TerrainMap()

    result = dll.parse_terrain_map(c_file, byref(terrain), c_delim)
    if result != 0:
        raise RuntimeError(f"parse_terrain_map failed with error code {result}")
    print(f"Parsed terrain map from {file} with dimensions {terrain.width}x{terrain.height}")

    return terrain


def tensor_map_terrain_biased(terrain, bias_array) -> KernelsMap4DPtr: # type: ignore
    return dll.tensor_map_terrain_biased(terrain, bias_array)


def get_kernels_map(terrain, kernel) -> KernelsMapPtr: # type: ignore
    return dll.kernels_map_new(terrain, kernel)


def get_tensor_map(terrain, kernels) -> TensorMapPtr: # type: ignore
    return dll.tensor_map_new(terrain, kernels)


def get_tensor_map_mixed(terrain, tensors) -> TensorMapPtr: # type: ignore
    if not tensors:
        raise MemoryError("Failed to create TensorSet in C.")
    return dll.tensor_map_mixed(terrain, tensors)


def get_tensor_map_terrain(terrain):
    return dll.tensor_map_terrain(terrain)


def free_tensor_map(tensor_map):
    dll.tensor_map_free(tensor_map)


def kernels_map_free(kernels_map):
    dll.kernels_map_free(kernels_map)


def get_terrain_map(file, delim) -> TerrainMapPtr: # type: ignore
    file = os.path.join(script_dir, 'resources', file)
    if not file:
        raise FileNotFoundError
    if len(delim) != 1:
        raise ValueError("Delimiter must be a single character (e.g., ',')")

    c_delim = c_char(delim.encode('ascii')[0])

    # Call C function
    terrain_ptr = dll.get_terrain_map(file.encode('ascii'), c_delim)
    return terrain_ptr.contents


def terrain_at(terrain, x, y):
    terrain_ptr = ctypes.pointer(terrain)
    return dll.terrain_at(x, y, terrain_ptr)


def landcover_to_discrete_ptr(file_path, res_x, res_y, min_lon, max_lat, max_lon, min_lat, txt_output_path="terrain_movebank.txt") -> TerrainMapPtr:  # type: ignore
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

