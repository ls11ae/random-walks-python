import ctypes
import os

from random_walk_package.bindings.data_structures.types import *
from random_walk_package.wrapper import dll, script_dir
from .kernel_terrain_mapping import create_mixed_kernel_parameters

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

dll.kernels_map_new.argtypes = [TerrainMapPtr, KernelParametersMappingPtr, MatrixPtr]
dll.kernels_map_new.restype = KernelsMapPtr

dll.tensor_map_new.argtypes = [TerrainMapPtr, KernelParametersMappingPtr, TensorPtr]
dll.tensor_map_new.restype = KernelsMap3DPtr

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

dll.tensor_map_terrain.argtypes = [TerrainMapPtr, KernelParametersMappingPtr]
dll.tensor_map_terrain.restype = KernelsMap3DPtr

dll.tensor_map_terrain_biased.argtypes = [TerrainMapPtr, Point2DArrayPtr, KernelParametersMappingPtr]
dll.tensor_map_terrain_biased.restype = KernelsMap4DPtr

dll.parse_terrain_map.argtypes = [ctypes.c_char_p, TerrainMapPtr, ctypes.c_char]
dll.parse_terrain_map.restype = ctypes.c_int

dll.tensor_map_terrain_biased_grid.argtypes = [TerrainMapPtr, Point2DArrayGridPtr, KernelParametersMappingPtr]
dll.tensor_map_terrain_biased_grid.restype = KernelsMap4DPtr

dll.tensor_map_terrain_biased_grid_serialized.argtypes = [
    TerrainMapPtr,
    Point2DArrayGridPtr,
    KernelParametersMappingPtr,
    ctypes.c_char_p,
]
dll.tensor_map_terrain_biased_grid_serialized.restype = None

dll.tensor_map_terrain_serialize.argtypes = [TerrainMapPtr, KernelParametersMappingPtr, ctypes.c_char_p]
dll.tensor_map_terrain_serialize.restype = None

dll.tensor_at.argtypes = [ctypes.c_char_p, ctypes.c_ssize_t, ctypes.c_ssize_t]
dll.tensor_at.restype = TensorPtr

dll.tensor_at_xyt.argtypes = [ctypes.c_char_p, ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_ssize_t]
dll.tensor_at_xyt.restype = TensorPtr

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

dll.tensor_map_terrain_serialize_time.argtypes = [ctypes.c_void_p, TerrainMapPtr, KernelParametersMappingPtr,
                                                  ctypes.c_char_p]
dll.tensor_map_terrain_serialize_time.restype = None

dll.terrain_single_value.argtypes = [ctypes.c_int, ctypes.c_ssize_t, ctypes.c_ssize_t]
dll.terrain_single_value.restype = TerrainMapPtr

dll.upscale_terrain_map.argtypes = [TerrainMapPtr, ctypes.c_double]
dll.upscale_terrain_map.restype = TerrainMapPtr

dll.upscale_terrain_map.argtypes = [TerrainMapPtr, c_ssize_t, c_ssize_t, c_ssize_t, c_ssize_t]
dll.upscale_terrain_map.restype = TerrainMapPtr


def extract_terrain_from_endpoints(terrain: TerrainMapPtr, start_x, start_y, end_x, end_y) -> TerrainMapPtr:
    return dll.extract_terrain_from_endpoints(terrain, start_x, start_y, end_x, end_y)


def upscale_terrain(terrain: TerrainMapPtr, factor: float) -> TerrainMapPtr:
    return dll.upscale_terrain_map(terrain, factor)


def terrain_single_value(land_type: int, width: int, height: int):
    return dll.terrain_single_value(land_type, width, height)


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


def tensor_map_terrain_biased(terrain: TerrainMapPtr, biases: Point2DArrayPtr,
                              mapping: KernelParametersMappingPtr) -> KernelsMap4DPtr:  # type: ignore
    return dll.tensor_map_terrain_biased(terrain, biases, mapping)


def tensor_map_terrain_biased_grid(terrain: TerrainMapPtr, biases: Point2DArrayGridPtr,
                                   mapping: KernelParametersMappingPtr) -> KernelsMap4DPtr:  # type: ignore
    return dll.tensor_map_terrain_biased_grid(terrain, biases, mapping)


def tensor_map_terrain_biased_grid_serialized(terrain: TerrainMapPtr, biases: Point2DArrayGridPtr,
                                              mapping: KernelParametersMappingPtr, output_path: str) -> None:
    dll.tensor_map_terrain_biased_grid_serialized(terrain, biases, mapping, output_path.encode('utf-8'))


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


def tensor_map_terrain_biased(terrain, bias_array) -> KernelsMap4DPtr:  # type: ignore
    return dll.tensor_map_terrain_biased(terrain, bias_array)


def get_kernels_map(terrain: TerrainMapPtr, mapping: KernelParametersMappingPtr,
                    kernel: MatrixPtr) -> KernelsMapPtr:  # type: ignore
    return dll.kernels_map_new(terrain, mapping, kernel)


def get_tensor_map(terrain: TerrainMapPtr, mapping: KernelParametersMappingPtr,
                   kernels: TensorPtr) -> KernelsMap3DPtr:  # type: ignore
    return dll.tensor_map_new(terrain, mapping, kernels)


def get_tensor_map_mixed(terrain, tensors) -> TensorMapPtr:  # type: ignore
    if not tensors:
        raise MemoryError("Failed to create TensorSet in C.")
    return dll.tensor_map_mixed(terrain, tensors)


def get_tensor_map_terrain(terrain: TerrainMapPtr,
                           mapping: KernelParametersMappingPtr) -> KernelsMap3DPtr:  # type: ignore
    return dll.tensor_map_terrain(terrain, mapping)


def free_tensor_map(tensor_map):
    dll.tensor_map_free(tensor_map)


def kernels_map_free(kernels_map):
    dll.kernels_map_free(kernels_map)


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


def terrain_at(terrain, x, y):
    terrain_ptr = ctypes.pointer(terrain)
    return dll.terrain_at(x, y, terrain_ptr)


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


import rasterio


def extract_terrain_map(file_path, x1, y1, x2, y2, res_x, res_y, padding, min_abs_padding=50):
    """
    Extract a TerrainMap from a UTM-projected GeoTIFF with optional padding.
    Also returns grid coordinates of the input points mapped to the TerrainMap.

    Parameters
    ----------
    file_path : str
        Path to UTM-projected GeoTIFF.
    x1, y1, x2, y2 : float
        UTM coordinates of the two endpoints.
    res_x, res_y : int
        Grid resolution for the TerrainMap.
    padding : float
        Relative padding factor (e.g. 0.05 for 5% extra space on each side).
    dll : ctypes.CDLL
        Loaded C library with terrain_map_new() and terrain_set().

    Returns
    -------
    terrain_ptr : POINTER(TerrainMap)

    (gx1, gy1), (gx2, gy2): tuple[int, int], tuple[int, int]
        Grid indices (col,row) of the two endpoints within the TerrainMap.
    """
    with rasterio.open(file_path) as src:
        # check CRS
        print("Raster CRS:", src.crs)
        print("Raster bounds:", src.bounds)

        # normalize order
        min_x, max_x = sorted([x1, x2])
        min_y, max_y = sorted([y1, y2])

        dx = max_x - min_x
        dy = max_y - min_y

        # padding berechnen (relativ + minimal)
        pad_x = max(dx * padding, min_abs_padding)
        pad_y = max(dy * padding, min_abs_padding)

        min_x -= pad_x
        max_x += pad_x
        min_y -= pad_y
        max_y += pad_y

        min_x = max(min_x, src.bounds.left)
        max_x = min(max_x, src.bounds.right)
        min_y = max(min_y, src.bounds.bottom)
        max_y = min(max_y, src.bounds.top)

        # Pixel-Indices bestimmen
        row_start, col_start = src.index(min_x, max_y)
        row_stop, col_stop = src.index(max_x, min_y)

        if row_start > row_stop:
            row_start, row_stop = row_stop, row_start
        if col_start > col_stop:
            col_start, col_stop = col_stop, col_start

        landcover_array = src.read(1)
        array_height, array_width = landcover_array.shape

        row_start = max(0, min(row_start, array_height - 1))
        row_stop = max(0, min(row_stop, array_height - 1))
        col_start = max(0, min(col_start, array_width - 1))
        col_stop = max(0, min(col_stop, array_width - 1))

        roi_rows = row_stop - row_start
        roi_cols = col_stop - col_start
        if roi_rows <= 0 or roi_cols <= 0:
            raise ValueError("BBox hat keine Überlappung nach Clamping.")

        step_y = roi_rows / (res_y - 1) if res_y > 1 else 0
        step_x = roi_cols / (res_x - 1) if res_x > 1 else 0

        # TerrainMap allokieren
        terrain_ptr = dll.terrain_map_new(res_x, res_y)

        for y_idx in range(res_y):
            r = row_start + int(y_idx * step_y)
            r = max(row_start, min(r, row_stop))
            for x_idx in range(res_x):
                c = col_start + int(x_idx * step_x)
                c = max(col_start, min(c, col_stop))
                pixel_value = int(landcover_array[r, c])
                dll.terrain_set(terrain_ptr, x_idx, y_idx, pixel_value)

        # Zellgröße
        cell_size_x = (max_x - min_x) / res_x
        cell_size_y = (max_y - min_y) / res_y
        cell_size = 0.5 * (cell_size_x + cell_size_y)

        # --- Mapping der Inputpunkte auf Grid ---
        def map_point_to_grid(x, y):
            gx = int((x - min_x) / (max_x - min_x) * (res_x - 1))
            gy = int((max_y - y) / (max_y - min_y) * (res_y - 1))
            gx = max(0, min(res_x - 1, gx))
            gy = max(0, min(res_y - 1, gy))
            return gx, gy

        gx1, gy1 = map_point_to_grid(x1, y1)
        gx2, gy2 = map_point_to_grid(x2, y2)

        return terrain_ptr, (min_x, min_y, max_x, max_y), (gx1, gy1), (gx2, gy2)
