from random_walk_package.bindings.data_structures.types import *
from random_walk_package.bindings.data_structures.terrain import *
from random_walk_package.wrapper import dll


class Vector2D(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.POINTER(Point2D))),
        ("sizes", ctypes.POINTER(ctypes.c_size_t)),
        ("count", ctypes.c_size_t),
    ]


dll.generate_chi_kernel.argtypes = [
    ctypes.c_ssize_t,  # size
    ctypes.c_ssize_t,  # subsample_size
    ctypes.c_int,  # k
    ctypes.c_int  # d
]
dll.generate_chi_kernel.restype = ctypes.POINTER(Matrix)

# Backtrace
dll.backtrace.argtypes = [
    ctypes.POINTER(ctypes.POINTER(Tensor)),  # DP_Matrix (Tensor**)
    ctypes.c_ssize_t,  # T
    ctypes.POINTER(Tensor),  # kernel
    TerrainMapPtr,  # terrain
    TensorMapPtr,  # tensormap
    ctypes.c_ssize_t,  # end_x
    ctypes.c_ssize_t,  # end_y
    ctypes.c_ssize_t,  # direction
    ctypes.c_ssize_t  # D
]
dll.backtrace.restype = ctypes.POINTER(Point2DArray)

# Backtrace multiple
dll.c_walk_backtrace_multiple.argtypes = [
    ctypes.c_ssize_t,  # T
    ctypes.c_ssize_t,  # W
    ctypes.c_ssize_t,  # H
    ctypes.POINTER(Tensor),  # DP_Matrix (Tensor**)
    TerrainMapPtr,
    TensorMapPtr,
    Point2DArrayPtr  # steps
]
dll.backtrace.restype = ctypes.POINTER(Point2DArray)

# DP Calculation
dll.dp_calculation.argtypes = [
    ctypes.c_ssize_t,  # matrix_start
    ctypes.c_ssize_t,  # matrix_start
    ctypes.POINTER(Tensor),  # kernel
    ctypes.c_ssize_t,  # T
    ctypes.c_ssize_t,  # start x
    ctypes.c_ssize_t,  # start y
]
dll.dp_calculation.restype = ctypes.POINTER(ctypes.POINTER(Tensor))  # Tensor**

# Generate Kernels
dll.generate_kernels.argtypes = [
    ctypes.c_ssize_t,  # directions
    ctypes.c_ssize_t,  # w
]
dll.generate_kernels.restype = ctypes.POINTER(Tensor)

# Assign Sectors Matrix
dll.assign_sectors_matrix.argtypes = [
    ctypes.c_ssize_t,  # width
    ctypes.c_ssize_t,  # height
    ctypes.c_ssize_t  # D
]
dll.assign_sectors_matrix.restype = ctypes.POINTER(Matrix)

# Assign Sectors Tensor
dll.assign_sectors_tensor.argtypes = [
    ctypes.c_ssize_t,  # width
    ctypes.c_ssize_t,  # height
    ctypes.c_int  # D
]
dll.assign_sectors_tensor.restype = ctypes.POINTER(Tensor)

# DP Calculation
dll.c_walk_init_terrain.argtypes = [
    ctypes.c_ssize_t,  # matrix_width
    ctypes.c_ssize_t,  # matrix_height
    ctypes.POINTER(Tensor),  # kernel
    TerrainMapPtr,
    TensorMapPtr,
    ctypes.c_ssize_t,  # T
    ctypes.c_ssize_t,  # start x
    ctypes.c_ssize_t,  # start y
]
dll.c_walk_init_terrain.restype = ctypes.POINTER(ctypes.POINTER(Tensor))  # dp matrix

# DP Calculation
dll.c_walk_init_terrain_low_ram.argtypes = [
    ctypes.c_ssize_t,  # matrix_width
    ctypes.c_ssize_t,  # matrix_height
    ctypes.POINTER(Tensor),  # kernel
    TerrainMapPtr,
    TensorMapPtr,
    ctypes.c_ssize_t,  # T
    ctypes.c_ssize_t,  # start x
    ctypes.c_ssize_t,  # start y
    ctypes.c_char_p
]
dll.c_walk_init_terrain.restype = None  # dp matrix

dll.free_Vector2D.argtypes = [ctypes.POINTER(Vector2D)]
dll.free_Vector2D.restype = None

dll.dp_calculation_low_ram.argtypes = [
    ctypes.c_ssize_t,  # W
    ctypes.c_ssize_t,  # H
    ctypes.POINTER(Tensor),  # kernel
    ctypes.c_ssize_t,  # T
    ctypes.c_ssize_t,  # start_x
    ctypes.c_ssize_t,  # start_y
    ctypes.c_char_p  # output_folder
]
dll.dp_calculation_low_ram.restype = None

dll.backtrace_low_ram.argtypes = [
    ctypes.c_char_p,  # dp_folder
    ctypes.c_ssize_t,  # T
    ctypes.POINTER(Tensor),  # kernel
    TensorMapPtr,  # tensor_map
    ctypes.c_ssize_t,  # end_x
    ctypes.c_ssize_t,  # end_y
    ctypes.c_ssize_t,  # dir
    ctypes.c_ssize_t  # D
]
dll.backtrace_low_ram.restype = ctypes.POINTER(Point2DArray)


def generate_correlated_kernel(width, D):
    return dll.generate_kernels(D, width)


def backtrace_low_ram(dp_folder, T, kernel, tensor_map, end_x, end_y, direction, directions):
    dp_folder_bytes = dp_folder.encode('utf-8')
    return dll.backtrace_low_ram(dp_folder_bytes, T, kernel, tensor_map, end_x, end_y, direction, directions)


def dp_calculation_low_ram(width, height, kernel, time, start_x, start_y, output_folder):
    output_folder_bytes = output_folder.encode('utf-8')
    dll.dp_calculation_low_ram(width, height, kernel, time, start_x, start_y, output_folder_bytes)


def dp_calculation_terrain_low_ram(W, H, kernel, terrain_map, kernels_map, T, start_x, start_y, output_folder):
    output_folder_bytes = output_folder.encode('utf-8')
    dll.c_walk_init_terrain_low_ram(W, H, kernel, terrain_map, kernels_map, T, start_x, start_y, output_folder_bytes)


def correlated_dp_matrix(kernel, width, height, time, start_x=None, start_y=None):
    if start_x is None or start_y is None:
        start_x = width // 2
        start_y = height // 2
    dp_matrix_tensor = dll.dp_calculation(width, height, kernel, time, start_x, start_y)
    return dp_matrix_tensor


def correlated_dp_matrix_terrain(width, height, kernel, terrain, tensor_map, time, start_x, start_y):
    dp_matrix_tensor = dll.c_walk_init_terrain(width, height, kernel, terrain, tensor_map, time, start_x, start_y)
    return dp_matrix_tensor


def correlated_backtrace(dp_mat, T, kernels, terrain, tensor_map, end_x, end_y, direction, directions):
    return dll.backtrace(dp_mat, T, kernels, terrain, tensor_map, end_x, end_y, direction, directions)


def walk_backtrace_multiple(T, W, H, kernel, tensor_map, steps):
    array_ptr = create_point2d_array(steps)
    multistep_walk = dll.c_walk_backtrace_multiple(T, W, H, kernel, tensor_map, array_ptr)
    dll.point2d_array_free(array_ptr)  # Add this line
    return multistep_walk
