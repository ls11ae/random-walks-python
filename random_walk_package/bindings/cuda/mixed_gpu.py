import ctypes
from ctypes import *

from random_walk_package import KernelsMap3DPtr, TerrainMapPtr, KernelParametersMappingPtr, Point2DArrayPtr
from random_walk_package.wrapper import dll


class Int2(Structure):
    _fields_ = [("x", c_int), ("y", c_int)]


class KernelPoolC(ctypes.Structure):
    _fields_ = [
        ("kernel_pool", POINTER(c_float)),
        ("kernel_pool_size", c_int),
        ("kernel_offsets", POINTER(c_int)),
        ("kernel_offsets_size", c_int),
        ("kernel_widths", POINTER(c_int)),
        ("kernel_widths_size", c_int),
        ("kernel_Ds", POINTER(c_int)),
        ("kernel_Ds_size", c_int),
        ("kernel_index_by_cell", POINTER(c_int)),
        ("kernel_index_by_cell_size", c_int),
        ("offsets_pool", POINTER(Int2)),  # musst du auch als ctypes.Structure definieren
        ("offsets_pool_size", c_int),
        ("offsets_index_per_kernel_dir", POINTER(c_int)),
        ("offsets_index_size", c_int),
        ("offsets_size_per_kernel_dir", POINTER(c_int)),
        ("offsets_size_size", c_int),
        ("max_D", c_int),
        ("max_kernel_width", c_int),
    ]


KernelPoolCPointer = POINTER(KernelPoolC)

dll.build_kernel_pool_c.argtypes = [KernelsMap3DPtr, TerrainMapPtr]
dll.build_kernel_pool_c.restype = KernelPoolCPointer

dll.kernelpoolc_free.argtypes = [KernelPoolCPointer]
dll.kernelpoolc_free.restype = None

dll.gpu_mixed_walk.argtypes = [c_int,  # T
                               c_int,  # W
                               c_int,  # H
                               c_int,  # start_x
                               c_int,  # start_y
                               c_int,  # end_x
                               c_int,  # end_y
                               KernelsMap3DPtr,  # kernels_map
                               KernelParametersMappingPtr,  # mapping
                               TerrainMapPtr,  # terrain
                               c_bool,  # serialize
                               c_char_p,  # serialization_path
                               KernelPoolCPointer]  # KernelPoolC
dll.gpu_mixed_walk.restype = Point2DArrayPtr


def preprocess_mixed_gpu(kernels_map: KernelsMap3DPtr, terrain_map: TerrainMapPtr):
    return dll.build_kernel_pool_c(kernels_map, terrain_map)


def free_kernel_pool(kernel_pool: KernelPoolCPointer):
    dll.kernelpoolc_free(kernel_pool)


def mixed_walk_gpu(time: int, width: int, height: int, start_x: int, start_y: int, end_x: int, end_y: int,
                   kernels_map: KernelsMap3DPtr, mapping: KernelParametersMappingPtr, terrain_map: TerrainMapPtr,
                   serialize: bool, serialization_path: str, kernel_pool: KernelPoolCPointer) -> Point2DArrayPtr:
    return dll.gpu_mixed_walk(time, width, height, start_x, start_y, end_x, end_y, kernels_map, mapping, terrain_map,
                              serialize, serialization_path.encode('ascii'), kernel_pool)
