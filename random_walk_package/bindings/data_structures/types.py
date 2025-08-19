from _ctypes import _Pointer
from ctypes import *

class Matrix(Structure):
    _fields_ = [
        ("width", c_ssize_t),
        ("height", c_ssize_t),
        ("len", c_ssize_t),
        ("data", POINTER(c_double))
    ]

MatrixPtr: type[_Pointer[Matrix]]

MatrixPtr = POINTER(Matrix)


class Tensor(Structure):
    _fields_ = [
        ("len", c_size_t),
        ("data", POINTER(POINTER(Matrix)))
    ]

class Point2D(Structure):
    _fields_ = [("x", c_ssize_t),
                ("y", c_ssize_t)]


class Point2DArray(Structure):
    _fields_ = [("points", POINTER(Point2D)),
                ("length", c_size_t)]

Point2DArrayPtr: type[_Pointer[Point2DArray]]
Point2DArrayPtr = POINTER(Point2DArray)
Point2DPtr: type[_Pointer[Point2D]]
Point2DPtr = POINTER(Point2D)

class Vector2D(Structure):
    _fields_ = [
        ("data", POINTER(POINTER(Point2D))),
        ("sizes", POINTER(c_size_t)),
        ("count", c_size_t)
    ]


class TensorSet(Structure):
    _fields_ = [
        ("len", c_size_t),
        ("max_D", c_size_t),
        ("data", POINTER(POINTER(Tensor))),
        ("grid_cells", POINTER(POINTER(Vector2D)))
    ]


TensorPtr: type[_Pointer[Tensor]]
TensorPtr = POINTER(Tensor)
TensorSetPtr: type[_Pointer[TensorSet]]
TensorSetPtr = POINTER(TensorSet)

class TerrainMap(Structure):
    _fields_ = [
        ("data", POINTER(POINTER(c_int))),
        ("width", c_ssize_t),
        ("height", c_ssize_t)
    ]


class CacheEntry(Structure):
    pass  # forward declaration


class DataUnion(Union):
    _fields_ = [
        ("array", TensorPtr),
        ("single", MatrixPtr)
    ]


CacheEntry._fields_ = [
    ("hash", c_uint64),
    ("data", DataUnion),
    ("is_array", c_bool),
    ("array_size", c_ssize_t),
    ("next", POINTER(CacheEntry))
]


class Cache(Structure):
    _fields_ = [
        ("buckets", POINTER(POINTER(CacheEntry))),
        ("num_buckets", c_size_t)
    ]


class KernelsMap(Structure):
    _fields_ = [
        ("kernels", POINTER(POINTER(POINTER(Matrix)))),
        ("width", c_ssize_t),
        ("height", c_ssize_t),
        ("cache", POINTER(Cache))
    ]


class KernelsMap3D(Structure):
    _fields_ = [
        ("kernels", POINTER(POINTER(POINTER(Tensor)))),
        ("width", c_ssize_t),
        ("height", c_ssize_t),
        ("cache", POINTER(Cache))
    ]


class KernelsMap4D(Structure):
    _fields_ = [
        ("kernels", POINTER(POINTER(POINTER(POINTER(Tensor))))),
        ("width", c_ssize_t),
        ("height", c_ssize_t),
        ("timesteps", c_ssize_t),
        ("max_D", c_ssize_t),
        ("cache", POINTER(Cache))
    ]


TerrainMapPtr: type[_Pointer[TerrainMap]]
TerrainMapPtr = POINTER(TerrainMap)
KernelsMapPtr: type[_Pointer[KernelsMap]]
KernelsMapPtr = POINTER(KernelsMap)
TensorMapPtr: type[_Pointer[KernelsMap3D]]
TensorMapPtr = POINTER(KernelsMap3D)
KernelsMap4DPtr: type[_Pointer[KernelsMap4D]]
KernelsMap4DPtr = POINTER(KernelsMap4D)
KernelsMap3DPtr: type[_Pointer[KernelsMap3D]]
KernelsMap3DPtr = POINTER(KernelsMap3D)

class Coordinate(Structure):
    _fields_ = [("x", c_double),
                ("y", c_double)]


class Coordinate_array(Structure):
    _fields_ = [("points", POINTER(Coordinate)),
                ("length", c_size_t)]

CoordArray: type[_Pointer[Coordinate_array]]

CoordArray = POINTER(Coordinate_array)

class KernelParameters(Structure):
    _fields_ = [("is_brownian", c_bool),
                ("S", c_ssize_t),
                ("D", c_ssize_t),
                ("diffusity", c_float),
                ("bias_x", c_ssize_t),
                ("bias_y", c_ssize_t)]
    
class KernelParametersMapping(Structure):
    _fields_ = [
        ("forbidden_landmarks", c_int * 12),   # enum -> c_int
        ("has_forbidden_landmarks", c_bool),   # bool -> c_bool
        ("forbidden_landmarks_count", c_int),  # int -> c_int
        ("parameters", KernelParameters * 12)  # fixed-size Array
    ]

KernelParametersMappingPtr: type[_Pointer[KernelParametersMapping]]
KernelParametersMappingPtr = POINTER(KernelParametersMapping)


class WeatherEntry(Structure):
    _fields_ = [
        ("temperature", c_float),
        ("humidity", c_int),
        ("precipitation", c_float),
        ("wind_speed", c_float),
        ("wind_direction", c_float),
        ("snow_fall", c_float),
        ("weather_code", c_int),
        ("cloud_cover", c_int)
    ]


class WeatherTimeline(Structure):
    _fields_ = [
        ("data", POINTER(POINTER(WeatherEntry))),
        ("length", c_size_t)
    ]


class WeatherGrid(Structure):
    _fields_ = [
        ("height", c_size_t),
        ("width", c_size_t),
        ("entries", POINTER(POINTER(POINTER(WeatherTimeline)))),
    ]


KernelParametersPtr: type[_Pointer[KernelParameters]]
KernelParametersPtr = POINTER(KernelParameters)
WeatherTimelinePtr: type[_Pointer[WeatherTimeline]]
WeatherTimelinePtr = POINTER(WeatherTimeline)
WeatherGridPtr: type[_Pointer[WeatherGrid]]
WeatherGridPtr = POINTER(WeatherGrid)
WeatherEntryPtr: type[_Pointer[WeatherEntry]]
WeatherEntryPtr = POINTER(WeatherEntry)

class Point2DArrayGrid(Structure):
    _fields_ = [("data", POINTER(POINTER(POINTER(Point2DArray)))),
                ("width", c_size_t),
                ("height", c_size_t),
                ("times", c_size_t)
                ]

Point2DArrayGridPtr: type[_Pointer[Point2DArrayGrid]]

Point2DArrayGridPtr = POINTER(Point2DArrayGrid)

