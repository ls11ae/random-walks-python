from ctypes import *


class Matrix(Structure):
    _fields_ = [
        ("width", c_ssize_t),
        ("height", c_ssize_t),
        ("len", c_ssize_t),
        ("data", POINTER(c_double))
    ]


MatrixPtr = POINTER(Matrix)


class Tensor(Structure):
    _fields_ = [
        ("len", c_size_t),
        ("data", POINTER(POINTER(Matrix)))
    ]


class Point2D(Structure):
    _fields_ = [("x", c_ssize_t),
                ("y", c_ssize_t)]


class DateTime(Structure):
    _fields_ = [("year", c_int),
                ("month", c_int),
                ("day", c_int),
                ("hour", c_int)]


class TimedLocation(Structure):
    _fields_ = [("time", DateTime),
                ("location", Point2D)]


class Point2DArray(Structure):
    _fields_ = [("points", POINTER(Point2D)),
                ("length", c_size_t)]


Point2DArrayPtr = POINTER(Point2DArray)
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


TensorPtr = POINTER(Tensor)
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


TerrainMapPtr = POINTER(TerrainMap)
KernelsMapPtr = POINTER(KernelsMap)
TensorMapPtr = POINTER(KernelsMap3D)
KernelsMap4DPtr = POINTER(KernelsMap4D)
KernelsMap3DPtr = POINTER(KernelsMap3D)


class Coordinate(Structure):
    _fields_ = [("x", c_double),
                ("y", c_double)]


class Coordinate_array(Structure):
    _fields_ = [("points", POINTER(Coordinate)),
                ("length", c_size_t)]


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
        ("forbidden_landmarks", c_int * 12),  # enum -> c_int
        ("has_forbidden_landmarks", c_bool),  # bool -> c_bool
        ("forbidden_landmarks_count", c_int),  # int -> c_int
        ("parameters", KernelParameters * 12)  # fixed-size Array
    ]


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


KernelParametersPtr = POINTER(KernelParameters)
WeatherTimelinePtr = POINTER(WeatherTimeline)
WeatherGridPtr = POINTER(WeatherGrid)
WeatherEntryPtr = POINTER(WeatherEntry)


class Point2DArrayGrid(Structure):
    _fields_ = [("data", POINTER(POINTER(POINTER(Point2DArray)))),
                ("width", c_size_t),
                ("height", c_size_t),
                ("times", c_size_t)
                ]


Point2DArrayGridPtr = POINTER(Point2DArrayGrid)
