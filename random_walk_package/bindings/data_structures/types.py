import ctypes


class Matrix(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_ssize_t),
        ("height", ctypes.c_ssize_t),
        ("len", ctypes.c_ssize_t),
        ("data", ctypes.POINTER(ctypes.c_double))
    ]


MatrixPtr = ctypes.POINTER(Matrix)


class Tensor(ctypes.Structure):
    _fields_ = [
        ("len", ctypes.c_size_t),
        ("data", ctypes.POINTER(ctypes.POINTER(Matrix)))
    ]

class Point2D(ctypes.Structure):
    _fields_ = [("x", ctypes.c_ssize_t),
                ("y", ctypes.c_ssize_t)]


class Point2DArray(ctypes.Structure):
    _fields_ = [("points", ctypes.POINTER(Point2D)),
                ("length", ctypes.c_size_t)]

Point2DArrayPtr = ctypes.POINTER(Point2DArray)
Point2DPtr = ctypes.POINTER(Point2D)

class Vector2D(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.POINTER(Point2D))),
        ("sizes", ctypes.POINTER(ctypes.c_size_t)),
        ("count", ctypes.c_size_t)
    ]


class TensorSet(ctypes.Structure):
    _fields_ = [
        ("len", ctypes.c_size_t),
        ("max_D", ctypes.c_size_t),
        ("data", ctypes.POINTER(ctypes.POINTER(Tensor))),
        ("grid_cells", ctypes.POINTER(ctypes.POINTER(Vector2D)))
    ]


TensorPtr = ctypes.POINTER(Tensor)
TensorSetPtr = ctypes.POINTER(TensorSet)

class TerrainMap(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.POINTER(ctypes.c_int))),
        ("width", ctypes.c_ssize_t),
        ("height", ctypes.c_ssize_t)
    ]


class CacheEntry(ctypes.Structure):
    pass  # forward declaration


class DataUnion(ctypes.Union):
    _fields_ = [
        ("array", TensorPtr),
        ("single", MatrixPtr)
    ]


CacheEntry._fields_ = [
    ("hash", ctypes.c_uint64),
    ("data", DataUnion),
    ("is_array", ctypes.c_bool),
    ("array_size", ctypes.c_ssize_t),
    ("next", ctypes.POINTER(CacheEntry))
]


class Cache(ctypes.Structure):
    _fields_ = [
        ("buckets", ctypes.POINTER(ctypes.POINTER(CacheEntry))),
        ("num_buckets", ctypes.c_size_t)
    ]


class KernelsMap(ctypes.Structure):
    _fields_ = [
        ("kernels", ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(Matrix)))),
        ("width", ctypes.c_ssize_t),
        ("height", ctypes.c_ssize_t),
        ("cache", ctypes.POINTER(Cache))
    ]


class KernelsMap3D(ctypes.Structure):
    _fields_ = [
        ("kernels", ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(Tensor)))),
        ("width", ctypes.c_ssize_t),
        ("height", ctypes.c_ssize_t),
        ("cache", ctypes.POINTER(Cache))
    ]


class KernelsMap4D(ctypes.Structure):
    _fields_ = [
        ("kernels", ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(Tensor))))),
        ("width", ctypes.c_ssize_t),
        ("height", ctypes.c_ssize_t),
        ("timesteps", ctypes.c_ssize_t),
        ("max_D", ctypes.c_ssize_t),
        ("cache", ctypes.POINTER(Cache))
    ]


TerrainMapPtr = ctypes.POINTER(TerrainMap)
KernelsMapPtr = ctypes.POINTER(KernelsMap)
TensorMapPtr = ctypes.POINTER(KernelsMap3D)
KernelsMap4DPtr = ctypes.POINTER(KernelsMap4D)



class Coordinate(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double),
                ("y", ctypes.c_double)]


class Coordinate_array(ctypes.Structure):
    _fields_ = [("points", ctypes.POINTER(Coordinate)),
                ("length", ctypes.c_size_t)]


CoordArray = ctypes.POINTER(Coordinate_array)

class KernelParameters(ctypes.Structure):
    _fields_ = [("is_brownian", ctypes.c_bool),
                ("S", ctypes.c_ssize_t),
                ("D", ctypes.c_ssize_t),
                ("diffusity", ctypes.c_float),
                ("bias_x", ctypes.c_ssize_t),
                ("bias_y", ctypes.c_ssize_t)]


class WeatherEntry(ctypes.Structure):
    _fields_ = [
        ("temperature", ctypes.c_float),
        ("humidity", ctypes.c_int),
        ("precipitation", ctypes.c_float),
        ("wind_speed", ctypes.c_float),
        ("wind_direction", ctypes.c_float),
        ("snow_fall", ctypes.c_float),
        ("weather_code", ctypes.c_int),
        ("cloud_cover", ctypes.c_int)
    ]


class WeatherTimeline(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.POINTER(WeatherEntry))),
        ("length", ctypes.c_size_t)
    ]


class WeatherGrid(ctypes.Structure):
    _fields_ = [
        ("height", ctypes.c_size_t),
        ("width", ctypes.c_size_t),
        ("entries", ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(WeatherTimeline)))),
    ]


KernelParametersPtr = ctypes.POINTER(KernelParameters)
WeatherTimelinePtr = ctypes.POINTER(WeatherTimeline)
WeatherGridPtr = ctypes.POINTER(WeatherGrid)
WeatherEntryPtr = ctypes.POINTER(WeatherEntry)

class Point2DArrayGrid(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(Point2DArray)))),
                ("width", ctypes.c_size_t),
                ("height", ctypes.c_size_t),
                ("times", ctypes.c_size_t)
                ]


Point2DArrayGridPtr = ctypes.POINTER(Point2DArrayGrid)

