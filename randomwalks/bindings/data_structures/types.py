import enum
from ctypes import *

PATH_MAX = 4096
HASH_CACHE_BUCKETS = 4096


class Pair(Structure):
    _fields_ = [
        ("first", c_double),
        ("second", c_double),
    ]


class Matrix(Structure):
    _fields_ = [
        ("width", c_ssize_t),
        ("height", c_ssize_t),
        ("len", c_ssize_t),
        ("points", POINTER(c_double)),
    ]

    @property
    def data(self):
        return self.points


class Tensor(Structure):
    _fields_ = [
        ("len", c_size_t),
        ("data", POINTER(POINTER(Matrix))),
    ]


class Point2D(Structure):
    _fields_ = [
        ("x", c_ssize_t),
        ("y", c_ssize_t),
    ]


class DateTime(Structure):
    _fields_ = [
        ("year", c_int),
        ("month", c_int),
        ("day", c_int),
        ("hour", c_int),
    ]


class TimedLocation(Structure):
    _fields_ = [
        ("timestamp", DateTime),
        ("coordinates", Point2D),
    ]

    def __init__(self, timestamp=None, coordinates=None, time=None, location=None):
        timestamp = timestamp if timestamp is not None else time
        coordinates = coordinates if coordinates is not None else location
        if timestamp is None and coordinates is None:
            super().__init__()
        else:
            super().__init__(timestamp or DateTime(), coordinates or Point2D())

    @property
    def time(self):
        return self.timestamp

    @time.setter
    def time(self, value):
        self.timestamp = value

    @property
    def location(self):
        return self.coordinates

    @location.setter
    def location(self, value):
        self.coordinates = value


class Point2DArray(Structure):
    _fields_ = [
        ("points", POINTER(Point2D)),
        ("length", c_size_t),
    ]


class DirOffsets(Structure):
    _fields_ = [
        ("offsets", POINTER(POINTER(Point2D))),
        ("sizes", POINTER(c_size_t)),
        ("count", c_size_t),
    ]

    @property
    def data(self):
        return self.offsets

    @data.setter
    def data(self, value):
        self.offsets = value


Vector2D = DirOffsets


class TerrainMap(Structure):
    _fields_ = [
        ("width", c_ssize_t),
        ("height", c_ssize_t),
        ("data", POINTER(POINTER(c_int))),
    ]


MatrixPtr = POINTER(Matrix)
Point2DPtr = POINTER(Point2D)
Point2DArrayPtr = POINTER(Point2DArray)
TensorPtr = POINTER(Tensor)
DirOffsetsPtr = POINTER(DirOffsets)
Vector2DPtr = DirOffsetsPtr
TerrainMapPtr = POINTER(TerrainMap)


class TensorSet(Structure):
    _fields_ = [
        ("len", c_size_t),
        ("max_D", c_size_t),
        ("max_M", c_size_t),
        ("terrain_values", POINTER(c_int)),
        ("data", POINTER(TensorPtr)),
        ("grid_cells", POINTER(DirOffsetsPtr)),
    ]


TensorSetPtr = POINTER(TensorSet)


class DataUnion(Union):
    _fields_ = [
        ("array", TensorPtr),
        ("single", MatrixPtr),
    ]


class CacheEntry(Structure):
    pass


CacheEntry._fields_ = [
    ("hash", c_size_t),
    ("data", DataUnion),
    ("is_array", c_bool),
    ("array_size", c_ssize_t),
    ("next", POINTER(CacheEntry)),
]


class Cache(Structure):
    _fields_ = [
        ("buckets", POINTER(POINTER(CacheEntry))),
        ("num_buckets", c_size_t),
    ]


CachePtr = POINTER(Cache)


class HashEntry(Structure):
    pass


HashEntry._fields_ = [
    ("hash", c_size_t),
    ("tensor", TensorPtr),
    ("path", c_char * PATH_MAX),
    ("next", POINTER(HashEntry)),
]


class HashCache(Structure):
    _fields_ = [
        ("buckets", POINTER(HashEntry) * HASH_CACHE_BUCKETS),
    ]


HashEntryPtr = POINTER(HashEntry)
HashCachePtr = POINTER(HashCache)


class Coordinate(Structure):
    _fields_ = [
        ("x", c_double),
        ("y", c_double),
    ]


class Coordinate_array(Structure):
    _fields_ = [
        ("points", POINTER(Coordinate)),
        ("length", c_size_t),
    ]


CoordinateArray = Coordinate_array
CoordArray = POINTER(Coordinate_array)


class KernelParameters(Structure):
    _fields_ = [
        ("is_brownian", c_bool),
        ("S", c_ssize_t),
        ("D", c_ssize_t),
        ("sigma_length", c_float),
        ("sigma_angle", c_float),
        ("bias_x", c_ssize_t),
        ("bias_y", c_ssize_t),
    ]


KernelParametersPtr = POINTER(KernelParameters)


class EnvWeightProfile(Structure):
    _fields_ = [
        ("override_mode", c_bool),
        ("S", c_float),
        ("D", c_float),
        ("sigma_length", c_float),
        ("sigma_angle", c_float),
        ("bias_x", c_float),
        ("bias_y", c_float),
    ]


EnvWeightProfilePtr = POINTER(EnvWeightProfile)


class TimedKernelParameters(Structure):
    _fields_ = [
        ("date_time", POINTER(DateTime)),
        ("params", KernelParametersPtr),
        ("terrain", c_int),
    ]

    @property
    def landmark(self):
        return self.terrain

    @landmark.setter
    def landmark(self, value):
        self.terrain = value


TimedKernelParametersPtr = POINTER(TimedKernelParameters)


class Dimensions3D(Structure):
    _fields_ = [
        ("y", c_size_t),
        ("x", c_size_t),
        ("t", c_size_t),
    ]


class GridDimensions(Structure):
    _fields_ = [
        ("T", c_size_t),
        ("D", c_size_t),
        ("W", c_size_t),
        ("H", c_size_t),
    ]


Dimensions3DPtr = POINTER(Dimensions3D)
GridDimensionsPtr = POINTER(GridDimensions)


class EnvironmentInfluenceGrid(Structure):
    _fields_ = [
        ("params", POINTER(POINTER(POINTER(POINTER(TimedKernelParameters))))),
        ("dims", Dimensions3DPtr),
    ]


EnvironmentInfluenceGridPtr = POINTER(EnvironmentInfluenceGrid)


class DateTimeInterval(Structure):
    _fields_ = [
        ("start", DateTime),
        ("end", DateTime),
    ]


DateTimeIntervalPtr = POINTER(DateTimeInterval)


class KernelModifier(Structure):
    _fields_ = [
        ("switch_model", c_bool),
        ("step_size_mod", c_float),
        ("directions_mod", c_float),
        ("diffusity_mod", c_float),
    ]


KernelModifierPtr = POINTER(KernelModifier)


class BiasKind(enum.IntEnum):
    OFFSET = 0
    ROTATION = 1


BiaKind = BiasKind


class _BiasData(Union):
    _fields_ = [
        ("offsets", Point2DPtr),
        ("rotation_deg", POINTER(c_double)),
    ]


class Biases(Structure):
    _fields_ = [
        ("kind", c_int),
        ("data", _BiasData),
        ("len", c_size_t),
    ]


BiasesPtr = POINTER(Biases)


class KernelMapKind(enum.IntEnum):
    PARAMETERS = 0
    KERNELS = 1


KPM_KIND_PARAMETERS = KernelMapKind.PARAMETERS
KPM_KIND_KERNELS = KernelMapKind.KERNELS


class KPM_Data(Union):
    _fields_ = [
        ("parameters", KernelParametersPtr),
        ("kernels", POINTER(TensorPtr)),
    ]


class KernelParametersMapping(Structure):
    _fields_ = [
        ("terrain_count", c_size_t),
        ("terrain_values", POINTER(c_int)),
        ("set", POINTER(c_bool)),
        ("barrier", POINTER(c_bool)),
        ("unmapped", POINTER(c_bool)),
        ("has_barrier", c_bool),
        ("transition_weights", POINTER(c_double)),
        ("kind", c_int),
        ("data", KPM_Data),
    ]


KernelParametersMappingPtr = POINTER(KernelParametersMapping)


class DirKernelsMap(Structure):
    _fields_ = [
        ("max_D", c_ssize_t),
        ("max_kernel_size", c_ssize_t),
        ("data", POINTER(POINTER(DirOffsetsPtr))),
    ]


DirKernelsMapPtr = POINTER(DirKernelsMap)


class KernelsMap3D(Structure):
    _fields_ = [
        ("soft_reachability", c_int),
        ("kernels", POINTER(POINTER(POINTER(Tensor)))),
        ("width", c_ssize_t),
        ("height", c_ssize_t),
        ("max_D", c_ssize_t),
        ("cache", CachePtr),
        ("dir_kernels", DirKernelsMapPtr),
    ]

    @property
    def reachability_mode(self):
        return self.soft_reachability


KernelsMap3DPtr = POINTER(KernelsMap3D)
TensorMapPtr = KernelsMap3DPtr


class KernelsMap4D(Structure):
    _fields_ = [
        ("kernels", POINTER(POINTER(POINTER(POINTER(Tensor))))),
        ("width", c_ssize_t),
        ("height", c_ssize_t),
        ("timesteps", c_ssize_t),
        ("max_D", c_ssize_t),
        ("cache", CachePtr),
    ]


KernelsMap4DPtr = POINTER(KernelsMap4D)


class KernelParametersTerrain(Structure):
    _fields_ = [
        ("width", c_size_t),
        ("height", c_size_t),
        ("data", POINTER(POINTER(POINTER(KernelParameters)))),
    ]


KernelParametersTerrainPtr = POINTER(KernelParametersTerrain)


class KernelParamsYXT(Structure):
    _fields_ = [
        ("width", c_size_t),
        ("height", c_size_t),
        ("time", c_size_t),
        ("max_D", c_size_t),
        ("max_S", c_size_t),
        ("data", POINTER(POINTER(POINTER(POINTER(KernelParameters))))),
    ]


KernelParamsYXTPtr = POINTER(KernelParamsYXT)
KernelParametersTerrainWeather = KernelParamsYXT
KernelParametersTerrainWeatherPtr = KernelParamsYXTPtr


class Point2DArrayGrid(Structure):
    _fields_ = [
        ("data", POINTER(POINTER(POINTER(Point2DArray)))),
        ("width", c_size_t),
        ("height", c_size_t),
        ("times", c_size_t),
    ]


Point2DArrayGridPtr = POINTER(Point2DArrayGrid)


class TimedLocationArray(Structure):
    _fields_ = [
        ("length", c_size_t),
        ("data", POINTER(TimedLocation)),
    ]


TimedLocationArrayPtr = POINTER(TimedLocationArray)


class KernelMapMeta(Structure):
    _fields_ = [
        ("width", c_ssize_t),
        ("height", c_ssize_t),
        ("timesteps", c_ssize_t),
        ("max_D", c_size_t),
    ]


KernelMapMetaPtr = POINTER(KernelMapMeta)


class Reachability(enum.IntEnum):
    RELAXED = 0
    HARD = 1
    FULL = 2


class ComputationMode(enum.IntEnum):
    ON_THE_FLY = 0
    KERNEL_POOL = 1
    SERIALIZATION = 2


class KernelContext(Structure):
    _fields_ = [
        ("reachability_mode", c_int),
        ("mode", c_int),
        ("mapping", KernelParametersMappingPtr),
        ("terrain", TerrainMapPtr),
        ("kernels_map", KernelsMap3DPtr),
        ("base_kernels", TensorSetPtr),
        ("dir_kernels_map", DirKernelsMapPtr),
        ("dp_dir", c_char_p),
        ("kernel_pool_dir", c_char_p),
        ("cuda_kernel_pool", c_void_p),
    ]


KernelContextPtr = POINTER(KernelContext)
