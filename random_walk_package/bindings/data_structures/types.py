import enum
from ctypes import *


class Pair(Structure):
    _fields_ = [("first", c_double),
                ("second", c_double)]


class MatrixData(Union):
    _fields_ = [
        ("points", POINTER(c_double)),
        ("pairs", POINTER(Pair)), ]


class Matrix(Structure):
    _fields_ = [
        ("width", c_ssize_t),
        ("height", c_ssize_t),
        ("len", c_ssize_t),
        ("data", MatrixData),
    ]


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
        ("max_M", c_size_t),
        ("data", POINTER(POINTER(Tensor))),
        ("grid_cells", POINTER(POINTER(Vector2D)))
    ]


class TerrainMap(Structure):
    _fields_ = [
        ("data", POINTER(POINTER(c_int))),
        ("width", c_ssize_t),
        ("height", c_ssize_t)
    ]


class CacheEntry(Structure):
    pass  # forward declaration


MatrixPtr = POINTER(Matrix)
Point2DArrayPtr = POINTER(Point2DArray)
Point2DPtr = POINTER(Point2D)
TensorPtr = POINTER(Tensor)
TensorSetPtr = POINTER(TensorSet)


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


class DirKernelsMap(Structure):
    _fields_ = [
        ("max_D", c_ssize_t),
        ("max_kernel_size", c_ssize_t),
        ("data", POINTER(POINTER(POINTER(Vector2D))))
    ]


DirKernelsMapPtr = POINTER(DirKernelsMap)


class KernelsMap3D(Structure):
    _fields_ = [
        ("soft_reachability", c_int),
        ("kernels", POINTER(POINTER(POINTER(Tensor)))),
        ("width", c_ssize_t),
        ("height", c_ssize_t),
        ("max_D", c_ssize_t),
        ("cache", POINTER(Cache)),
        ("dir_kernels", DirKernelsMapPtr)
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
                ("sigma_length", c_float),
                ("sigma_angle", c_float),
                ("bias_x", c_ssize_t),
                ("bias_y", c_ssize_t)]


class KernelParametersTerrainWeather(Structure):
    _fields_ = [
        ("width", c_size_t),
        ("height", c_size_t),
        ("time", c_size_t),
        ("max_D", c_size_t),
        ("data", POINTER(POINTER(POINTER(POINTER(KernelParameters)))))
    ]


KernelParametersTerrainWeatherPtr = POINTER(KernelParametersTerrainWeather)

LAND_MARKS_COUNT = 16


class KPM_Data(Union):
    _fields_ = [("parameters", KernelParameters * LAND_MARKS_COUNT),
                ("kernels", TensorPtr)]


TerrainMapPtr = POINTER(TerrainMap)
TensorMapPtr = POINTER(KernelsMap3D)
KernelsMap4DPtr = POINTER(KernelsMap4D)
KernelsMap3DPtr = POINTER(KernelsMap3D)


class KernelParametersMapping(Structure):
    _fields_ = [
        ("forbidden_landmarks", c_int * LAND_MARKS_COUNT),  # enum -> c_int
        ("has_forbidden_landmarks", c_bool),  # bool -> c_bool
        ("forbidden_landmarks_count", c_int),  # int -> c_int
        ("parameters", KernelParameters * LAND_MARKS_COUNT),  # fixed-size Array
        ("kind", c_int),
        ("animal", c_int),
        ("data", KPM_Data),
    ]


KernelParametersMappingPtr = POINTER(KernelParametersMapping)
KernelParametersPtr = POINTER(KernelParameters)


class Point2DArrayGrid(Structure):
    _fields_ = [("data", POINTER(POINTER(POINTER(Point2DArray)))),
                ("width", c_size_t),
                ("height", c_size_t),
                ("times", c_size_t)
                ]


Point2DArrayGridPtr = POINTER(Point2DArrayGrid)


class Reachability(enum.IntEnum):
    SOFT = 0
    HARD = 1
    FULL = 2


class KernelContext(Structure):
    _fields_ = [("reachability_mode", c_int),  # enum ReachabilityMode reachability_mode;
                ("mode", c_int),  # enum ComputationMode mode;
                ("mapping", KernelParametersMappingPtr),
                ("terrain", TerrainMapPtr),
                ("kernels_map", KernelsMap3DPtr),
                ("base_kernels", TensorSetPtr),
                ("dir_kernels_map", DirKernelsMapPtr),
                ("dp_dir", c_char_p),
                ("kernel_pool_dir", c_char_p)
                ]


KernelContextPtr = POINTER(KernelContext)

PATH_MAX = 4096
HASH_CACHE_BUCKETS = 4096


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BiaKind(enum.IntEnum):
    OFFSET = 0
    ROTATION = 1


# ---------------------------------------------------------------------------
# Biases
# ---------------------------------------------------------------------------

class _BiasData(Union):
    _fields_ = [
        ("offsets", POINTER(Point2D)),
        ("rotation_deg", POINTER(c_double)),
    ]


class Biases(Structure):
    _fields_ = [
        ("kind", c_int),  # enum bias_kind
        ("data", _BiasData),
        ("len", c_size_t),
    ]


# ---------------------------------------------------------------------------
# HashEntry / HashCache
# ---------------------------------------------------------------------------

class HashEntry(Structure):
    pass  # forward declaration


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


# ---------------------------------------------------------------------------
# KernelModifier
# ---------------------------------------------------------------------------

class KernelModifier(Structure):
    _fields_ = [
        ("switch_model", c_bool),
        ("step_size_mod", c_float),
        ("directions_mod", c_float),
        ("diffusity_mod", c_float),
    ]


# ---------------------------------------------------------------------------
# EnvWeightProfile
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TimedKernelParameters
# ---------------------------------------------------------------------------

class TimedKernelParameters(Structure):
    _fields_ = [
        ("date_time", POINTER(DateTime)),
        ("params", POINTER(KernelParameters)),
        ("landmark", c_int),
    ]


# ---------------------------------------------------------------------------
# Dimensions3D / GridDimensions
# ---------------------------------------------------------------------------

class Dimensions3D(Structure):
    _fields_ = [
        ("y", c_ssize_t),
        ("x", c_ssize_t),
        ("t", c_ssize_t),
    ]


class GridDimensions(Structure):
    _fields_ = [
        ("T", c_size_t),
        ("D", c_size_t),
        ("W", c_size_t),
        ("H", c_size_t),
    ]


# ---------------------------------------------------------------------------
# EnvironmentInfluenceGrid
# ---------------------------------------------------------------------------

class EnvironmentInfluenceGrid(Structure):
    _fields_ = [
        # TimedKernelParameters ****  →  4 levels of indirection
        ("params", POINTER(POINTER(POINTER(POINTER(TimedKernelParameters))))),
        ("dims", POINTER(Dimensions3D)),
    ]


# ---------------------------------------------------------------------------
# DateTimeInterval
# ---------------------------------------------------------------------------

class DateTimeInterval(Structure):
    _fields_ = [
        ("start", DateTime),
        ("end", DateTime),
    ]


# ---------------------------------------------------------------------------
# KernelParametersTerrain / KernelParamsYXT
# ---------------------------------------------------------------------------

class KernelParametersTerrain(Structure):
    _fields_ = [
        ("width", c_size_t),
        ("height", c_size_t),
        ("data", POINTER(POINTER(POINTER(KernelParameters)))),  # [y][x]
    ]


class KernelParamsYXT(Structure):
    _fields_ = [
        ("width", c_size_t),
        ("height", c_size_t),
        ("time", c_size_t),
        ("max_D", c_size_t),
        ("max_S", c_size_t),
        ("data", POINTER(POINTER(POINTER(POINTER(KernelParameters))))),  # [y][x][t]
    ]


# ---------------------------------------------------------------------------
# TimedLocationArray
# ---------------------------------------------------------------------------

class TimedLocationArray(Structure):
    _fields_ = [
        ("data", POINTER(TimedLocation)),
        ("length", c_size_t),
    ]


# ---------------------------------------------------------------------------
# Pointer aliases (consistent with existing style)
# ---------------------------------------------------------------------------

BiasesPtr = POINTER(Biases)
HashEntryPtr = POINTER(HashEntry)
HashCachePtr = POINTER(HashCache)
KernelModifierPtr = POINTER(KernelModifier)
EnvWeightProfilePtr = POINTER(EnvWeightProfile)
TimedKernelParametersPtr = POINTER(TimedKernelParameters)
Dimensions3DPtr = POINTER(Dimensions3D)
GridDimensionsPtr = POINTER(GridDimensions)
EnvironmentInfluenceGridPtr = POINTER(EnvironmentInfluenceGrid)
DateTimeIntervalPtr = POINTER(DateTimeInterval)
KernelParametersTerrainPtr = POINTER(KernelParametersTerrain)
KernelParamsYXTPtr = POINTER(KernelParamsYXT)
TimedLocationArrayPtr = POINTER(TimedLocationArray)
