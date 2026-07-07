import ctypes
import os
from encodings import utf_8

from randomwalks.bindings.data_structures.types import *
from randomwalks.wrapper import dll


class MesaLandcover(enum.IntEnum):
    TREE_COVER = 10
    SHRUBLAND = 20
    GRASSLAND = 30
    CROPLAND = 40
    BUILT_UP = 50
    BARE_OR_SPARSE_VEGETATION = 60
    SNOW_AND_ICE = 70
    PERMANENT_WATER = 80
    HERBACEOUS_WETLAND = 90
    MANGROVES = 95
    MOSS_AND_LICHEN = 100

    @property
    def label(self):
        return MESA_LANDCOVER_LABELS[self.value]

    @property
    def color(self):
        return MESA_LANDCOVER_COLORS[self.value]


MESA_LANDCOVER_LABELS = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

MESA_LANDCOVER_COLORS = {
    10: (0.0, 0.4, 0.0, 0.85),
    20: (0.67, 0.55, 0.26, 0.85),
    30: (0.0, 0.78, 0.0, 0.75),
    40: (0.78, 0.82, 0.24, 0.78),
    50: (0.72, 0.22, 0.16, 0.45),
    60: (0.82, 0.71, 0.55, 0.85),
    70: (0.9, 0.95, 1.0, 0.9),
    80: (0.0, 0.24, 0.8, 0.78),
    90: (0.45, 0.62, 0.52, 0.9),
    95: (0.0, 0.5, 0.5, 0.85),
    100: (0.33, 0.42, 0.18, 0.85),
}


def terrain_neighborhood_matrix(
        tiff_path,
        x,
        y,
        radius_m=80,
        *,
        coordinate_space="world",
        band=1,
        pixel_size_m=None,
        fill_value=None,
        as_labels=False,
):
    """
    Read a square terrain neighborhood from a GeoTIFF.

    Parameters
    ----------
    tiff_path
        Path to the landcover GeoTIFF.
    x, y
        Point at the center of the neighborhood. Interpreted as raster CRS
        coordinates by default, or as pixel column/row when
        coordinate_space="pixel".
    radius_m
        Neighborhood radius in meters. The returned matrix covers this radius
        in each cardinal direction, so radius_m=80 returns the point plus its
        80 m square neighborhood.
    coordinate_space
        "world" for GeoTIFF CRS coordinates, or "pixel" for column/row indices.
    band
        Raster band to read.
    pixel_size_m
        Meter size of one pixel. Overrides the GeoTIFF transform-derived pixel
        size.
    fill_value
        Value used outside raster bounds. Defaults to raster nodata, or 0 when
        nodata is absent.
    as_labels
        Return known MESA terrain names instead of integer terrain codes.
    """
    import numpy as np
    import rasterio

    if radius_m < 0:
        raise ValueError("radius_m must be non-negative")

    with rasterio.open(tiff_path) as src:
        _, _, window = _terrain_neighborhood_window(
            src,
            x,
            y,
            radius_m,
            coordinate_space=coordinate_space,
            pixel_size_m=pixel_size_m,
        )
        if fill_value is None:
            fill_value = src.nodata if src.nodata is not None else 0

        matrix = src.read(
            band,
            window=window,
            boundless=True,
            fill_value=fill_value,
        )

    if as_labels:
        label_for = np.vectorize(
            lambda value: MESA_LANDCOVER_LABELS.get(int(value), str(int(value)))
        )
        return label_for(matrix)

    return matrix


def plot_terrain_neighborhood(
        tiff_path,
        x,
        y,
        radius_m=80,
        *,
        trajectory_points=None,
        coordinate_space="world",
        band=1,
        pixel_size_m=None,
        fill_value=None,
        ax=None,
        show=True,
):
    """
    Plot a GeoTIFF terrain neighborhood with the focal point and nearby trajectory points.

    trajectory_points accepts an iterable of ``(x, y)``/``(col, row)`` pairs,
    a DataFrame/GeoDataFrame, a MovingPandas Trajectory, or a
    MovingPandas TrajectoryCollection. Points are interpreted in the same
    coordinate_space as the focal point.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio

    if radius_m < 0:
        raise ValueError("radius_m must be non-negative")

    with rasterio.open(tiff_path) as src:
        focal_row, focal_col, window = _terrain_neighborhood_window(
            src,
            x,
            y,
            radius_m,
            coordinate_space=coordinate_space,
            pixel_size_m=pixel_size_m,
        )
        if fill_value is None:
            fill_value = src.nodata if src.nodata is not None else 0
        matrix = src.read(
            band,
            window=window,
            boundless=True,
            fill_value=fill_value,
        )
        selected_points = _trajectory_points_in_window(
            trajectory_points,
            src,
            window,
            coordinate_space,
        )

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(matrix, interpolation="nearest")
    focal_x = focal_col - window.col_off
    focal_y = focal_row - window.row_off
    ax.scatter(
        [focal_x],
        [focal_y],
        marker="x",
        s=90,
        c="red",
        linewidths=2,
        label="Focal point",
    )

    if selected_points:
        points = np.asarray(selected_points, dtype=float)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            marker="o",
            s=28,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            label="Trajectory points in window",
        )

    ax.set_title(f"Terrain neighborhood, r={radius_m} m")
    ax.set_xlabel("Neighborhood column")
    ax.set_ylabel("Neighborhood row")
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    ax.legend(loc="best")
    ax.terrain_neighborhood_points = selected_points

    if show:
        plt.show()

    return ax


def _terrain_neighborhood_window(src, x, y, radius_m, *, coordinate_space, pixel_size_m=None):
    import math

    from rasterio.windows import Window

    if coordinate_space == "world":
        row, col = src.index(x, y)
    elif coordinate_space == "pixel":
        col = int(round(x))
        row = int(round(y))
    else:
        raise ValueError('coordinate_space must be "world" or "pixel"')

    pixel_width_m, pixel_height_m = _terrain_pixel_size_m(src, row, col, pixel_size_m)
    radius_cols = int(math.ceil(radius_m / pixel_width_m))
    radius_rows = int(math.ceil(radius_m / pixel_height_m))
    return row, col, Window(
        col_off=col - radius_cols,
        row_off=row - radius_rows,
        width=radius_cols * 2 + 1,
        height=radius_rows * 2 + 1,
    )


def _trajectory_points_in_window(trajectory_points, src, window, coordinate_space):
    if trajectory_points is None:
        return []

    selected = []
    min_col = window.col_off
    max_col = window.col_off + window.width
    min_row = window.row_off
    max_row = window.row_off + window.height

    for point_x, point_y in _iter_trajectory_coordinates(trajectory_points, src, coordinate_space):
        if coordinate_space == "world":
            row, col = src.index(point_x, point_y)
        else:
            col = int(round(point_x))
            row = int(round(point_y))

        if min_col <= col < max_col and min_row <= row < max_row:
            selected.append((col - window.col_off, row - window.row_off))

    return selected


def _iter_trajectory_coordinates(trajectory_points, src, coordinate_space):
    if hasattr(trajectory_points, "to_point_gdf"):
        trajectory_points = trajectory_points.to_point_gdf()
    elif hasattr(trajectory_points, "df"):
        trajectory_points = trajectory_points.df

    if (
            coordinate_space == "pixel"
            and hasattr(trajectory_points, "columns")
            and {"grid_x", "grid_y"}.issubset(trajectory_points.columns)
    ):
        for point_x, point_y in zip(trajectory_points["grid_x"], trajectory_points["grid_y"]):
            yield point_x, point_y
        return

    if hasattr(trajectory_points, "geometry"):
        frame = trajectory_points
        if coordinate_space == "world" and getattr(frame, "crs", None) is not None and src.crs is not None:
            frame = frame.to_crs(src.crs)
        for geometry in frame.geometry:
            if geometry is not None and not geometry.is_empty:
                yield geometry.x, geometry.y
        return

    if hasattr(trajectory_points, "columns"):
        frame = trajectory_points
        if {"x", "y"}.issubset(frame.columns):
            x_col, y_col = "x", "y"
        elif {"longitude", "latitude"}.issubset(frame.columns):
            x_col, y_col = "longitude", "latitude"
        elif {"location-long", "location-lat"}.issubset(frame.columns):
            x_col, y_col = "location-long", "location-lat"
        else:
            raise ValueError("trajectory_points must contain geometry, grid_x/grid_y, x/y, or lon/lat columns")
        for point_x, point_y in zip(frame[x_col], frame[y_col]):
            yield point_x, point_y
        return

    for point in trajectory_points:
        yield point[0], point[1]


def _terrain_pixel_size_m(src, row, col, pixel_size_m=None):
    if src.transform.b or src.transform.d:
        raise ValueError("Rotated GeoTIFF transforms are not supported")

    if pixel_size_m is not None:
        if isinstance(pixel_size_m, (tuple, list)):
            pixel_width_m, pixel_height_m = pixel_size_m
        else:
            pixel_width_m = pixel_height_m = pixel_size_m
        return _positive_pixel_size(pixel_width_m, pixel_height_m)

    if src.crs is not None and src.crs.is_projected:
        try:
            unit_factor = src.crs.linear_units_factor
        except Exception:
            unit_factor = None
        unit_to_m = 1.0 if unit_factor is None else unit_factor[1]
        return _positive_pixel_size(src.transform.a * unit_to_m, src.transform.e * unit_to_m)

    if src.crs is not None and src.crs.is_geographic:
        return _geographic_pixel_size_m(src, row, col)

    raise ValueError(
        "radius_m requires a GeoTIFF with CRS metadata, or pass pixel_size_m."
    )


def _geographic_pixel_size_m(src, row, col):
    from pyproj import Geod

    lon, lat = src.xy(row, col)
    pixel_width_deg = abs(float(src.transform.a))
    pixel_height_deg = abs(float(src.transform.e))
    geod = Geod(ellps="WGS84")
    _, _, pixel_width_m = geod.inv(lon, lat, lon + pixel_width_deg, lat)
    _, _, pixel_height_m = geod.inv(lon, lat, lon, lat + pixel_height_deg)
    return _positive_pixel_size(pixel_width_m, pixel_height_m)


def _positive_pixel_size(pixel_width_m, pixel_height_m):
    pixel_width_m = abs(float(pixel_width_m))
    pixel_height_m = abs(float(pixel_height_m))
    if pixel_width_m <= 0 or pixel_height_m <= 0:
        raise ValueError("pixel size must be positive")
    return pixel_width_m, pixel_height_m


class MovementPolicyCfg(str, enum.Enum):
    TIME_STEP = "TIME_STEP"
    FIXED_STEPS = "FIXED_STEPS"
    AUTO_SPEED = "AUTO_SPEED"


class BarrierMode(enum.IntEnum):
    FORBID = 0
    AVOID = 1
    ALLOW = 2


class Animal(enum.IntEnum):
    TERRESTRIAL = 0
    MARINE = 1
    AIRBORNE = 2


class TerrainMapHandle:
    dll.terrain_map_new.argtypes = [ctypes.c_ssize_t, ctypes.c_ssize_t]
    dll.terrain_map_new.restype = TerrainMapPtr
    new = dll.terrain_map_new

    dll.terrain_map_free.argtypes = [TerrainMapPtr]
    dll.terrain_map_free.restype = None
    free_ptr = dll.terrain_map_free

    dll.create_terrain_map.argtypes = [ctypes.c_char_p, ctypes.c_char]
    dll.create_terrain_map.restype = TerrainMapPtr
    from_file_ptr = dll.create_terrain_map

    dll.terrain_single_value.argtypes = [ctypes.c_int, ctypes.c_ssize_t, ctypes.c_ssize_t]
    dll.terrain_single_value.restype = TerrainMapPtr
    single_value_ptr = dll.terrain_single_value

    dll.parse_terrain_map.argtypes = [ctypes.c_char_p, TerrainMapPtr, ctypes.c_char]
    dll.parse_terrain_map.restype = ctypes.c_int
    parse_into_ptr = dll.parse_terrain_map

    dll.terrain_at.argtypes = [ctypes.c_ssize_t, ctypes.c_ssize_t, TerrainMapPtr]
    dll.terrain_at.restype = ctypes.c_int
    at_ptr = dll.terrain_at

    dll.terrain_set.argtypes = [TerrainMapPtr, ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_int]
    dll.terrain_set.restype = None
    set_ptr = dll.terrain_set

    dll.tensor_map_terrain.argtypes = [TerrainMapPtr, KernelParametersMappingPtr, ctypes.c_int]
    dll.tensor_map_terrain.restype = KernelsMap3DPtr
    tensor_map_ptr = dll.tensor_map_terrain

    dll.serialize_terrain.argtypes = [ctypes.c_char_p, TerrainMapPtr]
    dll.serialize_terrain.restype = ctypes.c_uint64
    serialize_ptr = dll.serialize_terrain

    dll.deserialize_terrain.argtypes = [ctypes.c_char_p]
    dll.deserialize_terrain.restype = TerrainMapPtr
    deserialize_ptr = dll.deserialize_terrain

    dll.tensor_map_terrain_serialize.argtypes = [
        TerrainMapPtr,
        KernelParametersMappingPtr,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    dll.tensor_map_terrain_serialize.restype = None
    tensor_map_serialize_ptr = dll.tensor_map_terrain_serialize

    dll.kernels_map_single.argtypes = [TerrainMapPtr, TensorPtr, KernelParametersMappingPtr, ctypes.c_int]
    dll.kernels_map_single.restype = KernelsMap3DPtr
    kernels_map_single_ptr = dll.kernels_map_single

    dll.tensor_at.argtypes = [ctypes.c_char_p, ctypes.c_ssize_t, ctypes.c_ssize_t]
    dll.tensor_at.restype = TensorPtr
    tensor_at_ptr = dll.tensor_at

    dll.load_meta_info.argtypes = [ctypes.c_char_p]
    dll.load_meta_info.restype = KernelMapMeta
    load_meta_info_ptr = dll.load_meta_info

    dll.landmarks_count.argtypes = [TerrainMapPtr]
    dll.landmarks_count.restype = ctypes.c_int
    landmarks_count_ptr = dll.landmarks_count

    def __init__(self, width=None, height=None, *, ptr=None):
        if ptr is None:
            if width is None or height is None:
                raise ValueError("width and height are required when ptr is not provided")
            ptr = self.new(width, height)
        if not ptr:
            raise RuntimeError("Failed to allocate TerrainMap")
        self._ptr = ptr

    @classmethod
    def from_ptr(cls, ptr):
        return cls(ptr=ptr)

    @classmethod
    def from_file(cls, file, delim=" "):
        if not os.path.exists(file):
            raise FileNotFoundError(f"File '{file}' does not exist.")
        c_file = file.encode("ascii")
        return cls(ptr=TerrainMapHandle.from_file_ptr(c_file, _c_delim(delim)))

    def to_file(self, file):
        c_file = file.encode("ascii")
        TerrainMapHandle.serialize_ptr(c_file, self.ptr)

    def landmarks_count(self):
        return self.landmarks_count_ptr(self._ptr)

    @classmethod
    def single_value(cls, land_type, width, height):
        ptr = cls.single_value_ptr(land_type, width, height)
        return cls.from_ptr(ptr)

    @property
    def ptr(self):
        return self._ptr

    @property
    def contents(self):
        return self._ptr.contents

    @property
    def width(self):
        return self.contents.width

    @property
    def height(self):
        return self.contents.height

    def at(self, x, y):
        return int(self.at_ptr(x, y, self._ptr))

    def set(self, x, y, value):
        self.set_ptr(self._ptr, x, y, value)

    def to_numpy(self):
        import numpy as np

        return np.array([
            [self.at(x, y) for x in range(self.width)]
            for y in range(self.height)
        ], dtype=int)

    def unique_values(self):
        import numpy as np

        return set(int(value) for value in np.unique(self.to_numpy()))

    def tensor_map(self, mapping, reachability=Reachability.RELAXED):
        ptr = self.tensor_map_ptr(
            self._ptr,
            _mapping_ptr(mapping),
            _reachability_value(reachability),
        )
        return KernelsMap3DHandle.from_ptr(ptr)

    def serialize_tensor_map(self, mapping, output_path, reachability=Reachability.RELAXED):
        self.tensor_map_serialize_ptr(
            self._ptr,
            _mapping_ptr(mapping),
            output_path.encode("utf-8"),
            _reachability_value(reachability),
        )

    @classmethod
    def landcover_to_discrete(cls, file_path, res_x, res_y, min_lon, min_lat, max_lon, max_lat):
        ptr = _landcover_to_discrete_ptr(file_path, res_x, res_y, min_lon, min_lat, max_lon, max_lat)
        if not ptr:
            raise RuntimeError("Failed to convert landcover raster to discrete terrain")
        return cls.from_ptr(ptr)

    def free(self):
        if self._ptr:
            self.free_ptr(self._ptr)
        self._ptr = None

    def __bool__(self):
        return bool(self._ptr)

    def __del__(self):
        self.free()


class KernelsMap3DHandle:
    dll.kernels_map3d_free.argtypes = [KernelsMap3DPtr]
    dll.kernels_map3d_free.restype = None
    free_ptr = dll.kernels_map3d_free

    dll.generate_dir_kernels.argtypes = [KernelParametersMappingPtr]
    dll.generate_dir_kernels.restype = DirKernelsMapPtr
    generate_dir_kernels = dll.generate_dir_kernels

    dll.get_dir_kernels.argtypes = [ctypes.c_ssize_t, ctypes.c_ssize_t]
    dll.get_dir_kernels.restype = DirKernelsMapPtr
    get_dir_kernels = dll.get_dir_kernels

    dll.dir_kernels_free.argtypes = [DirKernelsMapPtr]
    dll.dir_kernels_free.restype = None
    dir_kernels_free = dll.dir_kernels_free

    def __init__(self, *, ptr=None):
        if not ptr:
            raise ValueError("ptr is required")
        self._ptr = ptr

    @classmethod
    def from_ptr(cls, ptr):
        return cls(ptr=ptr)

    @property
    def ptr(self):
        return self._ptr

    @property
    def contents(self):
        return self._ptr.contents

    def free(self):
        if self._ptr:
            self.free_ptr(self._ptr)
        self._ptr = None

    def __bool__(self):
        return bool(self._ptr)

    def __del__(self):
        self.free()


def _terrain_ptr(terrain):
    if hasattr(terrain, "ptr"):
        return terrain.ptr
    if isinstance(terrain, TerrainMap):
        return ctypes.pointer(terrain)
    return terrain


def _mapping_ptr(mapping):
    return mapping.ptr if hasattr(mapping, "ptr") else mapping


def _tensor_ptr(tensor):
    return tensor.ptr if hasattr(tensor, "ptr") else tensor


def _reachability_value(reachability):
    return int(reachability.value) if hasattr(reachability, "value") else int(reachability)


def _resource_path(file):
    if os.path.isabs(file) or os.path.exists(file):
        return file
    return file


def _c_delim(delim):
    if len(delim) != 1:
        raise ValueError("Delimiter must be a single character.")
    return c_char(delim.encode("ascii")[0])


def _landcover_to_discrete_ptr(file_path, res_x, res_y, min_lon, min_lat, max_lon, max_lat):  # type: ignore
    try:
        import rasterio
        from pyproj import Transformer

        def lonlat_bbox_to_utm(min_lon_, min_lat_, max_lon_, max_lat_, epsg_code):
            transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
            min_x, min_y = transformer.transform(min_lon_, min_lat_)
            max_x, max_y = transformer.transform(max_lon_, max_lat_)
            return min_x, min_y, max_x, max_y

        bbox_lonlat = (min_lon, min_lat, max_lon, max_lat)

        with rasterio.open(file_path) as src:
            crs_epsg = src.crs.to_epsg()
            if crs_epsg is None:
                raise ValueError("Raster CRS has no valid EPSG code")

            min_x, min_y, max_x, max_y = lonlat_bbox_to_utm(*bbox_lonlat, crs_epsg)

            landcover_array = src.read(1)
            array_height, array_width = landcover_array.shape

            row_start, col_start = src.index(min_x, max_y)
            row_stop, col_stop = src.index(max_x, min_y)

            if row_start > row_stop:
                row_start, row_stop = row_stop, row_start
            if col_start > col_stop:
                col_start, col_stop = col_stop, col_start

            row_start = max(0, min(row_start, array_height - 1))
            row_stop = max(0, min(row_stop, array_height - 1))
            col_start = max(0, min(col_start, array_width - 1))
            col_stop = max(0, min(col_stop, array_width - 1))

            roi_rows = row_stop - row_start
            roi_cols = col_stop - col_start

            if roi_rows < 0 or roi_cols < 0 or (row_start == row_stop and col_start == col_stop):
                raise ValueError(
                    "Requested bounding box does not overlap the landcover raster. "
                    f"Raster bounds (lon, lat): {src.bounds}. "
                    f"Requested bbox: ({min_lon}, {min_lat}, {max_lon}, {max_lat})."
                )

            step_y = roi_rows / (res_y - 1) if res_y > 1 else 0
            step_x = roi_cols / (res_x - 1) if res_x > 1 else 0
            terrain_ptr = TerrainMapHandle.new(res_x, res_y)

            for y_idx in range(res_y):
                r = row_start + int(y_idx * step_y)
                r = max(row_start, min(r, row_stop))
                r = min(r, array_height - 1)

                for x_idx in range(res_x):
                    c = col_start + int(x_idx * step_x)
                    c = max(col_start, min(c, col_stop))
                    c = min(c, array_width - 1)

                    TerrainMapHandle.set_ptr(terrain_ptr, x_idx, y_idx, int(landcover_array[r, c]))

            return terrain_ptr
    except ImportError as e:
        raise ImportError("TerrainMapHandle.landcover_to_discrete requires rasterio and pyproj") from e
    except Exception as e:
        print(f"Error opening or converting the file: {e}")
        return None
