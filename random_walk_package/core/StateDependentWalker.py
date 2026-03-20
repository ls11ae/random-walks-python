import math

from pyproj import Transformer
from skimage.transform import resize

from random_walk_package.core.move_apps_patch import merge_traj_collections, apply_moveapps_id_dtype_patch, \
    debug_patch_state, force_tc_id_object_inplace
import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from random_walk_package import MixedWalker, get_walk_points, dll, tensor_free, tensor4D_free, AnimalMovementProcessor
from random_walk_package.bindings import kernels_map3d_free, Animal, create_mixed_kernel_parameters, \
    landcover_to_discrete_ptr, WaterMode, terrain_map_free
from random_walk_package.bindings.correlated_walk import correlated_walk_init, correlated_backtrace
from random_walk_package.bindings.data_structures.kernel_terrain_mapping import marine_kernels_baseline_crw
from random_walk_package.bindings.data_structures.kernels import normalize_kernel, clip_kernel, \
    correlated_kernels_from_matrix
from random_walk_package.bindings.mixed_walk import single_state_walk, kernels_map_single_kernel
from random_walk_package.core.MovementPolicy import TimeStepPolicy

def direction_from_points(start_x, start_y, end_x, end_y, dirs=8):
    dx = start_x - end_x
    dy = start_y - end_y

    if dx == 0 and dy == 0:
        return 0

    angle_deg = math.degrees(math.atan2(dy, dx))
    angle_west_deg = (angle_deg - 180.0) % 360.0

    step = 360.0 / dirs
    return int(round(angle_west_deg / step)) % dirs


def bilinear_sample(K, x, y):
    h, w = K.shape

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    # outside kernel -> zero probability
    if x0 < 0 or y0 < 0 or x1 >= h or y1 >= w:
        return 0.0

    dx = x - x0
    dy = y - y0

    return (
        K[x0, y0] * (1 - dx) * (1 - dy) +
        K[x1, y0] * dx       * (1 - dy) +
        K[x0, y1] * (1 - dx) * dy +
        K[x1, y1] * dx       * dy
    )

def resample_kernel_to_grid(K_meter, cell_size, S):
    size = 2 * S + 1
    center = K_meter.shape[0] // 2

    K_grid = np.zeros((size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            dx_m = (i - S) * cell_size
            dy_m = (j - S) * cell_size

            x = center + dx_m
            y = center + dy_m

            K_grid[i, j] = bilinear_sample(K_meter, x, y)

    return K_grid / K_grid.sum()


def trajectory_segments(steps, max_cell_size, resolution):
    if len(steps) == 0:
        return []
    segments = []
    max_radius = max_cell_size * resolution / 2.0
    start_idx = 0
    ref_x = steps["geo_x"].iloc[0]
    ref_y = steps["geo_y"].iloc[0]

    for i in range(1, len(steps)):
        x = steps["geo_x"].iloc[i]
        y = steps["geo_y"].iloc[i]
        dist = np.hypot(ref_x - x, ref_y - y)
        if dist >= max_radius:
            segments.append((start_idx, i - 1))
            start_idx = i
            ref_x, ref_y = x, y

    segments.append((start_idx, len(steps) - 1))
    return segments

def merge_singletons(segments):
    if not segments:
        return segments
    merged = []
    i = 0
    while i < len(segments):
        s, e = segments[i]
        if s == e:
            if merged:
                ps, pe = merged[-1]
                merged[-1] = (ps, e)
            elif i + 1 < len(segments):
                ns, ne = segments[i + 1]
                merged.append((s, ne))
                i += 1
            else:
                merged.append((s, e))
        else:
            merged.append((s, e))
        i += 1

    return merged

def make_overlapping(segments):
    if not segments:
        return []

    out = [segments[0]]
    for i in range(1, len(segments)):
        _, prev_e = out[-1]
        _, cur_e = segments[i]
        out.append((prev_e, cur_e))
    return out


def bbox_of_segment(steps, segment):
    s, e = segment
    min_lon, min_lat = float("inf"), float("inf")
    max_lon, max_lat = float("-inf"), float("-inf")

    for i in range(s, e + 1):
        lon = steps["geo_x"].iloc[i]
        lat = steps["geo_y"].iloc[i]

        min_lon = min(min_lon, lon)
        min_lat = min(min_lat, lat)
        max_lon = max(max_lon, lon)
        max_lat = max(max_lat, lat)

    return min_lon, min_lat, max_lon, max_lat

def utm_zone_from_lon(lon):
    return int((lon + 180) // 6) + 1

def make_segment_transformer(min_lon, min_lat, max_lon, max_lat):
    center_lon = 0.5 * (min_lon + max_lon)
    center_lat = 0.5 * (min_lat + max_lat)

    zone = utm_zone_from_lon(center_lon)
    hemi = "N" if center_lat >= 0 else "S"
    epsg = 32600 + zone if hemi == "N" else 32700 + zone

    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    return fwd, inv, zone, hemi, epsg

def padded_utm_bbox(min_lon, min_lat, max_lon, max_lat, padding, max_cell_size):
    fwd, inv, zone, hemi, epsg = make_segment_transformer(min_lon, min_lat, max_lon, max_lat)
    # all bbox corners in same crs
    corners_lonlat = [
        (min_lon, min_lat),
        (min_lon, max_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
    ]
    corners_utm = [fwd.transform(lon, lat) for lon, lat in corners_lonlat]

    xs = [p[0] for p in corners_utm]
    ys = [p[1] for p in corners_utm]

    min_utm_x = min(xs)
    max_utm_x = max(xs)
    min_utm_y = min(ys)
    max_utm_y = max(ys)
    pad_x = max((max_utm_x - min_utm_x) * padding, max_cell_size)
    pad_y = max((max_utm_y - min_utm_y) * padding, max_cell_size)

    utm_bbox = (
        min_utm_x - pad_x,
        min_utm_y - pad_y,
        max_utm_x + pad_x,
        max_utm_y + pad_y,
    )
    return utm_bbox, zone, hemi, epsg, fwd, inv

def grid_to_geo_walk(walk_segment, utm_bbox, width, height, inv_transformer):
    min_x, min_y, max_x, max_y = utm_bbox
    result = []

    for x, y in walk_segment:
        if width <= 1 or height <= 1:
            result.append((np.nan, np.nan))
            continue
        utm_x = min_x + x / (width - 1) * (max_x - min_x)
        utm_y = max_y - y / (height - 1) * (max_y - min_y)
        lon, lat = inv_transformer.transform(utm_x, utm_y)
        result.append((lon, lat))

    return result

class StateDependentWalker(MixedWalker):
    def __init__(self, data, animal_type, resolution, out_directory,
                 n_hmm_states=3,
                 time_col="timestamp",
                 lon_col="location-long",
                 lat_col="location-lat",
                 id_col="individual_local_identifier",
                 crs="EPSG:4326"):
        print("Version 0.1.7")
        apply_moveapps_id_dtype_patch()
        debug_patch_state()
        force_tc_id_object_inplace(data)
        self.original_data = None
        if isinstance(data, mpd.TrajectoryCollection):
            import copy
            data_copy = copy.deepcopy(data)
            self.original_data = data_copy
        self.animal = animal_type
        self.n_hmm_states = n_hmm_states
        is_marine = animal_type == Animal.MARINE or animal_type == Animal.AIRBORNE
        if animal_type is Animal.AIRBORNE:
            mapping = None
            self.is_marine = True
        elif animal_type is Animal.MARINE:
            mapping = marine_kernels_baseline_crw(5, 5, 1, 1)
        else:
            mapping = create_mixed_kernel_parameters(animal_type, 5)
        super().__init__(data, mapping, resolution, out_directory, time_col, lon_col, lat_col, id_col, crs, is_marine)


    def generate_walks(self, out_dir=None, dt_tolerance=0.5, rnge=200, movement_policy=None, max_cell_size=10, water_mode:WaterMode=WaterMode.AVOID, is_brownian = False):
        super()._process_movebank_data()

        if self.original_data is None:
            self.original_data = self.animal_proc.traj_coll

        t_col = self.original_data.t
        id_col = self.original_data.get_traj_id_col()

        [corZs, brwZs] = self.animal_proc.get_hmm_kernels(dt_tolerance=dt_tolerance,
                                                          rnge=rnge,
                                                          out_dir=out_dir,
                                                          num_states=self.n_hmm_states)
        Za, Zb, Zc = brwZs if is_brownian and self.animal is not Animal.AIRBORNE else corZs
        rnge = Za.rnge
        py_kernels = [
            normalize_kernel(Z.Z)
            for Z in [Za, Zb, Zc]
            if Z.Z is not None
            and np.sum(Z) != 0
        ]
        NUM_STATES = len(py_kernels)

        t_pol = TimeStepPolicy(timestep_s=20 * 60) if movement_policy is None else movement_policy

        steps_dict = self.animal_proc.create_movement_data_dict(has_states=True)
        per_animal_gdfs = []
        aid = 0
        for animal_id, trajectory in steps_dict.items():
            aid += 1
            steps = trajectory.df
            # segments as index-intervals with overlaps, e.g. [(0, 3), (3, 5), (5, 9)].
            segments = trajectory_segments(steps, max_cell_size, self.resolution)
            segments = merge_singletons(segments)
            segments = make_overlapping(segments)

            animal_rows = []

            for segment in segments:
                min_lon, min_lat, max_lon, max_lat = bbox_of_segment(steps, segment)
                # convert bbox to UTM
                print(f"min_lon: {min_lon}, min_lat: {min_lat}, max_lon: {max_lon}, max_lat: {max_lat}")
                utm_bbox, zone, hemi, epsg_code, fwd, inv = padded_utm_bbox(
                    min_lon, min_lat, max_lon, max_lat,
                    padding=0.2,
                    max_cell_size=max_cell_size
                )
                print(f"utm_bbox: {utm_bbox}")
                min_utm_x, min_utm_y, max_utm_x, max_utm_y = utm_bbox
                # regular grid
                Nx, Ny = AnimalMovementProcessor.grid_shape_from_bbox(utm_bbox, self.resolution)
                # padded geo bbox
                min_lon, min_lat = inv.transform(min_utm_x, min_utm_y)
                max_lon, max_lat = inv.transform(max_utm_x, max_utm_y)
                # terrain for this segment
                terrain = None
                # sample landcover of new bounds
                if self.animal != Animal.AIRBORNE:
                    terrain = landcover_to_discrete_ptr(file_path=self.animal_proc.terrain_TIFFs[str(animal_id)],
                                                        res_x=Nx, res_y=Ny,
                                                        min_lon=min_lon, min_lat=min_lat,
                                                        max_lon=max_lon, max_lat=max_lat)
                cell_size = (max_utm_x - min_utm_x) / Nx
                cell_size = min(max(cell_size, 1.0), max_cell_size)

                # track segment boundaries so we can slice full_path per original segment
                seg_start, seg_end = segment
                for st_idx in range(seg_start, seg_end):
                    en_idx = st_idx + 1
                    print(f"[{aid-1} | {len(steps_dict)}] : ({st_idx} / {len(steps) - 1})\n")
                    start_lon = steps["geo_x"].iloc[st_idx]
                    start_lat = steps["geo_y"].iloc[st_idx]
                    end_lon = steps["geo_x"].iloc[en_idx]
                    end_lat = steps["geo_y"].iloc[en_idx]

                    start_time, end_time = steps["time"].iloc[st_idx], steps["time"].iloc[en_idx]
                    state = min(NUM_STATES - 1, steps["state"].iloc[st_idx])

                    # start coordinates to UTM
                    st_utm_x, st_utm_y = fwd.transform(start_lon, start_lat)
                    en_utm_x, en_utm_y = fwd.transform(end_lon, end_lat)
                    # start, end GRID
                    start_x, start_y = AnimalMovementProcessor.utm_to_grid(
                        Nx, Ny, min_utm_x, min_utm_y, max_utm_x, max_utm_y,
                        st_utm_x, st_utm_y
                    )
                    end_x, end_y = AnimalMovementProcessor.utm_to_grid(
                        Nx, Ny, min_utm_x, min_utm_y, max_utm_x, max_utm_y,
                        en_utm_x, en_utm_y
                    )

                    if start_x == end_x and start_y == end_y:
                        animal_rows.append({
                            id_col: animal_id,
                            t_col: steps["time"].iloc[st_idx],
                            "geometry": Point(start_lon, start_lat)
                        })
                        continue

                    # Walker parameters
                    T, S = t_pol.resolve((start_x, start_y), (end_x, end_y), start_time, end_time)
                    T = int(np.ceil(T * 1.5))
                    D = 1 if is_brownian else 8

                    if self.animal is Animal.AIRBORNE:
                        target = 2 * S + 1
                        grid_kernel = resize(
                            py_kernels[state],
                            (target, target),
                            order=1,
                            mode="reflect",
                            anti_aliasing=True,
                            preserve_range=True
                        )
                        grid_kernel = np.maximum(grid_kernel, 0)
                        grid_kernel /= grid_kernel.sum()
                    else:
                        # kernel parameters
                        kernel_radius = int(S * cell_size)
                        kernel_radius = min(rnge, kernel_radius)

                        clipped_kernel = normalize_kernel(clip_kernel(py_kernels[state], kernel_radius))
                        grid_kernel = resample_kernel_to_grid(clipped_kernel, cell_size, S)

                    h, w = grid_kernel.shape
                    print(f"W{w} : S: {S}\n")
                    assert w == 2 * S + 1 and h == 2 * S + 1
                    c_kernels = correlated_kernels_from_matrix(grid_kernel, w,h, directions=D)

                    print(f"[{start_time} - {end_time}]: {start_x}, {start_y} -> {end_x}, {end_y}: S - {S} T - {T} {Nx} x {Ny} - State {state}\n")

                    if self.animal is not Animal.AIRBORNE:
                        kmap = kernels_map_single_kernel(terrain, c_kernels, self.mapping, water_allowed=water_mode is not WaterMode.FORBID)
                        # Initialize DP matrix for the current start point
                        walk_ptr = single_state_walk(T,
                                                     kmap=kmap,
                                                     terrain=terrain,
                                                     start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y)
                        kernels_map3d_free(kmap)
                    else:
                        dp = correlated_walk_init(c_kernels, Nx, Ny,
                                                  T, start_x, start_y)
                        d = direction_from_points(start_x, start_y, end_x, end_y, D)

                        walk_ptr = correlated_backtrace(dp, T, c_kernels, end_x, end_y, d, out_ptr=True)
                        tensor4D_free(dp, T)
                        tensor_free(c_kernels)

                    if walk_ptr is not None:
                        walk_segment = get_walk_points(walk_ptr)
                        geo_walk = grid_to_geo_walk(walk_segment, utm_bbox, Nx, Ny, inv)
                        times = pd.date_range(
                            start=start_time,
                            end=end_time,
                            periods=len(geo_walk)
                        )
                        print("geo_walk sample:", geo_walk[:5])
                        for (lon, lat), t in zip(geo_walk, times):
                            animal_rows.append({
                                id_col: animal_id,
                                t_col: t,
                                "geometry": Point(lon, lat)
                            })
                        dll.point2d_array_free(walk_ptr)
                    else:
                        animal_rows.append({
                            id_col: animal_id,
                            t_col: steps["time"].iloc[st_idx],
                            "geometry": Point(start_lon, start_lat)
                        })

                terrain_map_free(terrain)

            animal_gdf = gpd.GeoDataFrame(animal_rows, geometry="geometry" ,crs="EPSG:4326")
            per_animal_gdfs.append(animal_gdf)

        # Combine all animals into a single GeoDataFrame and create one TrajectoryCollection
        combined_gdf = pd.concat(per_animal_gdfs, ignore_index=True)
        combined_gdf[t_col] = pd.to_datetime(combined_gdf[t_col])

        return merge_traj_collections(self.original_data, combined_gdf)
