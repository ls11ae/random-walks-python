import math

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
from random_walk_package.core.move_apps_patch import merge_traj_collections
from random_walk_package.data_sources.movebank_adapter import padded_bbox

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

class StateDependentWalker(MixedWalker):
    def __init__(self, data, animal_type, resolution, out_directory,
                 n_hmm_states=3,
                 time_col="timestamp",
                 lon_col="location-long",
                 lat_col="location-lat",
                 id_col="individual_local_identifier",
                 crs="EPSG:4326"):
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

    @staticmethod
    def __should_resample(st_utm_x, st_utm_y, en_utm_x, en_utm_y, cell_size, cell_factor=3.0,dist_factor=4.0):
        if cell_size > cell_factor:
            return True
        dx = en_utm_x - st_utm_x
        dy = en_utm_y - st_utm_y
        dist = (dx*dx + dy*dy)**0.5
        return dist > dist_factor


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
            steps_df = steps_dict[animal_id].df
            idx = steps_df.index

            animal_rows = []

            global_bbox = self.animal_proc.bbox_geo(animal_id)

            # track segment boundaries so we can slice full_path per original segment
            for i in range(len(idx) - 1):
                print(f"[{aid-1} | {len(steps_dict)}] : ({i} / {len(idx) - 1})\n")
                start_lat, start_lon = steps["geo_x"].iloc[i], steps["geo_y"].iloc[i]
                end_lat, end_lon = steps["geo_x"].iloc[i + 1], steps["geo_y"].iloc[i + 1]

                # geo bbox
                min_lon = min(start_lon, end_lon)
                min_lat = min(start_lat, end_lat)
                max_lon = max(start_lon, end_lon)
                max_lat = max(start_lat, end_lat)
                # add padding to geo bbox
                min_lon, min_lat, max_lon, max_lat = padded_bbox(min_lon, min_lat, max_lon, max_lat, padding=0.0)
                # convert bbox to UTM
                min_utm_x, min_utm_y, zone, hemi = AnimalMovementProcessor.geo_to_utm(min_lat, min_lon)
                max_utm_x, max_utm_y, _, _ = AnimalMovementProcessor.geo_to_utm(max_lat, max_lon)
                epsg_code = 32600 + zone if hemi >= "N" else 32700 + zone

                # start coordinates to UTM
                st_utm_x, st_utm_y,_,_ = AnimalMovementProcessor.geo_to_utm(start_lat, start_lon)
                en_utm_x, en_utm_y,_,_ = AnimalMovementProcessor.geo_to_utm(end_lat, end_lon)

                start_time, end_time = steps["time"].iloc[i], steps["time"].iloc[i + 1]

                state = min(NUM_STATES - 1, steps["state"].iloc[i])

                # upper bound for grid
                MAX_GRID_CELLS = 1000
                MAX_CELL_SIZE = max_cell_size
                max_window_size = MAX_GRID_CELLS * MAX_CELL_SIZE

                dx = abs(en_utm_x - st_utm_x)
                dy = abs(en_utm_y - st_utm_y)
                dist = max(dx, dy)

                if dist > max_window_size:
                    n = int(np.ceil(dist / max_window_size))
                else:
                    n = 1

                xs = np.linspace(st_utm_x, en_utm_x, n + 1)
                ys = np.linspace(st_utm_y, en_utm_y, n + 1)
                ts = pd.date_range(start=start_time, end=end_time, periods=n + 1)

                for j in range(n):
                    sub_st_x, sub_st_y = xs[j], ys[j]
                    sub_en_x, sub_en_y = xs[j + 1], ys[j + 1]
                    sub_start_time = ts[j]
                    sub_end_time = ts[j + 1]

                    min_utm_x = min(sub_st_x, sub_en_x)
                    max_utm_x = max(sub_st_x, sub_en_x)
                    min_utm_y = min(sub_st_y, sub_en_y)
                    max_utm_y = max(sub_st_y, sub_en_y)

                    # metric padding
                    dx = max_utm_x - min_utm_x
                    dy = max_utm_y - min_utm_y
                    pad_x = max(0.2 * dx, 2 * MAX_CELL_SIZE)
                    pad_y = max(0.2 * dy, 2 * MAX_CELL_SIZE)

                    min_utm_x -= pad_x
                    max_utm_x += pad_x
                    min_utm_y -= pad_y
                    max_utm_y += pad_y

                    utm_bbox = (min_utm_x, min_utm_y, max_utm_x, max_utm_y)

                    Nx, Ny = AnimalMovementProcessor.grid_shape_from_bbox(utm_bbox, self.resolution)
                    Nx = min(Nx, MAX_GRID_CELLS)
                    Ny = min(Ny, MAX_GRID_CELLS)

                    min_lon, min_lat = AnimalMovementProcessor.utm_to_geo(min_utm_x, min_utm_y, zone, hemi)
                    max_lon, max_lat = AnimalMovementProcessor.utm_to_geo(max_utm_x, max_utm_y, zone, hemi)

                    terrain = None
                    # sample landcover of new bounds
                    if self.animal != Animal.AIRBORNE:
                        terrain = landcover_to_discrete_ptr(file_path=self.animal_proc.terrain_TIFFs[str(animal_id)],
                                                            res_x=Nx, res_y=Ny,
                                                            min_lon=min_lon, min_lat=min_lat,
                                                            max_lon=max_lon, max_lat=max_lat)

                    cell_size = (max_utm_x - min_utm_x) / Nx
                    cell_size = min(max(cell_size, 1.0), MAX_CELL_SIZE)

                    start_x, start_y = AnimalMovementProcessor.utm_to_grid(
                        Nx, Ny, min_utm_x, min_utm_y, max_utm_x, max_utm_y,
                        sub_st_x, sub_st_y
                    )
                    end_x, end_y = AnimalMovementProcessor.utm_to_grid(
                        Nx, Ny, min_utm_x, min_utm_y, max_utm_x, max_utm_y,
                        sub_en_x, sub_en_y
                    )

                    T, S = t_pol.resolve((start_x, start_y), (end_x, end_y), sub_start_time, sub_end_time)
                    D = 1 if is_brownian else 8

                    kernel_radius = int(S * cell_size)
                    kernel_radius = max(rnge, kernel_radius)

                    clipped_kernel = normalize_kernel(clip_kernel(py_kernels[state], kernel_radius))
                    grid_kernel = resample_kernel_to_grid(clipped_kernel, cell_size, S)

                    h, w = grid_kernel.shape
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
                        terrain_map_free(terrain)
                    else:
                        dp = correlated_walk_init(c_kernels, Nx, Ny,
                                                  T, start_x, start_y)
                        d = direction_from_points(start_x, start_y, end_x, end_y, D)

                        walk_ptr = correlated_backtrace(dp, T, c_kernels, end_x, end_y, d, out_ptr=True)
                        tensor4D_free(dp, T)
                        tensor_free(c_kernels)

                    if walk_ptr is not None:
                        segment = get_walk_points(walk_ptr)
                        geo_walk = AnimalMovementProcessor.grid_to_geo_walk(segment, utm_bbox, Nx, Ny, epsg_code)
                        times = pd.date_range(
                            start=sub_start_time,
                            end=sub_end_time,
                            periods=len(geo_walk)
                        )
                        for (y, x), t in zip(geo_walk, times):
                            animal_rows.append({
                                id_col: animal_id,
                                t_col: t,
                                "geometry": Point(x, y)
                            })
                        dll.point2d_array_free(walk_ptr)
                    else:
                        animal_rows.append({
                            id_col: animal_id,
                            t_col: steps["time"].iloc[i],
                            "geometry": Point(start_lon, start_lat)
                        })

            animal_gdf = gpd.GeoDataFrame(animal_rows, geometry="geometry" ,crs="EPSG:4326")
            per_animal_gdfs.append(animal_gdf)


        # Combine all animals into a single GeoDataFrame and create one TrajectoryCollection
        combined_gdf = pd.concat(per_animal_gdfs, ignore_index=True)
        combined_gdf[t_col] = pd.to_datetime(combined_gdf[t_col])

        return merge_traj_collections(self.original_data, combined_gdf)
