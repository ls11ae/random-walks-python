import math

import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd

from random_walk_package import MixedWalker, get_walk_points, dll, tensor_free, tensor4D_free, AnimalMovementProcessor
from random_walk_package.bindings import parse_terrain, terrain_map_free, kernels_map3d_free, AIRBORNE, \
    create_mixed_kernel_parameters, MARINE, landcover_to_discrete_ptr
from random_walk_package.bindings.correlated_walk import correlated_walk_init, correlated_backtrace
from random_walk_package.bindings.data_structures.EnvWeights import EnvWeights
from random_walk_package.bindings.data_structures.kernel_terrain_mapping import marine_kernels_baseline
from random_walk_package.bindings.data_structures.kernels import normalize_kernel, clip_kernel, \
    correlated_kernels_from_matrix
from random_walk_package.bindings.mixed_walk import single_state_walk, kernels_map_single
from random_walk_package.core.MovementPolicy import TimeStepPolicy
from random_walk_package.core.WalkerHelper import WalkerHelper
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
                 time_col="timestamp",
                 lon_col="location-long",
                 lat_col="location-lat",
                 id_col="individual-local-identifier",
                 crs="EPSG:4326"):
        self.animal = animal_type
        is_marine = animal_type == MARINE or animal_type == AIRBORNE
        if animal_type is AIRBORNE:
            mapping = None
            self.is_marine = True
        elif animal_type is MARINE:
            mapping = marine_kernels_baseline(5, 5, 1, 1)
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

    def generate_walks(self, serialization_dir=None, dt_tolerance=0.5, rnge=200, movement_policy=None):
        super()._process_movebank_data()
        [corZs, brwZs] = self.animal_proc.get_hmm_kernels(dt_tolerance=dt_tolerance, rnge=rnge)

        Za, Zb, Zc = corZs

        dx_meter = dy_meter = Za.dx
        reso = Za.reso
        rnge = Za.rnge  # kernel area: [-rnge:rnge] x [-rnge:rnge]

        py_kernels = [
            normalize_kernel(Z.Z)
            for Z in [Za, Zb, Zc]
            if Z.Z is not None
        ]

        NUM_STATES = len(py_kernels)

        t_pol = TimeStepPolicy(timestep_s=20 * 60) if movement_policy is None else movement_policy

        steps_dict = self.animal_proc.create_movement_data_dict(has_states=True)
        per_animal_gdfs = []  # collect final GeoDataFrames per animal
        aid = 0
        for animal_id, trajectory in steps_dict.items():
            print(f"{aid} / {len(steps_dict) - 1}")
            aid += 1
            base_terrain = parse_terrain(file=self.animal_proc.terrain_paths[animal_id], delim=' ')
            steps = trajectory.df
            full_path = []
            steps_df = steps_dict[animal_id].df
            idx = steps_df.index

            # grid params
            (xmin, ymin, xmax, ymax),_ = self.animal_proc.bbox_utm(animal_id)
            Nx = base_terrain.contents.width
            Ny = base_terrain.contents.height

            cell_size = (xmax - xmin) / Nx

            print(f"cell size: {cell_size}\n")

            # track segment boundaries so we can slice full_path per original segment
            segment_boundaries = [0]
            for i in range(len(idx) - 1):
                print(f"{i} / {len(idx) - 1}\n")
                # Get start/end positions and timestamps
                start_x, start_y = steps["grid_x"].iloc[i], steps["grid_y"].iloc[i]
                end_x, end_y = steps["grid_x"].iloc[i + 1], steps["grid_y"].iloc[i + 1]

                start_lat, start_lon = steps["geo_x"].iloc[i], steps["geo_y"].iloc[i]
                end_lat, end_lon = steps["geo_x"].iloc[i + 1], steps["geo_y"].iloc[i + 1]

                st_utm_x, st_utm_y, zn, zl = AnimalMovementProcessor.geo_to_utm(start_lon, start_lat)
                en_utm_x, en_utm_y, _, _ = AnimalMovementProcessor.geo_to_utm(end_lon, end_lat)
                terrain = base_terrain
                if StateDependentWalker.__should_resample(st_utm_x, st_utm_y, en_utm_x, en_utm_y, cell_size):
                    print("RESAMPLING")
                    min_lon = min(start_lon, end_lon)
                    min_lat = min(start_lat, end_lat)
                    max_lon = max(start_lon, end_lon)
                    max_lat = max(start_lat, end_lat)
                    min_lon, min_lat, max_lon, max_lat = padded_bbox(min_lon, min_lat, max_lon, max_lat, padding=0.2)
                    min_utm_x,_,_,_ = AnimalMovementProcessor.geo_to_utm(min_lat, min_lon)
                    max_utm_x,_,_,_ = AnimalMovementProcessor.geo_to_utm(max_lat, max_lon)
                    min_utm_y,_,_,_ = AnimalMovementProcessor.geo_to_utm(min_lat, min_lon)
                    max_utm_y,_,_,_ = AnimalMovementProcessor.geo_to_utm(max_lat, max_lon)

                    start_x, start_y = AnimalMovementProcessor.utm_to_grid(Nx, Ny, min_utm_x, min_utm_y, max_utm_x, max_utm_y, st_utm_x, st_utm_y)
                    end_x, end_y = AnimalMovementProcessor.utm_to_grid(Nx, Ny, min_utm_x, min_utm_y, max_utm_x, max_utm_y, en_utm_x, en_utm_y)

                    cell_size = (max_utm_x - min_utm_x) / Nx
                    if cell_size < 1.0:
                        cell_size = 1.0
                        Nx = int(max_utm_x - min_utm_x)
                        Ny = int(max_utm_y - min_utm_y)
                    print(f"cell size: {cell_size}\n")
                    print(f"Nx: {Nx}, Ny: {Ny}\n")

                    terrain = landcover_to_discrete_ptr(file_path=self.animal_proc.terrain_TIFFs[str(animal_id)],
                                                        res_x=Nx, res_y=Ny,
                                                        min_lon=min_lon, min_lat=min_lat,
                                                        max_lon=max_lon, max_lat=max_lat,
                                                        txt_output_path=None)

                state = min(NUM_STATES - 1, steps["state"].iloc[i])
                start_time, end_time = steps["time"].iloc[i], steps["time"].iloc[i + 1]

                if abs(st_utm_x - en_utm_x) < 1 and abs(st_utm_y - en_utm_y) < 1:
                    segment = [(start_x, start_y)]
                    full_path.extend(segment)
                    segment_boundaries.append(len(full_path))
                    continue

                T, S = t_pol.resolve((start_x, start_y), (end_x, end_y), start_time, end_time, 2)
                D = 8

                kernel_radius = int(S * cell_size)
                print(f"Kernel radius: {kernel_radius}\n")
                print(f"T: {T} S: {S} \n")
                clipped_kernel = normalize_kernel(clip_kernel(py_kernels[state], kernel_radius))
                grid_kernel = resample_kernel_to_grid(clipped_kernel, cell_size, S)
                h, w = grid_kernel.shape
                assert w == 2 * S + 1 and h == 2 * S + 1

                c_kernels = correlated_kernels_from_matrix(grid_kernel, w,h, directions=D)

                if self.animal is not AIRBORNE:
                    kmap = kernels_map_single(terrain, c_kernels, self.mapping)
                    # Initialize DP matrix for the current start point
                    print("start walks")
                    walk_ptr = single_state_walk(T,
                                                 kmap=kmap,
                                                 terrain=terrain,
                                                 start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y)
                    kernels_map3d_free(kmap)
                else:
                    dp = correlated_walk_init(c_kernels, terrain.contents.width,
                                              terrain.contents.height,
                                              T, start_x, start_y)
                    d = direction_from_points(start_x, start_y, end_x, end_y, D)
                    print(f"from {(start_x, start_y)} to {(end_x, end_y)}\n")
                    print(f"T: {T} S: {S} ")
                    print(f"d: {d}\n")
                    walk_ptr = correlated_backtrace(dp, T, c_kernels, end_x, end_y, d, out_ptr=True)
                    tensor4D_free(dp, T)
                    tensor_free(c_kernels)
                if walk_ptr is not None:
                    segment = get_walk_points(walk_ptr)
                    dll.point2d_array_free(walk_ptr)
                else:
                    segment = [(start_x, start_y), (end_x, end_y)]
                full_path.extend(segment[:-1] if len(segment) > 1 else segment)
                segment_boundaries.append(len(full_path))

            # After loop, append final endpoint of last original step (to close path)
            last_row = steps_df.iloc[-1]
            last_grid = (int(last_row["grid_x"]), int(last_row["grid_y"]))
            # ensure last point is present
            if len(full_path) == 0 or full_path[-1][0] != last_grid[0] or full_path[-1][1] != last_grid[1]:
                full_path.append(last_grid)



        # Combine all animals into a single GeoDataFrame and create one TrajectoryCollection
        if len(per_animal_gdfs) == 0:
            return mpd.TrajectoryCollection(gpd.GeoDataFrame(columns=["geometry"]), traj_id_col="traj_id", t="time")

        combined = pd.concat(per_animal_gdfs, ignore_index=True)
        combined_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")

        # Ensure 'time' column is datetime-like
        combined_gdf["time"] = pd.to_datetime(combined_gdf["time"])
        print(combined_gdf.head())

        # Create a TrajectoryCollection with traj_id column used to split trajectories
        traj_collection = mpd.TrajectoryCollection(combined_gdf, traj_id_col="traj_id", t="time")
        return traj_collection
