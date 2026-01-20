import math

import geopandas as gpd
import movingpandas as mpd
import pandas as pd

from random_walk_package import MixedWalker, get_walk_points, dll, tensor_free, tensor4D_free
from random_walk_package.bindings import parse_terrain, terrain_map_free, kernels_map3d_free, AIRBORNE, \
    create_mixed_kernel_parameters, MARINE
from random_walk_package.bindings.correlated_walk import correlated_walk_init, correlated_backtrace
from random_walk_package.bindings.data_structures.EnvWeights import EnvWeights
from random_walk_package.bindings.data_structures.kernel_terrain_mapping import marine_kernels_baseline
from random_walk_package.bindings.data_structures.kernels import normalize_kernel, clip_kernel, \
    correlated_kernels_from_matrix
from random_walk_package.bindings.mixed_walk import single_state_walk, kernels_map_single
from random_walk_package.core.MovementPolicy import TimeStepPolicy
from random_walk_package.core.WalkerHelper import WalkerHelper


def direction_from_points(start_x, start_y, end_x, end_y, dirs=8):
    dx = start_x - end_x
    dy = start_y - end_y

    if dx == 0 and dy == 0:
        return 0

    angle_deg = math.degrees(math.atan2(dy, dx))
    angle_west_deg = (angle_deg - 180.0) % 360.0

    step = 360.0 / dirs
    return int(round(angle_west_deg / step)) % dirs


class StateDependentWalker(MixedWalker):
    def __init__(self, data, animal_type, resolution, out_directory,
                 time_col="timestamp",
                 lon_col="location-long",
                 lat_col="location-lat",
                 id_col="tag-local-identifier",
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

    def generate_walks(self, serialization_dir=None, dt_tolerance=0.5, rnge=200, movement_policy=None):
        super()._process_movebank_data()
        Za, Zb, Zc = self.animal_proc.get_hmm_kernels(dt_tolerance=dt_tolerance, rnge=rnge)
        dx_meter = dy_meter = Za.dx
        reso = Za.reso
        rnge = Za.rnge

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
            terrain_map = parse_terrain(file=self.animal_proc.terrain_paths[animal_id], delim=' ')
            steps = trajectory.df
            full_path = []
            steps_df = steps_dict[animal_id].df
            idx = steps_df.index
            kmaps = []

            # grid params
            """xmin, ymin, xmax, ymax = self.animal_proc.bbox_utm(animal_id)
            Nx = terrain_map.contents.width
            Ny = terrain_map.contents.height
            cell_size_x = (xmax - xmin) / Nx
            cell_size_y = (ymax - ymin) / Ny

            dx_cell = dx_meter / cell_size_x
            dy_cell = dy_meter / cell_size_y"""

            # track segment boundaries so we can slice full_path per original segment
            segment_boundaries = [0]
            for i in range(len(idx) - 1):
                print(f"{i} / {len(idx) - 1}\n")
                # Get start/end positions and timestamps
                start_x, start_y = steps["grid_x"].iloc[i], steps["grid_y"].iloc[i]
                end_x, end_y = steps["grid_x"].iloc[i + 1], steps["grid_y"].iloc[i + 1]

                state = min(NUM_STATES - 1, steps["state"].iloc[i])
                start_time, end_time = steps["time"].iloc[i], steps["time"].iloc[i + 1]

                if start_x == end_x and start_y == end_y:
                    segment = [(start_x, start_y)]
                    full_path.extend(segment)
                    segment_boundaries.append(len(full_path))
                    continue

                T, S = t_pol.resolve((start_x, start_y), (end_x, end_y), start_time, end_time, 2)
                D = 8
                print(f"T: {T} S: {S} \n")

                c_kernels = [
                    correlated_kernels_from_matrix(clip_kernel(k, S), width=2 * S + 1, height=2 * S + 1, directions=D)
                    for k in py_kernels]

                if self.animal is not AIRBORNE:
                    kmaps = [kernels_map_single(terrain_map, clip_kernel(Z, S), self.mapping) for Z in py_kernels]
                # Initialize DP matrix for the current start point
                if self.animal is not AIRBORNE:
                    print("start walks")
                    walk_ptr = single_state_walk(T,
                                                 kmap=kmaps[state],
                                                 terrain=terrain_map,
                                                 start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y)
                else:
                    dp = correlated_walk_init(c_kernels[state], terrain_map.contents.width,
                                              terrain_map.contents.height,
                                              T, start_x, start_y)
                    d = direction_from_points(start_x, start_y, end_x, end_y, D)
                    print(f"from {(start_x, start_y)} to {(end_x, end_y)}\n")
                    print(f"T: {T} S: {S} ")
                    print(f"d: {d}\n")
                    walk_ptr = correlated_backtrace(dp, T, c_kernels[state], end_x, end_y, d, out_ptr=True)
                    tensor4D_free(dp, T)
                    for k in c_kernels:
                        tensor_free(k)
                if walk_ptr is not None:
                    segment = get_walk_points(walk_ptr)
                    dll.point2d_array_free(walk_ptr)
                else:
                    segment = [(start_x, start_y), (end_x, end_y)]
                full_path.extend(segment[:-1] if len(segment) > 1 else segment)
                segment_boundaries.append(len(full_path))

                for kmap in kmaps:
                    kernels_map3d_free(kmap)
            terrain_map_free(terrain_map)
            # After loop, append final endpoint of last original step (to close path)
            last_row = steps_df.iloc[-1]
            last_grid = (int(last_row["grid_x"]), int(last_row["grid_y"]))
            # ensure last point is present
            if len(full_path) == 0 or full_path[-1][0] != last_grid[0] or full_path[-1][1] != last_grid[1]:
                full_path.append(last_grid)

            # convert full_path (list of (x,y)) into geodetic DataFrame
            geodetic_path_df = self.animal_proc.grid_to_geo_path(full_path, animal_id)
            # If grid_to_geo_path returns list of tuples, convert:
            if not isinstance(geodetic_path_df, pd.DataFrame):
                geodetic_path_df = pd.DataFrame(geodetic_path_df, columns=["longitude", "latitude"])

            # Build timestamped segments using segment_boundaries
            rows = WalkerHelper.create_timed_df(steps_df, geodetic_path_df, animal_id, idx, segment_boundaries)

            final_df = pd.concat(rows, ignore_index=True)
            final_df["geometry"] = gpd.points_from_xy(final_df.longitude, final_df.latitude)
            final_gdf = gpd.GeoDataFrame(final_df, geometry="geometry", crs="EPSG:4326")

            per_animal_gdfs.append(final_gdf)

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
