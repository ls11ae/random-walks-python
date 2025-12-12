import os.path
import subprocess
from pathlib import Path

import geopandas as gpd
import movingpandas as mpd
import pandas as pd

from random_walk_package.bindings.mixed_walk import *
from random_walk_package.bindings.plotter import plot_combined_terrain
from random_walk_package.core.AnimalMovementNew import AnimalMovementProcessor
from random_walk_package.core.WalkerHelper import WalkerHelper

try:
    from random_walk_package.bindings.cuda.mixed_gpu import preprocess_mixed_gpu, mixed_walk_gpu, free_kernel_pool

    CUDA_AVAILABLE = True
except (AttributeError, ImportError, OSError) as e:
    print(f"CUDA not available: {e}")
    CUDA_AVAILABLE = False


    def preprocess_mixed_gpu(*args, **kwargs):
        print("CUDA not available - using CPU fallback")
        return None


    def mixed_walk_gpu(*args, **kwargs):
        raise RuntimeError("CUDA not available on this system")


    def free_kernel_pool(*args, **kwargs):
        pass


class MixedWalker:
    def __init__(self, data,
                 kernel_mapping,
                 resolution,
                 out_directory,
                 time_col="timestamp",
                 lon_col="longitude",
                 lat_col="latitude",
                 id_col="animal_id",
                 crs="EPSG:4326"):
        self.data = data
        self.time_col = time_col
        self.lon_col = lon_col
        self.lat_col = lat_col
        self.id_col = id_col
        self.crs = crs
        self.resolution = resolution
        self.out_directory = out_directory
        self.movebank_processor = None
        self.mapping = kernel_mapping

    def _process_movebank_data(self):
        self.movebank_processor = AnimalMovementProcessor(data=self.data,
                                                          time_col=self.time_col,
                                                          lon_col=self.lon_col,
                                                          lat_col=self.lat_col,
                                                          id_col=self.id_col,
                                                          crs=self.crs)
        self.movebank_processor.create_landcover_data_txt(resolution=self.resolution,
                                                          out_directory=self.out_directory)

    @staticmethod
    def has_cuda():
        try:
            out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
            return b"CUDA" in out or b"NVIDIA" in out
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def generate_movebank_walks(self, serialization_dir=None):
        """
        Build random-walk trajectories for each animal, linear-interpolate timestamps
        for intermediate points, return a single mpd.TrajectoryCollection containing
        all animals.
        """
        self._process_movebank_data()
        use_cuda = self.has_cuda()
        steps_dict = self.movebank_processor.create_movement_data_dict()

        serialized: bool = serialization_dir is not None
        recmp: bool = True

        if serialization_dir is not None and Path(serialization_dir).exists() and any(
                Path(serialization_dir).iterdir()):
            recmp = False

        per_animal_gdfs = []  # collect final GeoDataFrames per animal

        for animal_id, trajectory in steps_dict.items():
            terrain_map = parse_terrain(file=self.movebank_processor.terrain_paths[animal_id], delim=' ')
            kernel_map = None
            steps = trajectory.df
            if serialized and recmp:
                tensor_map_terrain_serialize(terrain_map, self.mapping, serialization_dir)
                print(f"Serialized terrain map to {serialization_dir}")
            else:
                print("create kernels")
                kernel_map = get_tensor_map_terrain(terrain_map, self.mapping)
                print("create tensor map")

            kernel_pool = preprocess_mixed_gpu(kernel_map, terrain_map) if use_cuda else None

            width = terrain_map.width
            height = terrain_map.height
            full_path = []
            steps_df = steps_dict[animal_id].df
            idx = steps_df.index

            # track segment boundaries so we can slice full_path per original segment
            segment_boundaries = [0]

            for i in range(len(idx) - 1):
                start_x, start_y = steps["grid_x"].iloc[i], steps["grid_y"].iloc[i]
                end_x, end_y = steps["grid_x"].iloc[i + 1], steps["grid_y"].iloc[i + 1]

                print("Start: " + str(start_x) + ", " + str(start_y))
                manhattan = abs(start_x - end_x) + abs(start_y - end_y)
                T = 5 if manhattan < 5 else manhattan

                dp_dir = None
                if start_x == end_x and start_y == end_y:
                    # still record a single point segment
                    segment = [(start_x, start_y)]
                else:
                    if serialized:
                        dp_dir = os.path.join(serialization_dir, str(start_x), str(start_y),
                                              "DP_T" + str(T) + "_X" + str(start_x) + "_Y" + str(start_y))
                        if os.path.exists(dp_dir):
                            recmp = True
                        else:
                            os.makedirs(dp_dir)

                    # Initialize DP matrix for the current start point
                    dp_matrix_step = None
                    if use_cuda:
                        walk_ptr = mixed_walk_gpu(T, width, height, start_x, start_y, end_x, end_y, kernel_map,
                                                  self.mapping, terrain_map, serialized, serialization_dir,
                                                  kernel_pool)
                    else:
                        dp_matrix_step = mix_walk(W=width, H=height, terrain_map=terrain_map,
                                                  kernels_map=kernel_map,
                                                  start_x=int(start_x),
                                                  start_y=int(start_y),
                                                  T=T,
                                                  serialize=serialized,
                                                  recompute=recmp,
                                                  serialize_path=serialization_dir,
                                                  mapping=self.mapping)
                        # Backtrace from the end point
                        walk_ptr = mix_backtrace_c(
                            DP_Matrix=dp_matrix_step,
                            T=T,
                            tensor_map=kernel_map,
                            terrain=terrain_map,
                            end_x=int(end_x),
                            end_y=int(end_y),
                            serialize=serialized,
                            serialize_path=serialization_dir if serialized else "",
                            dp_dir=dp_dir if dp_dir else "",
                            mapping=self.mapping
                        )

                    if walk_ptr is not None:
                        segment = get_walk_points(walk_ptr)
                    else:
                        segment = [(start_x, start_y), (end_x, end_y)]

                    if not serialized and not use_cuda:
                        dll.tensor4D_free(dp_matrix_step, T)
                    dll.point2d_array_free(walk_ptr)

                full_path.extend(segment[:-1] if len(segment) > 1 else segment)
                segment_boundaries.append(len(full_path))

            # After loop, append final endpoint of last original step (to close path)
            last_row = steps_df.iloc[-1]
            last_grid = (int(last_row["grid_x"]), int(last_row["grid_y"]))
            # ensure last point is present
            if len(full_path) == 0 or full_path[-1][0] != last_grid[0] or full_path[-1][1] != last_grid[1]:
                full_path.append(last_grid)

            # convert full_path (list of (x,y)) into geodetic DataFrame
            geodetic_path_df = self.movebank_processor.grid_to_geo_path(full_path, animal_id)
            # If grid_to_geo_path returns list of tuples, convert:
            if not isinstance(geodetic_path_df, pd.DataFrame):
                geodetic_path_df = pd.DataFrame(geodetic_path_df, columns=["longitude", "latitude"])

            # Build timestamped segments using segment_boundaries
            rows = WalkerHelper.create_timed_df(steps_df, geodetic_path_df, animal_id, idx, segment_boundaries)

            final_df = pd.concat(rows, ignore_index=True)
            final_df["geometry"] = gpd.points_from_xy(final_df.longitude, final_df.latitude)
            final_gdf = gpd.GeoDataFrame(final_df, geometry="geometry", crs="EPSG:4326")

            per_animal_gdfs.append(final_gdf)

            free_kernel_pool(kernel_pool)
            kernels_map3d_free(kernel_map)

        # Combine all animals into a single GeoDataFrame and create one TrajectoryCollection
        if len(per_animal_gdfs) == 0:
            return mpd.TrajectoryCollection(gpd.GeoDataFrame(columns=["geometry"]), traj_id_col="traj_id", t="time")

        combined = pd.concat(per_animal_gdfs, ignore_index=True)
        combined_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")

        # Ensure 'time' column is datetime-like
        combined_gdf["time"] = pd.to_datetime(combined_gdf["time"])

        # Create a TrajectoryCollection with traj_id column used to split trajectories
        traj_collection = mpd.TrajectoryCollection(combined_gdf, traj_id_col="traj_id", t="time")
        return traj_collection

    @staticmethod
    def generate_custom_walks(terrain, steps, T, kernel_mapping, plot=False, plot_title="Mixed Walk"):
        tensor_map = get_tensor_map_terrain(terrain, kernel_mapping)
        walk = WalkerHelper.generate_multistep_walk(terrain, steps, T, kernel_mapping, tensor_map)
        kernels_map3d_free(tensor_map)
        if plot:
            plot_combined_terrain(terrain=terrain, walk_points=walk, steps=steps, title="Mixed Walk")
        return walk
