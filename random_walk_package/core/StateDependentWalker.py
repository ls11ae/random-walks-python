import geopandas as gpd
import movingpandas as mpd
import pandas as pd

from random_walk_package import MixedWalker, get_walk_points, dll
from random_walk_package.bindings import parse_terrain, terrain_map_free
from random_walk_package.bindings.data_structures.kernels import normalize_kernel, clip_kernel
from random_walk_package.bindings.mixed_walk import state_dep_walk
from random_walk_package.core.WalkerHelper import WalkerHelper


class StateDependentWalker(MixedWalker):
    def __init__(self, data, mapping, resolution, out_directory,
                 time_col="timestamp",
                 lon_col="location-long",
                 lat_col="location-lat",
                 id_col="tag-local-identifier",
                 crs="EPSG:4326"):
        super().__init__(data, mapping, resolution, out_directory, time_col, lon_col, lat_col, id_col, crs)

    def generate_walks(self, serialization_dir=None):
        super()._process_movebank_data()
        MAX_T = 20
        KERNEL_RADIUS_CELLS = 10
        Za, Zb, Zc = self.animal_proc.get_hmm_kernels()
        dx_meter = dy_meter = Za.dx
        reso = Za.reso
        rnge = Za.rnge
        Za, Zb, Zc = self.animal_proc.get_hmm_kernels()

        Za = normalize_kernel(clip_kernel(Za.Z, KERNEL_RADIUS_CELLS))
        Zb = normalize_kernel(clip_kernel(Zb.Z, KERNEL_RADIUS_CELLS))
        Zc = normalize_kernel(clip_kernel(Zc.Z, KERNEL_RADIUS_CELLS))

        print("Kernel sum:", Za.sum(), Zb.sum(), Zc.sum())
        print("Kernel shape:", Za.shape)

        kernels = [Za, Zb, Zc]
        steps_dict = self.animal_proc.create_movement_data_dict(has_states=True)
        per_animal_gdfs = []  # collect final GeoDataFrames per animal
        for animal_id, trajectory in steps_dict.items():
            terrain_map = parse_terrain(file=self.animal_proc.terrain_paths[animal_id], delim=' ')
            steps = trajectory.df
            full_path = []
            steps_df = steps_dict[animal_id].df
            idx = steps_df.index

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
                # Get start/end positions and timestamps
                start_x, start_y = steps["grid_x"].iloc[i], steps["grid_y"].iloc[i]
                end_x, end_y = steps["grid_x"].iloc[i + 1], steps["grid_y"].iloc[i + 1]
                start_state, end_state = steps["state"].iloc[i], steps["state"].iloc[i + 1]
                start_time, end_time = steps["time"].iloc[i], steps["time"].iloc[i + 1]
                if start_x == end_x and start_y == end_y:
                    segment = [(start_x, start_y)]
                    full_path.extend(segment)
                    segment_boundaries.append(len(full_path))
                    continue

                manhattan = abs(start_x - end_x) + abs(start_y - end_y)
                T = min(MAX_T, max(5, abs(start_x - end_x) + abs(start_y - end_y)))
                print(T)
                # Initialize DP matrix for the current start point
                walk_ptr = state_dep_walk(T, start_state,
                                          kernels=kernels,
                                          mapping=self.mapping,
                                          terrain=terrain_map,
                                          start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y)
                print("walk created")

                if walk_ptr is not None:
                    segment = get_walk_points(walk_ptr)
                else:
                    segment = [(start_x, start_y), (end_x, end_y)]
                dll.point2d_array_free(walk_ptr)
                full_path.extend(segment[:-1] if len(segment) > 1 else segment)
                segment_boundaries.append(len(full_path))

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

        # Create a TrajectoryCollection with traj_id column used to split trajectories
        traj_collection = mpd.TrajectoryCollection(combined_gdf, traj_id_col="traj_id", t="time")
        return traj_collection
