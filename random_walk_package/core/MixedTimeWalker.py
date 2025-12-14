import os

import geopandas as gpd
import movingpandas as mpd
import pandas as pd

from random_walk_package import dll
from random_walk_package import get_walk_points
from random_walk_package.bindings import parse_terrain
from random_walk_package.bindings.mixed_walk import environment_mixed_walk
from random_walk_package.core.MixedWalker import MixedWalker
from random_walk_package.core.WalkerHelper import WalkerHelper


class MixedTimeWalker(MixedWalker):
    def __init__(self, data, env_data, kernel_mapping, resolution, out_directory, env_samples,
                 kernel_resolver=None,
                 time_col="timestamp",
                 lon_col="location-long",
                 lat_col="location-lat",
                 id_col="tag-local-identifier",
                 crs="EPSG:4326"
                 ):
        super().__init__(data, kernel_mapping, resolution, out_directory, time_col, lon_col, lat_col, id_col, crs)
        self.env_data = env_data
        self.env_paths: dict[str, str] = {}
        self.kernel_resolver = kernel_resolver
        self.env_samples = env_samples
        self._process_movebank_data()

    def _process_movebank_data(self):
        super()._process_movebank_data()
        self.animal_proc.env_samples = self.env_samples
        kernel_dir = os.path.join(self.out_directory, 'kernel_data')
        self.env_paths = self.animal_proc.kernel_params_per_animal_csv(df=self.env_data,
                                                                       kernel_resolver=self.kernel_resolver,
                                                                       time_stamp='timestamp',
                                                                       lon='longitude',
                                                                       lat='latitude',
                                                                       out_directory=kernel_dir)

    def generate_walks(self, env_weight=0.5):
        """
        Build random-walk trajectories for each animal, linear-interpolate timestamps
        for intermediate points, return a single mpd.TrajectoryCollection containing
        all animals.
        """
        steps_dict = self.animal_proc.create_movement_data_dict()
        per_animal_gdfs = []  # collect final GeoDataFrames per animal
        for animal_id, trajectory in steps_dict.items():
            terrain_map = parse_terrain(file=self.animal_proc.terrain_paths[animal_id], delim=' ')
            steps = trajectory.df
            full_path = []
            steps_df = steps_dict[animal_id].df
            idx = steps_df.index

            # track segment boundaries so we can slice full_path per original segment
            segment_boundaries = [0]
            for i in range(len(idx) - 1):
                # Get start/end positions and timestamps
                start_x, start_y = steps["grid_x"].iloc[i], steps["grid_y"].iloc[i]
                end_x, end_y = steps["grid_x"].iloc[i + 1], steps["grid_y"].iloc[i + 1]
                start_date, end_date = steps["time"].iloc[i], steps["time"].iloc[i + 1]
                if start_x == end_x and start_y == end_y:
                    segment = [(start_x, start_y)]
                    full_path.extend(segment)
                    segment_boundaries.append(len(full_path))
                    continue
                # Count how many timestamps exist for the given interval
                sub_df = pd.read_csv(self.env_paths[animal_id])
                sub_df["timestamp"] = pd.to_datetime(sub_df["timestamp"], errors='coerce')
                sub_df = sub_df[(sub_df["timestamp"] >= start_date) & (sub_df["timestamp"] <= end_date)]
                number_records = sub_df["timestamp"].nunique()

                dimensions = self.animal_proc.env_samples, self.animal_proc.env_samples, number_records
                print("start")
                print(start_x, start_y, end_x, end_y)
                print(start_date, end_date, number_records)
                print(self.env_paths[animal_id])
                print(dimensions)
                manhattan = abs(start_x - end_x) + abs(start_y - end_y)
                T = 5 if manhattan < 5 else manhattan
                print(T)
                # Initialize DP matrix for the current start point
                walk_ptr = environment_mixed_walk(T=T, mapping=self.mapping,
                                                  terrain=terrain_map,
                                                  csv_path=self.env_paths[animal_id],
                                                  dimensions=dimensions,
                                                  start_date=start_date,
                                                  end_date=end_date,
                                                  start_point=[start_x, start_y],
                                                  end_point=[end_x, end_y])
                print("walk created")
                dll.point2d_array_print(walk_ptr)

                if walk_ptr is not None:
                    segment = get_walk_points(walk_ptr)
                else:
                    segment = [(start_x, start_y), (end_x, end_y)]
                dll.point2d_array_free(walk_ptr)
                print("walk freed")
                print("grid freed")
                full_path.extend(segment[:-1] if len(segment) > 1 else segment)
                segment_boundaries.append(len(full_path))

            # terrain_map_free(terrain_map)
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
