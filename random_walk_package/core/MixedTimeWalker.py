from random_walk_package.bindings.brownian_walk import plot_combined_terrain, plot_terrain_from_json

from random_walk_package.bindings.data_structures.point2D import get_walk_points

from random_walk_package.bindings.data_processing.movebank_parser import extract_steps_from_csv

from random_walk_package.bindings.data_processing.walk_json import walk_to_json

from random_walk_package.bindings.correlated_walk import *
from random_walk_package.bindings.mixed_walk import *
from random_walk_package.core.AnimalMovement import AnimalMovementProcessor
import numpy as np
import os


class MixedTimeWalker:
    def __init__(self, resolution = 200, T=30, grid_points_per_edge = 5, duration_in_days=7, study_folder=None):
        self.T = T
        self.resolution = resolution
        self.grid_points_per_edge = grid_points_per_edge

        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        base_project_dir = os.path.join(self.script_dir, '..')
        self.study_folder = os.path.join(base_project_dir, 'resources')
        if study_folder:
            self.study_folder = os.path.join(self.study_folder, study_folder)

        print(f"Using study folder: {self.study_folder}")

        # Find the only CSV file in the study_folder
        csv_files = [f for f in os.listdir(self.study_folder) if f.endswith('.csv')]
        if len(csv_files) != 1:
            raise FileNotFoundError("Expected exactly one CSV file in the study folder.")
        self.movebank_study = os.path.join(self.study_folder, csv_files[0])

        # Create weather_data folder in study_folder and set as csv_path
        self.csv_path = os.path.join(self.study_folder, 'weather_data')
        if not os.path.exists(self.csv_path):
            os.makedirs(self.csv_path)

        # Create walks folder in study_folder and set as walks_path
        self.walks_path = os.path.join(self.study_folder, 'walks')
        if not os.path.exists(self.walks_path):
            os.makedirs(self.walks_path)
        
        self.terrain_path = self.study_folder
        self.duration_in_days = duration_in_days

    def preprocess(self):
        self.movebank_processor = AnimalMovementProcessor(self.movebank_study)
        self.terrain_path = self.movebank_processor.create_landcover_data_txt(self.resolution, out_directory=self.study_folder)
        self.csv_path = self.movebank_processor.fetch_gridded_weather_data(self.csv_path, days_to_fetch=self.duration_in_days, grid_points_per_edge=self.grid_points_per_edge)
        

    def generate_walk(self, start, end, output_file='time_walk.json'):
        """
        Generates a time-dependent walk using the C function time_walk_geo.
        Args:
            start (tuple): Starting coordinates (x, y).
            end (tuple): Ending coordinates (x, y).
            grid_x (int): Number of grid cells in the x direction.
            grid_y (int): Number of grid cells in the y direction.
        Returns:
            np.ndarray: Array of points representing the walk.
        """
        walk_path = os.path.join(self.walks_path, output_file)
        walk_ptr = time_walk_geo(
            T=self.T,
            csv_path=self.csv_path,
            terrain_path=self.terrain_path,
            walk_path=walk_path,
            grid_x=self.grid_points_per_edge,
            grid_y=self.grid_points_per_edge,
            start=start,
            goal=end
        )
        walk_np = get_walk_points(walk_ptr)
        dll.point2d_array_free(walk_ptr)
        plot_terrain_from_json(walk_path, title=os.path.basename(self.movebank_study))

    def generate_walk_multi(self, steps, output_file='time_walk.json'):
        """
        Generates a time-dependent walk using the C function time_walk_geo.
        Args:
            start (tuple): Starting coordinates (x, y).
            end (tuple): Ending coordinates (x, y).
            grid_x (int): Number of grid cells in the x direction.
            grid_y (int): Number of grid cells in the y direction.
        Returns:
            np.ndarray: Array of points representing the walk.
        """
        walk_path = os.path.join(self.walks_path, output_file)
        walk_ptr = time_walk_geo_multi(
            T=self.T,
            csv_path=self.csv_path,
            terrain_path=self.terrain_path,
            walk_path=walk_path,
            grid_x=self.grid_points_per_edge,
            grid_y=self.grid_points_per_edge,
            steps=steps
        )
        walk_np = get_walk_points(walk_ptr)
        dll.point2d_array_free(walk_ptr)
        plot_terrain_from_json(walk_path, title=os.path.basename(self.movebank_study))
        

    def generate_walk_movebank(self, csv_file, step_count):
        width = self.spatial_map.width
        height = self.spatial_map.height
        steps_ctype = extract_steps_from_csv(csv_file, step_count, width, height)
        height = self.spatial_map.height
        walk = mix_walk(width, height, self.spatial_map, self.tensor_map, self.T, 200, 200)
        walk_to_json(walk=walk, json_file='mixed_movebank_walks.json', steps=steps_ctype, terrain_map=self.spatial_map,
                     W=self.spatial_map.width, H=self.spatial_map.height)

        walk_to_json(walk=walk, json_file='walks.json', steps=steps_ctype, terrain_map=self.spatial_map,
                     W=self.spatial_map.width, H=height)
        return walk

    def mixed_walk_time(self, tensor4D, steps=None):
        full_path = []
        width = self.spatial_map.contents.width
        height = self.spatial_map.contents.height
        T = self.T

        if steps is None:
            steps = self.movebank_processor.create_movement_data(width=width, height=height, samples=10)

        # TODO:  add padding to bbox
        steps = [(57, 20), (130, 55)]

        for i in range(len(steps) - 1):
            start_x, start_y = steps[i]
            print(start_x)
            print(start_y)
            end_x, end_y = steps[i + 1]

            dp_matrix = time_walk_init(
                width, height,
                self.spatial_map,
                tensor4D,
                self.T,
                start_x,
                start_y
            )

            walk_ptr = time_walk_backtrace(dp_matrix, T, self.spatial_map, tensor4D, end_x, end_y, 0)

            segment = get_walk_points(walk_ptr)

            dll.tensor4D_free(dp_matrix, T)
            dll.point2d_array_free(walk_ptr)

            full_path.extend(segment[:-1])

        full_path.append(steps[-1])
        walk = np.array(full_path)
        plot_combined_terrain(self.spatial_map, walk, steps=steps, title=self.movebank_study)

    