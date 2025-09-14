import os

import numpy as np

from random_walk_package.bindings.brownian_walk import plot_walk_from_json
from random_walk_package.bindings.data_structures.point2D import get_walk_points
from random_walk_package.bindings.mixed_walk import *
from random_walk_package.core.AnimalMovement import AnimalMovementProcessor


class MixedTimeWalker:
    def __init__(self, resolution=200, T=30, grid_points_per_edge=5, duration_in_days=7, study_folder=None):
        self.T = T
        self.resolution = resolution
        self.grid_points_per_edge = grid_points_per_edge
        self.movebank_processor = None

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

        # Create the weather_data folder in study_folder and set as csv_path
        self.csv_path = os.path.join(self.study_folder, 'weather_data')
        if not os.path.exists(self.csv_path):
            os.makedirs(self.csv_path)

        self.serialization_path = os.path.join(self.study_folder, 'serialization')
        if not os.path.exists(self.serialization_path):
            os.makedirs(self.serialization_path)

        # Create the walks folder in study_folder and set as walks_path
        self.walks_path = os.path.join(self.study_folder, 'walks')
        if not os.path.exists(self.walks_path):
            os.makedirs(self.walks_path)

        self.terrain_path = self.study_folder
        self.duration_in_days = duration_in_days

    def preprocess(self):
        self.movebank_processor = AnimalMovementProcessor(self.movebank_study)
        self.terrain_path = self.movebank_processor.create_landcover_data_txt(self.resolution,
                                                                              out_directory=self.study_folder)
        self.csv_path = self.movebank_processor.fetch_gridded_weather_data(self.csv_path,
                                                                           days_to_fetch=self.duration_in_days,
                                                                           grid_points_per_edge=self.grid_points_per_edge)

    def generate_walk(self, start: tuple[int, int], end: tuple[int, int], output_file='time_walk.json',
                      serialized=True):
        """
        Generates a time-dependent walk using the C function time_walk_geo.
        Args:
            start (tuple): Starting coordinates (x, y).
            end (tuple): Ending coordinates (x, y).
            output_file (str): Name of the output file.
            serialized (bool): Whether to use serialized data.
        Returns:
            np.ndarray: Array of points representing the walk.
        """
        walk_path = os.path.join(self.walks_path, output_file)
        walk_ptr = time_walk_geo(
            T=self.T,
            csv_path=self.csv_path,
            terrain_path=self.terrain_path,
            walk_path=walk_path,
            serialization_path=self.serialization_path,
            grid_x=self.grid_points_per_edge,
            grid_y=self.grid_points_per_edge,
            start=start,
            goal=end,
            use_serialized=serialized
        )
        walk_np = get_walk_points(walk_ptr)
        dll.point2d_array_free(walk_ptr)
        plot_walk_from_json(walk_path, title=os.path.basename(self.movebank_study))
        return walk_np

    def generate_walk_multi(self, steps, output_file='time_walk.json', serialized=True):
        """
        Generates a time-dependent walk using the C function time_walk_geo.
        Args:
            steps (list of tuples): List of (x, y) points the walk should pass through.
            output_file (str): Name of the output file.
            serialized (bool): Whether to use serialized data.
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
            steps=steps,
            use_serialized=serialized,
            serialization_path=self.serialization_path
        )
        walk_np = get_walk_points(walk_ptr)
        dll.point2d_array_free(walk_ptr)
        plot_walk_from_json(walk_path, title=os.path.basename(self.movebank_study))

        return walk_np
