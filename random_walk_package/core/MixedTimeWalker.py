import numpy as np

from random_walk_package import timed_location_of
from random_walk_package.bindings.mixed_walk import *
from random_walk_package.bindings.plotter import plot_walk_from_json
from random_walk_package.core.AnimalMovement import AnimalMovementProcessor
from random_walk_package.data_sources.walk_visualization import walk_to_osm


class MixedTimeWalker:
    def __init__(self, resolution=200, T=30, grid_points_per_edge=5, duration_in_days=7, mapping=None,
                 study_folder=None):
        self.T = T
        self.resolution = resolution
        self.grid_points_per_edge = grid_points_per_edge
        self.movebank_processor = None
        self.mapping = mapping if mapping is not None else create_mixed_kernel_parameters(MEDIUM, 7)

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

        # weather data path per animal
        self.weather_data_dict: dict[str, str] = {}

        self.serialization_path = os.path.join(self.study_folder, 'serialization')
        if not os.path.exists(self.serialization_path):
            os.makedirs(self.serialization_path)

        # Create the walks folder in study_folder and set as walks_path
        self.walks_path = os.path.join(self.study_folder, 'walks')
        if not os.path.exists(self.walks_path):
            os.makedirs(self.walks_path)

        self.terrain_path = self.study_folder
        self.duration_in_days = duration_in_days
        self._preprocess()

    def _preprocess(self):
        self.movebank_processor = AnimalMovementProcessor(self.movebank_study)
        self.terrain_path = self.movebank_processor.create_landcover_data_txt(self.resolution,
                                                                              out_directory=self.study_folder)
        self.weather_data_dict = self.movebank_processor.fetch_gridded_weather_data(self.csv_path,
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
            mapping=self.mapping,
            grid_x=self.grid_points_per_edge,
            grid_y=self.grid_points_per_edge,
            start=start,
            goal=end
        )
        walk_np = get_walk_points(walk_ptr)
        dll.point2d_array_free(walk_ptr)
        plot_walk_from_json(walk_path, title=os.path.basename(self.movebank_study))
        return walk_np

    def generate_walk_from_movebank(self, serialized=True, output_prefix='time_walk'):
        """
        Generate time-dependent walks for each animal using Movebank steps.
        - Computes T per segment automatically as Manhattan distance (min 5).
        - Uses per-animal terrain files from preprocessed landcover data.
        - Uses the same weather logic via csv_path.

        Returns:
            dict[str, list[tuple[int, int]]]: A dict mapping animal_id to the concatenated grid-coordinate path.
        """
        # Ensure preprocessing is done (terrain, weather)
        if self.movebank_processor is None:
            self._preprocess()

        # Ensure per-animal terrain mapping is available
        if not isinstance(self.terrain_path, dict):
            self.terrain_path = self.movebank_processor.create_landcover_data_txt(
                self.resolution, out_directory=self.study_folder
            )

        # Steps from Movebank
        grid_steps_dict, geo_steps_dict, time_stamps_dict = self.movebank_processor.create_movement_data(samples=-1,
                                                                                                         time_stamped=True)

        geodetic_walks: dict[str, list[tuple[float, float]]] = {}
        for animal_id, steps in grid_steps_dict.items():
            if not steps or len(steps) < 2:
                continue

            # Pick per-animal terrain file if available, else fall back to a shared path
            terrain_for_animal = (
                self.terrain_path[animal_id]
                if isinstance(self.terrain_path, dict) and animal_id in self.terrain_path
                else self.terrain_path
            )

            full_path: list[tuple[int, int]] = []

            for i in range(len(steps) - 1):
                start_x, start_y = steps[i]
                end_x, end_y = steps[i + 1]
                start_time = time_stamps_dict[animal_id][i]
                end_time = time_stamps_dict[animal_id][i + 1]

                start_t_loc = timed_location_of(start_x, start_y, start_time)
                end_t_loc = timed_location_of(end_x, end_y, end_time)

                sx, sy = int(start_x), int(start_y)
                ex, ey = int(end_x), int(end_y)

                # Skip zero-length segments but keep the point
                if sx == ex and sy == ey:
                    full_path.append((sx, sy))
                    continue

                # Automatic T based on segment length (Manhattan), min 5
                t = abs(sx - ex) + abs(sy - ey)
                self.T = 5 if t < 5 else t

                # Unique output per segment (keeps JSON artifacts for inspection)
                print("Starting walk: " + str(sx) + ", " + str(sy) + " -> " + str(ex) + ", " + str(ey) + "")
                walk_ptr = time_walk_geo(
                    T=self.T,
                    csv_path=self.weather_data_dict[animal_id],
                    terrain_path=terrain_for_animal,
                    mapping=self.mapping,
                    grid_x=self.grid_points_per_edge,
                    grid_y=self.grid_points_per_edge,
                    start=start_t_loc,
                    goal=end_t_loc,
                )

                segment = (
                    get_walk_points(walk_ptr)
                    if walk_ptr is not None
                    else [(sx, sy), (ex, ey)]
                )

                # Free native memory
                dll.point2d_array_free(walk_ptr)

                # Stitch segments (omit last point to avoid duplication across segments)
                if len(segment) > 0:
                    full_path.extend(segment[:-1])

            # Append the final target point
            last_px, last_py = int(steps[-1][0]), int(steps[-1][1])
            full_path.append((last_px, last_py))

            grid_steps_dict[animal_id] = self.movebank_processor.grid_coordinates_to_geodetic(steps, animal_id)
            geodetic_path = self.movebank_processor.grid_coordinates_to_geodetic(full_path, animal_id)
            geodetic_walks[animal_id] = geodetic_path
            walk_to_osm(walk_coords_or_dict=geodetic_path, original_coords=geo_steps_dict[animal_id],
                        step_annotations=grid_steps_dict, animal_id=animal_id, walk_path=self.walks_path,
                        annotated=True)
        map_path = os.path.join(self.walks_path, "entire_study.html")
        walk_to_osm(geodetic_walks, None, "entire study", self.walks_path, grid_steps_dict, map_path)
        return map_path
