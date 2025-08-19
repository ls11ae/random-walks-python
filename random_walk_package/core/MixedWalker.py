from pathlib import Path

import numpy as np

from random_walk_package.bindings.brownian_walk import plot_combined_terrain
from random_walk_package.bindings.data_structures.point2D import get_walk_points
from random_walk_package.bindings.mixed_walk import *
from random_walk_package.core.AnimalMovement import AnimalMovementProcessor


class MixedWalker:
    def __init__(self, T=30, resolution=100, kernel_mapping=None,study_folder=None):
        self.T = T
        self.resolution = resolution
        self.spatial_map = None
        self.mapping = kernel_mapping if kernel_mapping is not None else create_mixed_kernel_parameters(MEDIUM, 7)
        self.tensor_map = None
        self.movebank_processor = None

        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        self.base_project_dir = os.path.abspath(os.path.join(self.script_dir, '..'))

        self.study_folder = os.path.join(self.base_project_dir, 'resources')
        if study_folder:
            self.study_folder = os.path.join(self.study_folder, study_folder)

        print(f"Using study folder: {self.study_folder}")

        # Find the only CSV file in the study_folder
        csv_files = [f for f in os.listdir(self.study_folder) if f.endswith('.csv')]
        if len(csv_files) != 1:
            raise FileNotFoundError("Expected exactly one CSV file in the study folder.")
        self.movebank_study = os.path.join(self.study_folder, csv_files[0])

        self.serialization_path = os.path.join(self.study_folder, 'serialization')
        if not os.path.exists(self.serialization_path):
            os.makedirs(self.serialization_path)

        # Create walks folder in study_folder and set as walks_path
        self.walks_path = os.path.join(self.study_folder, 'walks')
        if not os.path.exists(self.walks_path):
            os.makedirs(self.walks_path)

        self.terrain_path = self.study_folder
        self._set_kernels()

    def _set_kernels(self):
        self.movebank_processor = AnimalMovementProcessor(self.movebank_study)
        self.terrain_path = self.movebank_processor.create_landcover_data_txt(self.resolution,
                                                                              out_directory=self.study_folder)
        self.spatial_map = parse_terrain(file=self.terrain_path, delim=' ')

    def generate_walk(self, steps=None, serialized=False):
        recmp: bool = True
        serialization_dir = Path(self.base_project_dir) / 'resources' / self.serialization_path / 'tensors'
        print(
            f"Generating mixed walk with {self.T} time steps, resolution {self.resolution}, kernels {self.mapping}, "
            f"serialized {serialized}"
        )
        if serialization_dir.exists() and any(serialization_dir.iterdir()):
            recmp = False
        if serialized and recmp:
            tensor_map_terrain_serialize(self.spatial_map, self.mapping, self.serialization_path)
            print(f"Serialized terrain map to {self.serialization_path}")
        else:
            self.tensor_map = get_tensor_map_terrain(self.spatial_map, self.mapping)

        full_path = []
        width = self.spatial_map.width
        height = self.spatial_map.height
        if steps is None:
            steps = self.movebank_processor.create_movement_data(width=width, height=height, samples=3)

        print(steps)
        if steps is not None:
            steps = steps

        for i in range(len(steps) - 1):
            start_x, start_y = steps[i]
            end_x, end_y = steps[i + 1]
            print(start_x, start_y, end_x, end_y)
            print(self.serialization_path)
            dp_dir = os.path.join(self.base_project_dir, 'resources', self.serialization_path,
                                  "DP_T" + str(self.T) + "_X" + str(start_x) + "_Y" + str(start_y))
            print(f"Recomputing {recmp}")
            # Initialize DP matrix for the current start point
            dp_matrix_step = mix_walk(W=width, H=height, terrain_map=self.spatial_map, kernels_map=self.tensor_map,
                                      start_x=int(start_x), start_y=int(start_y), T=self.T, serialize=serialized,
                                      recompute=recmp,
                                      serialize_path=self.serialization_path)

            print(dp_dir)
            # Backtrace from the end point
            walk_ptr = mix_backtrace(
                DP_Matrix=dp_matrix_step,
                T=self.T,
                tensor_map=self.tensor_map,
                terrain=self.spatial_map,
                end_x=int(end_x),
                end_y=int(end_y),
                directory=0,
                serialize=serialized,
                serialize_path=self.serialization_path,
                dp_dir=dp_dir
            )
            segment = get_walk_points(walk_ptr)

            # Cleanup C memory
            if not serialized:
                dll.tensor4D_free(dp_matrix_step, self.T)
            dll.point2d_array_free(walk_ptr)

            # Concatenate paths (skip duplicate point)
            full_path.extend(segment[:-1])

        # Add final point
        full_path.append(steps[-1])
        walk = np.array(full_path)
        plot_combined_terrain(pointer(self.spatial_map), walk, steps=steps, title=self.movebank_study)
