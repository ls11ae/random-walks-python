from random_walk_package.bindings.brownian_walk import plot_combined_terrain

from random_walk_package.bindings.data_structures.point2D import get_walk_points

from random_walk_package.bindings.data_processing.movebank_parser import extract_steps_from_csv

from random_walk_package.bindings.data_processing.walk_json import walk_to_json

from random_walk_package.bindings.correlated_walk import *
from random_walk_package.bindings.mixed_walk import *
from random_walk_package.core.AnimalMovement import AnimalMovementProcessor
import numpy as np


class MixedWalker:
    def __init__(self, T=30, D=4, S=3, tensor_list=None, tensor_set=None, spatial_map=None, brownian_kernel=None,
                 correlated_kernel=None,
                 width=None,
                 height=None, kernels_map=None, tensor_map=None, movebank_study=None):
        self.T = T
        self.D = D  # Number of directions
        self.S = S  # Step size
        self.spatial_map = spatial_map
        self.brownian_kernel = brownian_kernel
        self.correlated_kernel = correlated_kernel
        self.width = width
        self.height = height
        self.kernels_map = kernels_map
        self.tensor_map = tensor_map
        self.tensor_list = tensor_list
        self.tensor_set = tensor_set
        self.movebank_study = movebank_study
        self.movebank_processor = None

    def set_kernels(self, terrain_only = True):
        self.movebank_processor = AnimalMovementProcessor(self.movebank_study)
        self.spatial_map = self.movebank_processor.create_landcover_data(self.width)
        if terrain_only:
            self.tensor_map = get_tensor_map_terrain(self.spatial_map)

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

    def generate_walk(self, steps=None):

        full_path = []
        width = self.spatial_map.contents.width
        height = self.spatial_map.contents.height
        if steps is None:
            steps = self.movebank_processor.create_movement_data(width=width, height=height, samples=10)

        # TODO: (very) cheap fix here ... add padding to bbox
        steps = [(400, 400), (1600, 1340)]

        for i in range(len(steps) - 1):
            start_x, start_y = steps[i]
            end_x, end_y = steps[i + 1]
            print(start_x, start_y, end_x, end_y)

            # Initialize DP matrix for the current start point
            dp_matrix_step = mix_walk(W=width, H=height, terrain_map=self.spatial_map, kernels_map=self.tensor_map,
                                      start_x=int(start_x), start_y=int(start_y), T=self.T)

            # Backtrace from the end point
            walk_ptr = mix_backtrace(
                DP_Matrix=dp_matrix_step,
                T=self.T,
                tensor_map=self.tensor_map,
                terrain=self.spatial_map,
                end_x=int(end_x),
                end_y=int(end_y),
                dir=0
            )
            segment = get_walk_points(walk_ptr)

            # Cleanup C memory
            dll.tensor4D_free(dp_matrix_step, self.T)
            dll.point2d_array_free(walk_ptr)

            # Concatenate paths (skip duplicate point)
            full_path.extend(segment[:-1])

        # Add final point
        full_path.append(steps[-1])
        walk = np.array(full_path)
        plot_combined_terrain(self.spatial_map, walk, steps=steps, title=self.movebank_study)
