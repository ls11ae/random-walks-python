from random_walk_package import get_walk_points, create_point2d_array
from random_walk_package.bindings.correlated_walk import *
from random_walk_package.bindings.mixed_walk import time_walk_init, time_walk_backtrace


class BiasedWalker:
    def __init__(self, terrain=None, bias_array=None):
        self.terrain = terrain
        self.T = len(bias_array)
        self.bias_array = create_point2d_array(bias_array)
        self.tensor_map = tensor_map_terrain_biased(self.terrain, self.bias_array)
        self.W = terrain.width
        self.H = terrain.height
        self.dp_matrix = None

    def generate(self, start_x=None, start_y=None):
        if start_x is None or start_y is None:
            start_x, start_y = self.W // 2, self.H // 2
        self.dp_matrix = time_walk_init(self.W, self.H, self.terrain, self.tensor_map, self.T, start_x, start_y)

    def backtrace(self, end_x, end_y):
        walk = time_walk_backtrace(self.dp_matrix, self.T, self.terrain, self.tensor_map, end_x, end_y, 0)
        walk_np = get_walk_points(walk)
        dll.point2d_array_free(walk)
        return walk_np

    def generate_multistep_walk(self, steps):
        full_path = []
        for i in range(len(steps) - 1):
            start_x, start_y = steps[i]
            end_x, end_y = steps[i + 1]
            self.generate(start_x, start_y)
            segment = self.backtrace(end_x, end_y)
            dll.tensor4D_free(self.dp_matrix, self.T)
            full_path.extend(segment[:-1])

        full_path.append(steps[-1])
        return np.array(full_path)
