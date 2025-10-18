import hashlib

from random_walk_package.bindings.brownian_walk import *
from random_walk_package.bindings.correlated_walk import *
from random_walk_package.bindings.data_processing.utils import use_low_ram
from random_walk_package.bindings.data_structures.point2D import get_walk_points


def _dp_folder_name(D, W, H, T, start_x, start_y, is_terrain):
    key = f"{D}-{W}-{H}-{T}-{start_x}-{start_y}-{int(is_terrain)}"
    h = hashlib.sha1(key.encode()).hexdigest()[:8]
    folder = os.path.join("/tmp", f"dp_{h}")
    os.makedirs(folder, exist_ok=True)
    return folder


class CorrelatedWalker:
    def __init__(self, D=4, S=3, W=101, H=101, T=50, kernel=None, dp_mat=None, terrain=None, kernel_mapping=None,
                 tensor_map=None):
        self.D = D  # Number of directions
        self.S = S  # Step size
        self.W = W  # Width of the grid
        self.H = H  # Height of the grid
        self.T = T  # Number of time steps
        self.kernels = kernel
        self.dp_matrix = dp_mat
        self.terrain = terrain
        self.kernel_mapping = kernel_mapping
        self.tensor_map = tensor_map

    def generate_from_terrain(self, terrain=None, kernel_mapping=None, start_x=None, start_y=None):
        if start_x is None:
            start_x = terrain.width // 2
        if start_y is None:
            start_y = terrain.height // 2

        if self.terrain is None:
            self.terrain = terrain

        self.kernel_mapping = create_correlated_kernel_parameters(MEDIUM,
                                                                  7) if self.kernel_mapping is None else kernel_mapping
        if self.tensor_map is None:
            self.tensor_map = get_tensor_map_terrain(terrain, mapping=self.kernel_mapping)
        self.W = terrain.width
        self.H = terrain.height

        if self.kernels is None:
            self.generate_kernel()

        if use_low_ram(self.D, self.W, self.H, self.T, False):
            dp_folder = "tensor"
            dp_calculation_terrain_low_ram(self.W, self.H, self.kernels, self.terrain, self.tensor_map,
                                           self.T, start_x, start_y, dp_folder)
            self.dp_matrix = dp_folder
        else:
            self.dp_matrix = correlated_dp_matrix_terrain(self.W, self.H, self.kernels, self.terrain, self.tensor_map,
                                                          self.T, start_x,
                                                          start_y)

    def generate(self, start_x=None, start_y=None):
        self.kernels = create_correlated_chi_kernels(self.D, self.S)
        if start_x is None or start_y is None:
            start_x, start_y = self.W // 2, self.H // 2
        dp_folder = "tensor"
        """_dp_folder_name(self.D, self.W, self.H, self.T, start_x, start_x, False)"""
        if use_low_ram(self.D, self.W, self.H, self.T, False):
            dp_calculation_low_ram(self.W, self.H, self.kernels,
                                   self.T, start_x, start_y, dp_folder)
            self.dp_matrix = dp_folder
        else:
            self.dp_matrix = correlated_dp_matrix(kernel=self.kernels,
                                                  width=self.W,
                                                  height=self.H,
                                                  time=self.T,
                                                  start_x=start_x,
                                                  start_y=start_y)

    def generate_kernel(self):
        self.kernels = create_correlated_chi_kernels(self.D, self.S)

    def backtrace(self, end_x, end_y, initial_direction=0):
        is_terrain = (self.terrain is not None)
        dp_folder = "tensor"
        """_dp_folder_name(self.D, self.W, self.H, self.T,
                                    getattr(self, 'start_x', None),
                                    getattr(self, 'start_y', None),
                                    is_terrain)"""

        if use_low_ram(self.D, self.W, self.H, self.T, is_terrain):
            walk = backtrace_low_ram(dp_folder,
                                     self.T,
                                     self.kernels,
                                     self.tensor_map if is_terrain else None,
                                     end_x, end_y,
                                     initial_direction,
                                     self.D)
        else:
            walk = correlated_backtrace(
                dp_mat=self.dp_matrix,
                T=self.T,
                kernels=self.kernels,
                terrain=self.terrain,
                tensor_map=self.tensor_map if is_terrain else None,
                end_x=end_x,
                end_y=end_y,
                direction=initial_direction,
                directions=self.D
            )
        walk_np = get_walk_points(walk)
        return walk_np

    def generate_multistep_walk(self, steps):
        """Python-implemented multistep backtrace with low RAM support"""
        dp_matrix_step = None
        if self.kernels is None:
            self.generate()

        is_terrain = self.terrain is not None
        low_ram = use_low_ram(self.D, self.W, self.H, self.T, is_terrain)

        full_path = []
        for i in range(len(steps) - 1):
            start_x, start_y = steps[i]
            end_x, end_y = steps[i + 1]

            if low_ram:
                dp_folder = f"tensor_seg_{i}"
                if is_terrain:
                    dp_calculation_terrain_low_ram(
                        self.W, self.H, self.kernels, self.terrain, self.tensor_map,
                        self.T, start_x, start_y, dp_folder
                    )
                else:
                    dp_calculation_low_ram(
                        self.W, self.H, self.kernels, self.T, start_x, start_y, dp_folder
                    )
                walk_ptr = backtrace_low_ram(
                    dp_folder,
                    self.T,
                    self.kernels,
                    self.tensor_map if is_terrain else None,
                    end_x, end_y,
                    direction=0,
                    directions=self.D
                )
            else:
                dp_matrix_step = correlated_dp_matrix(
                    kernel=self.kernels,
                    width=self.W,
                    height=self.H,
                    time=self.T,
                    start_x=start_x,
                    start_y=start_y
                )
                walk_ptr = correlated_backtrace(
                    dp_mat=dp_matrix_step,
                    T=self.T,
                    kernels=self.kernels,
                    terrain=self.terrain,
                    tensor_map=self.tensor_map if is_terrain else None,
                    end_x=end_x,
                    end_y=end_y,
                    direction=0,
                    directions=self.D
                )

            segment = get_walk_points(walk_ptr)

            if low_ram:
                pass
            else:
                dll.tensor4D_free(dp_matrix_step, self.T)
            dll.point2d_array_free(walk_ptr)

            full_path.extend(segment[:-1])

        full_path.append(steps[-1])
        return np.array(full_path)
