from random_walk_package import create_gaussian_kernel
from random_walk_package.bindings.brownian_walk import *
from random_walk_package.bindings.data_processing.movebank_parser import extract_steps_from_csv
from random_walk_package.bindings.data_structures.point2D import *
from random_walk_package.bindings.data_processing.walk_json import walk_to_json


class BrownianWalker:

    @staticmethod
    def generate_movebank_walk(csv_file, step_count, S, T, width, height, sigma=0.5, scale=0.8, x_offset=0,
                               y_offset=0):
        """
        Static method to generate a Movebank walk from CSV data.
        """
        file = str(os.path.join(script_dir, 'resources', csv_file))
        steps = extract_steps_from_csv(file, step_count, width, height)
        steps_np = get_walk_points(steps)
        kernel = create_gaussian_kernel(width=2 * S + 1, height=2 * S + 1, sigma=sigma, scale=scale, x_offset=x_offset,
                                        y_offset=y_offset)
        walk = brownian_backtrace_multiple_ctype(kernel, steps, T, width, height, None)
        walk_to_json(walk=walk, json_file="multistepwalk.json", steps=steps, terrain_map=None,
                     W=ctypes.c_size_t(width),
                     H=ctypes.c_size_t(height))
        dll.point2d_array_free(steps)
        return [walk, steps_np]

    @classmethod
    def from_parameters(cls, T, S, sigma=0.5, scale=0.8, x_offset=0, y_offset=0, W=None, H=None):
        """
        Creates a BrownianWalker instance using the specified parameters to generate a Gaussian kernel and DP matrix.

        Parameters:
        T (int): Time parameter.
        S (int): Kernel size parameter, the kernel dimensions are (2*S+1 x 2*S+1).
        sigma (float): Sigma for Gaussian kernel.
        scale (float): Scale factor for the kernel.
        x_offset (int): X-offset for the kernel center.
        y_offset (int): Y-offset for the kernel center.
        W (int, optional): Width of the DP matrix. Defaults to 2*T + 1.
        H (int, optional): Height of the DP matrix. Defaults to 2*T + 1.
        """
        if W is None:
            W = 2 * T + 1
        if H is None:
            H = 2 * T + 1
        kernel = create_gaussian_kernel(width=2 * S + 1, height=2 * S + 1, sigma=sigma, scale=scale, x_offset=x_offset,
                                        y_offset=y_offset)
        dp_tensor = brownian_dp_matrix(width=W, height=H, kernel=kernel, time=T)
        return cls(kernel, dp_tensor, T, W, H)

    @classmethod
    def from_terrain_map(cls, terrain, T, S, sigma=0.5, scale=0.8, x_offset=0, y_offset=0, kernel_mapping = None,start_x=None, start_y=None):
        """
        Creates a BrownianWalker instance using the specified parameters to generate a Gaussian kernel and DP matrix.

        Parameters:
        T (int): Time parameter.
        S (int): Kernel size parameter, the kernel dimensions are (2*S+1 x 2*S+1).
        sigma (float): Sigma for Gaussian kernel.
        scale (float): Scale factor for the kernel.
        x_offset (int): X-offset for the kernel center.
        y_offset (int): Y-offset for the kernel center.
        W (int, optional): Width of the DP matrix. Defaults to 2*T + 1.
        H (int, optional): Height of the DP matrix. Defaults to 2*T + 1.
        """

        if start_x is None:
            start_x = terrain.width // 2
        if start_y is None:
            start_y = terrain.height // 2

        if kernel_mapping is None:
            kernel_mapping = create_brownian_kernel_parameters(animal_type=MEDIUM, base_step_size=S)

        kernel = create_gaussian_kernel(width=2 * S + 1, height=2 * S + 1, sigma=sigma, scale=scale, x_offset=x_offset,
                                        y_offset=y_offset)
        kernels_map = get_kernels_map(terrain=terrain, kernel=kernel, mapping=kernel_mapping)
        dp_tensor = brownian_dp_matrix_terrain(kernel=kernel, terrain=terrain, kernels_map=kernels_map, time=T,
                                               start_x=start_x, start_y=start_y)
        return cls(kernel=kernel, dp_tensor=dp_tensor, T=T, W=terrain.width, H=terrain.height, terrain=terrain,
                   kernels_map=kernels_map)

    @classmethod
    def from_kernel_and_tensor(cls, kernel, dp_tensor, T, W, H):
        """
        Creates a BrownianWalker instance using an existing kernel and DP tensor.

        Parameters:
        kernel: Precomputed Gaussian kernel.
        dp_tensor: Precomputed DP tensor.
        T (int): Time parameter used in generating the DP tensor.
        W (int): Width of the DP tensor.
        H (int): Height of the DP tensor.
        """
        return cls(kernel, dp_tensor, T, W, H)

    @classmethod
    def load_tensor(cls, filename, kernel, T, W, H):
        """
        Loads a DP tensor from a file and creates a BrownianWalker instance.

        Parameters:
        filename (str): Path to the saved tensor file.
        kernel: Gaussian kernel corresponding to the DP tensor.
        T (int): Time parameter used in generating the DP tensor.
        W (int): Width of the DP tensor.
        H (int): Height of the DP tensor.
        """
        dp_tensor = dll.tensor_load(ctypes.c_char_p(filename.encode('utf-8')))
        return cls(kernel, dp_tensor, T, W, H)

    def __init__(self, kernel, dp_tensor, T, W, H, terrain=None, kernels_map=None):
        self.kernel = kernel
        self.dp_tensor = dp_tensor
        self.T = T
        self.W = W
        self.H = H
        self.terrain = terrain
        self.kernels_map = kernels_map

    def save_tensor(self, filename):
        """
        Saves the DP tensor to a file.

        Parameters:
        filename (str): Path to save the tensor file.
        """
        dll.tensor_save(self.dp_tensor, ctypes.c_char_p(filename.encode('utf-8')))

    def generate_walk(self, end_x, end_y):
        """
        Generates a walk from the specified endpoint using the DP tensor and kernel.

        Parameters:
        end_x (int): X-coordinate of the endpoint.
        end_y (int): Y-coordinate of the endpoint.

        Returns:
        numpy.ndarray: Array of points representing the walk.
        """
        walk_ptr = brownian_backtrace(dp_matrix=self.dp_tensor, kernel=self.kernel, end_x=end_x, end_y=end_y,
                                      kernels_map=self.kernels_map)
        walk_points = get_walk_points(walk_ptr)
        if self.terrain is not None:
            walk_to_json(walk=walk_ptr, json_file="terrain.json", terrain_map=self.terrain, W=self.W, H=self.H)
        else:
            walk_to_json(walk_ptr, "simple_walk.json", W=self.W, H=self.H)
        return walk_points

    def generate_multistep_walk(self, points):
        """
        Generates a multistep walk passing through the specified points.

        Parameters:
        points (list of tuples): List of (x, y) points the walk should pass through.

        Returns:
        numpy.ndarray: Array of points representing the complete walk path.
        """
        multistep_walk = brownian_backtrace_multiple(kernel=self.kernel, points=points, time=self.T, width=self.W,
                                                     height=self.H, kernels_map=self.kernels_map)
        complete_path = get_walk_points(multistep_walk)
        walk_to_json(walk=multistep_walk, json_file="multistepwalk.json", steps=create_point2d_array(points), W=self.W,
                     H=self.H)
        return complete_path
