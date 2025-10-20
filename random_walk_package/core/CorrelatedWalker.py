import hashlib
import weakref

from random_walk_package import tensor_free, tensor4D_free
from random_walk_package.bindings.correlated_walk import *
from random_walk_package.bindings.data_processing.utils import use_low_ram
from random_walk_package.bindings.data_structures.kernel_terrain_mapping import kernel_mapping_free, \
    create_correlated_kernel_parameters
from random_walk_package.bindings.data_structures.kernels import correlated_kernels_from_matrix, \
    generate_correlated_kernels
from random_walk_package.bindings.plotter import plot_walk, plot_walk_multistep
from random_walk_package.core.WalkerHelper import *

logger = logging.getLogger(__name__)


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
        self.kernels = None
        if kernel is not None:
            self.kernels = kernel
        else:
            self.set_kernel()
        self.dp_matrix = dp_mat
        self.terrain = terrain
        self.kernel_mapping = kernel_mapping
        self.tensor_map = tensor_map

        # State tracking
        self._is_initialized = False  # is the DP matrix initialized?
        self._using_terrain = terrain is not None

        self._setup_initial_configuration()
        self._finalizer = weakref.finalize(self, self._cleanup_resources)

    def _setup_initial_configuration(self) -> None:
        """Set up the initial configuration based on provided parameters."""
        if self.terrain is not None:
            if self.kernel_mapping is None:
                self.kernel_mapping = create_correlated_kernel_parameters(MEDIUM, self.S)
            try:
                if self.tensor_map is None:
                    self.tensor_map = get_tensor_map_terrain(self.terrain, mapping=self.kernel_mapping)
                    self._is_initialized = True
            except Exception as e:
                logger.error(f"Failed to create tensor map: {e}")
                self._is_initialized = False
            self.W = self.terrain.contents.width
            self.H = self.terrain.contents.height

    def _validate_parameters(self) -> None:
        """Validate that all required parameters are set."""
        WalkerHelper.validate_parameters(self.T, self.W, self.H, self.S, self.D)

    def _cleanup_resources(self) -> None:
        """Clean up all C resources."""
        resources = ["kernels", "dp_matrix", "kernel_mapping", "tensor_map", "terrain"]
        for resource in resources:
            if getattr(self, resource, None) is not None:
                try:
                    if resource == "tensor_map":
                        kernels_map3d_free(self.tensor_map)
                    if resource == "kernels":
                        tensor_free(self.kernels)
                    elif resource == "dp_matrix":
                        tensor4D_free(self.dp_matrix, self.T)
                    elif resource == "kernel_mapping":
                        kernel_mapping_free(self.kernel_mapping)
                    elif resource == "terrain":
                        terrain_map_free(self.terrain)
                    setattr(self, resource, None)
                except Exception as e:
                    logger.warning(f"Failed to free {resource}: {e}")

    @property
    def is_ready_for_walk(self) -> bool:
        """Check if the walker is ready for initialization."""
        if self._using_terrain:
            return all([self.terrain, self.tensor_map, self.kernel_mapping])
        else:
            return self.kernels is not None

    @property
    def is_ready_for_backtrace(self) -> bool:
        """Check if walker is ready for the generation of a backtrace (walk array)."""
        return (self.dp_matrix is not None) and self.is_ready_for_walk

    def set_kernel(self, kernel_np: Optional[np.ndarray] = None,
                   d: int = None, S: Optional[int] = None) -> None:
        """Set the kernel for walk generation.
            kernel_np: Custom kernel as numpy array
            S: Step size (uses existing if None)
            d: Number of directions
        """
        self.S = S if S is not None else self.S
        kernel_width, kernel_height = kernel_np.shape if kernel_np is not None else (2 * self.S + 1, 2 * self.S + 1)
        self.S = kernel_width // 2
        self.D = d if d is not None else self.D

        # Clean up the existing kernel
        if self.kernels is not None:
            tensor_free(self.kernels)
            self.kernels = None

        try:
            if kernel_np is not None:
                if kernel_width != 2 * self.S + 1 or kernel_height != 2 * self.S + 1:
                    raise ValueError(
                        "Custom kernel must have dimensions 2S+1x2S+1. Stepsize and passed Array are contradictory")
                tensor_c = correlated_kernels_from_matrix(kernel_np, kernel_width, kernel_height, self.D)
                self.kernels = tensor_c
            else:
                self.kernels = generate_correlated_kernels(kernel_width, self.D)

            logger.info(f"Successfully set kernel with size {kernel_width}x{kernel_width}x{self.D}")

        except Exception as e:
            logger.error(f"Failed to set kernel: {e}")
            raise

    def generate(self, start_x=None, start_y=None, use_serialization=False):
        if self.kernels is None:
            self.set_kernel()
        if start_x is None:
            start_x = self.W // 2
        if start_y is None:
            start_y = self.H // 2

        self._validate_parameters()
        # Validate start position
        if not (0 <= start_x < self.W and 0 <= start_y < self.H):
            raise ValueError(f"Start position ({start_x}, {start_y}) out of bounds "
                             f"for grid {self.W}x{self.H}")
        if self.dp_matrix is not None:
            tensor4D_free(self.dp_matrix, self.T)

        use_serialization = use_serialization or use_low_ram(self.D, self.W, self.H, self.T, False)
        dp_folder = _dp_folder_name(self.D, self.W, self.H, self.T, start_x, start_x, False)
        dp_mat = correlated_walk_init(kernel=self.kernels,
                                      width=self.W,
                                      height=self.H,
                                      time=self.T,
                                      start_x=start_x,
                                      start_y=start_y,
                                      use_serialization=use_serialization,
                                      output_folder=dp_folder)
        # save path to serialized dp matrix
        self.dp_matrix = None if use_serialization else dp_mat
        print(f"DP matrix serialized to: {dp_folder}")
        return dp_folder

    def backtrace(self, end_x, end_y, dp_folder=None, initial_direction=0, plot=False):
        walk = correlated_backtrace(dp_mat=self.dp_matrix if dp_folder is None else None,
                                    T=self.T, kernels=self.kernels,
                                    end_x=end_x, end_y=end_y,
                                    direction=initial_direction,
                                    use_serialization=dp_folder is not None,
                                    dp_folder=dp_folder)
        if plot:
            plot_walk(walk, self.W, self.H, title="Correlated Walk")
        return walk

    def multistep_walk(self, steps, direction=0, use_serialization=False, dp_folder=None,
                       plot=False) -> np.ndarray:
        """Generate multistep walk

        Args:
            steps: Array of step points
            direction: Initial direction
            use_serialization: Whether to use serialized data
            dp_folder: Folder to save serialized data
            plot: Whether to plot the walk or not
        Returns:
            Full path as a numpy array
        """
        if not self.is_ready_for_walk:
            raise ValueError("Walker not properly initialized for multistep walk")
        try:
            if use_serialization and dp_folder is None:
                dp_folder = _dp_folder_name(self.D, self.W, self.H, self.T, steps[0][0], steps[0][1], False)
            result = correlated_multi_step(W=self.W, H=self.H, T=self.T, kernels=self.kernels, steps=steps,
                                           direction=direction,
                                           use_serialization=use_serialization, dp_folder=dp_folder)
        except Exception as e:
            logger.error(f"Failed to generate multistep walk: {e}")
            raise
        if plot:
            plot_walk_multistep(steps, result, self.W, self.H)
        return result

    ####################################################################################################################
    ##################################### TERRAIN DEPENDANT BROWNIAN WALKS #############################################
    ####################################################################################################################

    def generate_from_terrain(self, terrain: Optional[Any] = None,
                              start_x: Optional[int] = None,
                              start_y: Optional[int] = None,
                              use_serialization=False) -> None:
        """Generate walk from terrain data.

        Args:
            terrain: Terrain map (uses existing if None)
            start_x: Starting X coordinate (center if None)
            start_y: Starting Y coordinate (center if None)
            use_serialization: Whether to use serialized data
        """
        if terrain is not None:
            self.terrain = terrain
            self._using_terrain = True

        if self.terrain is None:
            raise ValueError("No terrain provided and no existing terrain set")

        # Calculate the default start position
        if start_x is None:
            start_x = self.terrain.contents.width // 2
        if start_y is None:
            start_y = self.terrain.contents.height // 2

        # Initialize kernel mapping if needed
        if self.kernel_mapping is None:
            self.kernel_mapping = create_correlated_kernel_parameters(MEDIUM, self.S)

        # Create a tensor map if needed
        if self.tensor_map is None:
            self.tensor_map = get_tensor_map_terrain(self.terrain, mapping=self.kernel_mapping)

        self.W = self.terrain.contents.width
        self.H = self.terrain.contents.height

        self._validate_parameters()
        dp_folder = None
        if use_serialization:
            dp_folder = _dp_folder_name(self.D, self.W, self.H, self.T, start_x, start_y, True)

        # Clean up previous DP matrix
        if self.dp_matrix is not None:
            tensor4D_free(self.dp_matrix, self.T)
            self.dp_matrix = None

        try:
            # Use WalkerHelper for the core generation logic
            self.dp_matrix = WalkerHelper.generate_single_segment(
                self.terrain, start_x, start_y, self.T, self.kernel_mapping,
                self.tensor_map, use_serialization, dp_folder
            )
            self._is_initialized = True
            logger.info(f"Successfully generated walk from terrain, start=({start_x}, {start_y})")
        except Exception as e:
            self._is_initialized = False
            logger.error(f"Failed to generate walk from terrain: {e}")
            raise

    def backtrace_from_terrain(self, end_x: int, end_y: int,
                               terrain: Optional[Any] = None,
                               plot: bool = False,
                               plot_title: str = "Terrain Walk",
                               use_serialization=False,
                               dp_folder=None) -> np.ndarray:
        """Backtrace walk from terrain data.

        Args:
            end_x: End X coordinate
            end_y: End Y coordinate
            terrain: Terrain map (uses existing if None)
            plot: Whether to plot the walk
            plot_title: Title of the plot
            use_serialization: Whether to use serialized data
            dp_folder: Folder to save serialized data

        Returns:
            numpy array of walk points
        """
        if not self.is_ready_for_backtrace:
            raise ValueError('Initialize walk first before calling backtrace. Call generate_from_terrain first.')

        if terrain is not None:
            self.terrain = terrain
            self.tensor_map = get_tensor_map_terrain(self.terrain, mapping=self.kernel_mapping)

        try:
            # Use WalkerHelper for the core backtrace logic
            walk_np = WalkerHelper.backtrace_single_segment(
                self.dp_matrix, self.T, self.tensor_map, self.terrain,
                end_x, end_y, self.kernel_mapping, use_serialization
            )

            logger.info(f"Successfully backtraced walk to ({end_x}, {end_y})")
            if plot:
                plot_combined_terrain(self.terrain, terrain_width=self.W, terrain_height=self.H,
                                      walk_points=walk_np, steps=None, title=plot_title)
            return walk_np

        except Exception as e:
            logger.error(f"Failed to backtrace walk: {e}")
            raise

    def generate_from_terrain_multistep(self, terrain: Optional[Any] = None,
                                        steps: List[Tuple[int, int]] = None,
                                        plot=False,
                                        plot_title="Correlated Walk on terrain with multiple steps") -> np.ndarray:
        """Generate a multistep walk from terrain data.

        Args:
            terrain: Terrain map (uses existing if None)
            steps: List of step points
            plot: Whether to plot the walk
            plot_title: Title of the plot

        Returns:
            list of walk point tuples
        """
        if terrain is not None:
            self.terrain = terrain

        if steps is None:
            raise ValueError("Steps list cannot be None")

        # Initialize kernel mapping and tensor map if needed
        if self.kernel_mapping is None:
            self.kernel_mapping = create_correlated_kernel_parameters(MEDIUM, self.S)
        if self.tensor_map is None:
            self.tensor_map = get_tensor_map_terrain(self.terrain, mapping=self.kernel_mapping)

        # Use WalkerHelper for the multistep generation
        full_path = WalkerHelper.generate_multistep_walk(
            self.terrain, steps, self.T, self.kernel_mapping, self.tensor_map,
            plot, plot_title
        )
        return full_path

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self._cleanup_resources()

    def __del__(self):
        """Destructor as backup for cleanup."""
        self._cleanup_resources()
