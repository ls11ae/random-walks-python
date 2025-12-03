import typing
import typing
import weakref

from random_walk_package import tensor_free, tensor4D_free, matrix_free
from random_walk_package.bindings import get_tensor_map_terrain, MEDIUM, terrain_map_free, kernels_map3d_free
from random_walk_package.bindings.brownian_walk import *
from random_walk_package.bindings.data_structures.kernel_terrain_mapping import create_brownian_kernel_parameters, \
    kernel_mapping_free
from random_walk_package.bindings.plotter import plot_walk, plot_walk_multistep
from random_walk_package.core.WalkerHelper import *

logger = logging.getLogger(__name__)


class BrownianWalker:
    def __init__(self, S: int = 3, W: int = 101, H: int = 101, T: int = 50,
                 kernel: Optional[Any] = None, terrain: Optional[Any] = None,
                 k_mapping: Optional[Any] = None):
        """
            Brownian Walker class.

            Args:
                S (int): Stepsize, meaning the distance that can be moved in the grid in one step.
                W (int): Width of the grid.
                H (int): Height of the grid.
                T (int): Number of time steps.
                kernel: Optional Kernel which defines transition probabilities between grid cells, if not provided, a standard bivariate Gaussian kernel is used.
                terrain: Optional terrain data, which, with k_mapping, defines kernels depending on terrain value (MESA landcover classes).
                k_mapping: Optional Kernel-parameters-mapping, which defines the kernel parameters for each terrain class.
        """
        self.S = S
        self.W = W if terrain is None else terrain.contents.width
        self.H = H if terrain is None else terrain.contents.height
        self.T = T
        self.kernels = kernel
        self.terrain = terrain
        self.dp_matrix = None
        self.dp_matrix_terrain = None
        self.kernel_mapping = k_mapping
        self.tensor_map = None

        # State tracking
        self._is_initialized = False  # is the DP matrix initialized?
        self._using_terrain = terrain is not None

        self._setup_initial_configuration()
        self._finalizer = weakref.finalize(self, self._cleanup_resources)

    def _setup_initial_configuration(self) -> None:
        """Set up the initial configuration based on provided parameters."""
        if self.terrain is not None:
            if self.kernel_mapping is None:
                self.kernel_mapping = create_brownian_kernel_parameters(MEDIUM, self.S)
            self.W = self.terrain.contents.width
            self.H = self.terrain.contents.height
            try:
                self.tensor_map = get_tensor_map_terrain(self.terrain, self.kernel_mapping)
                self._is_initialized = True
            except Exception as e:
                logger.error(f"Failed to create tensor map: {e}")
                self._is_initialized = False

    def _validate_parameters(self) -> None:
        """Validate that all required parameters are set."""
        WalkerHelper.validate_parameters(self.T, self.W, self.H, self.S)

    def _cleanup_resources(self) -> None:
        """Clean up all C resources."""
        resources = ["kernels", "dp_matrix", "kernel_mapping", "tensor_map", "terrain"]
        for resource in resources:
            if getattr(self, resource, None) is not None:
                try:
                    if resource == "tensor_map":
                        kernels_map3d_free(self.tensor_map)
                    if resource == "kernels":
                        matrix_free(self.kernels)
                    elif resource == "dp_matrix":
                        tensor_free(self.dp_matrix)
                    elif resource == "dp_matrix_terrain":
                        tensor4D_free(self.dp_matrix_terrain, self.T)
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
        return (self.dp_matrix is not None or self.dp_matrix_terrain is not None) and self.is_ready_for_walk

    def set_kernel(self, kernel_np: Optional[np.ndarray] = None,
                   sigma: float = 3.0, S: Optional[int] = None) -> None:
        """Set the kernel for walk generation.

        Args:
            kernel_np: Custom kernel as numpy array
            sigma: Sigma for Gaussian kernel
            S: Step size (uses existing if None)
        """
        self.S = S if S is not None else self.S
        # Clean up the existing kernel
        if self.kernels is not None:
            matrix_free(self.kernels)
            self.kernels = None

        self.kernels = WalkerHelper.set_custom_kernel(kernel_np, self.S)

    def generate(self, start_x: Optional[int] = None, start_y: Optional[int] = None) -> None:
        """Generate walk using a simple kernel approach."""
        if self.kernels is None:
            self.set_kernel()

        if start_x is None:
            start_x = self.W // 2
        if start_y is None:
            start_y = self.H // 2

        self._validate_parameters()

        # Validate start position
        WalkerHelper.validate_point_location(start_x, start_y, self.W, self.H)

        # Clean up previous DP matrix
        if self.dp_matrix is not None:
            tensor_free(self.dp_matrix)

        try:
            self.dp_matrix = brownian_walk_init(self.kernels, self.W, self.H, self.T, start_x, start_y)
            self._is_initialized = True
            self._using_terrain = False
            logger.info(f"Successfully generated walk, start=({start_x}, {start_y})")
        except Exception as e:
            self._is_initialized = False
            logger.error(f"Failed to generate walk: {e}")
            raise

    def backtrace(self, end_x: int, end_y: int, plot=False, plot_title="Einfacher Gauß-Walk") -> np.ndarray:
        """Backtrace walk using a simple kernel approach.

        Args:
            end_x: End X coordinate
            end_y: End Y coordinate
            plot: Whether to plot the walk
            plot_title: Title of the plot

        Returns:
            numpy array of walk points
        """
        if not self.is_ready_for_backtrace:
            raise ValueError('Initialize walk first before calling backtrace. Call generate first.')

        # Validate end position
        WalkerHelper.validate_point_location(end_x, end_y, self.W, self.H)

        try:
            walk_np = brownian_backtrace(self.dp_matrix, self.kernels, end_x, end_y)
            if walk_np is None:
                raise RuntimeError("Backtrace returned null path")

            logger.info(f"Successfully backtraced walk to ({end_x}, {end_y})")
            if plot:
                plot_walk(walk_np, self.W, self.H, plot_title)
            return walk_np

        except Exception as e:
            logger.error(f"Failed to backtrace walk: {e}")
            raise

    def multistep_walk(self, steps: typing.Union[np.ndarray, list[tuple[int, int]]], plot=False) -> np.ndarray:
        """Generate multistep walk

        Args:
            steps: Array of step points
            plot: Whether to plot the walk

        Returns:
            Full path as numpy array
        """
        if not self.is_ready_for_walk:
            raise ValueError("Walker not properly initialized for multistep walk")
        try:
            result = brownian_backtrace_multiple(self.kernels, steps, self.T, self.W, self.H)
            print("dkfj")
        except Exception as e:
            logger.error(f"Failed to generate multistep walk: {e}")
            raise
        if plot:
            plot_walk_multistep(steps, result, self.W, self.H)
        return result

    ####################################################################################################################
    ##################################### TERRAIN DEPENDANT BROWNIAN WALKS #############################################
    ####################################################################################################################

    def generate_with_terrain(self, terrain: Optional[Any] = None,
                              start_x: Optional[int] = None,
                              start_y: Optional[int] = None) -> None:
        """Generate walk from terrain data.

        Args:
            terrain: Terrain map (uses existing if None)
            start_x: Starting X coordinate (center if None)
            start_y: Starting Y coordinate (center if None)
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

        # Validate start position
        if not (0 <= start_x < self.terrain.contents.width and 0 <= start_y < self.terrain.contents.height):
            raise ValueError(f"Start position ({start_x}, {start_y}) out of bounds "
                             f"for terrain {self.terrain.contents.width}x{self.terrain.contents.height}")

        # Initialize kernel mapping if needed
        if self.kernel_mapping is None:
            self.kernel_mapping = create_brownian_kernel_parameters(MEDIUM, self.S)

        # Create a tensor map if needed
        if self.tensor_map is None:
            self.tensor_map = get_tensor_map_terrain(self.terrain, mapping=self.kernel_mapping)

        self.W = self.terrain.contents.width
        self.H = self.terrain.contents.height

        self._validate_parameters()

        # Clean up previous DP matrix
        if self.dp_matrix_terrain is not None:
            tensor4D_free(self.dp_matrix_terrain, self.T)
            self.dp_matrix_terrain = None

        try:
            # Use WalkerHelper for the core generation logic
            self.dp_matrix = WalkerHelper.generate_single_segment(
                self.terrain, start_x, start_y, self.T, self.kernel_mapping,
                self.tensor_map, False, ""
            )
            self._is_initialized = True
            logger.info(f"Successfully generated walk from terrain, start=({start_x}, {start_y})")
        except Exception as e:
            self._is_initialized = False
            logger.error(f"Failed to generate walk from terrain: {e}")
            raise

    def backtrace_terrain(self, end_x: int, end_y: int,
                          terrain: Optional[Any] = None,
                          plot: bool = False,
                          plot_title: str = "Terrain Walk") -> np.ndarray:
        """Backtrace walk from terrain data.

        Args:
            end_x: End X coordinate
            end_y: End Y coordinate
            terrain: Terrain map (uses existing if None)
            plot: Whether to plot the walk
            plot_title: Title of the plot

        Returns:
            numpy array of walk points

        """
        if not self.is_ready_for_backtrace:
            raise ValueError('Initialize walk first before calling backtrace. Call generate_from_terrain first.')

        if terrain is not None:
            self.terrain = terrain
            self.tensor_map = get_tensor_map_terrain(self.terrain, mapping=self.kernel_mapping)

        try:
            walk_np = WalkerHelper.backtrace_single_segment(
                self.dp_matrix, self.T, self.tensor_map, self.terrain,
                end_x, end_y, self.kernel_mapping, False
            )

            logger.info(f"Successfully backtraced walk to ({end_x}, {end_y})")
            if plot:
                plot_combined_terrain(self.terrain, terrain_width=self.W, terrain_height=self.H,
                                      walk_points=walk_np, steps=None, title=plot_title)
            return walk_np

        except Exception as e:
            logger.error(f"Failed to backtrace walk: {e}")
            raise

    def terrain_multistep_walk(self, terrain: Optional[Any] = None,
                               steps: typing.Union[np.ndarray, list[tuple[int, int]]] = None,
                               plot=False, plot_title="Brownian Walk on terrain") -> np.ndarray:
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
            self.kernel_mapping = create_brownian_kernel_parameters(MEDIUM, self.S)
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
