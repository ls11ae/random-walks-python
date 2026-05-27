import logging
from typing import Optional, Any, Tuple, List

import numpy as np

from random_walk_package import create_gaussian_kernel, MatrixPtr, KernelParametersMapping, KernelParametersMappingPtr, \
    Reachability
from random_walk_package.bindings.data_structures.kernel_context import kernel_context_pool
from random_walk_package.bindings.data_structures.kernels import kernel_from_array
from random_walk_package.bindings.mixed_walk import mix_backtrace, mix_utilization_distribution, mix_walk
from random_walk_package.bindings.plotter import plot_combined_terrain

logger = logging.getLogger(__name__)


class WalkerHelper:
    """Helper class for terrain-based walk generation operations."""

    @staticmethod
    def validate_point_location(start_x, start_y, W, H):
        if not (0 <= start_x < W and 0 <= start_y < H):
            raise ValueError(f"Start position ({start_x}, {start_y}) out of bounds "
                             f"for grid {W}x{H}")

    @staticmethod
    def validate_parameters(T, W, H, S, D=None) -> None:
        if D is not None and D <= 0:
            raise ValueError(f"Invalid directions: {D}")
        if W <= 0 or H <= 0:
            raise ValueError(f"Invalid grid dimensions: {W}x{H}")
        if T <= 0:
            raise ValueError(f"Invalid time steps: {T}")
        if S <= 0:
            raise ValueError(f"Invalid step size: {S}")

    @staticmethod
    def generate_single_segment(kernel_context,
                                T: int,
                                start_x: int, start_y) -> Any:

        W = kernel_context.contents.terrain.contents.width
        H = kernel_context.contents.terrain.contents.height

        # Validate start position
        if not (0 <= start_x < W and 0 <= start_y < H):
            raise ValueError(f"Start position ({start_x}, {start_y}) out of bounds "
                             f"for terrain {W}x{H}")

        try:

            dp_matrix = mix_walk(
                kernel_context, T, start_x, start_y
            )
            logger.info(f"Successfully generated walk segment from terrain, start=({start_x}, {start_y})")
            return dp_matrix
        except Exception as e:
            logger.error(f"Failed to generate walk segment: {e}")
            raise

    @staticmethod
    def backtrace_single_segment(dp_matrix: Any, T: int, kernel_context,
                                 end_x: int, end_y: int) -> np.ndarray:
        """Backtrace a single walk segment from terrain data.

        Args:
            dp_matrix: DP matrix for the walk
            T: Time steps
            tensor_map: Tensor map for terrain
            terrain: Terrain map
            end_x: End X coordinate
            end_y: End Y coordinate
            kernel_mapping: Kernel mapping parameters
            use_serialization: Whether to use serialized data

        Returns:
            numpy array of walk points
        """
        terrain = kernel_context.contents.terrain
        W = terrain.contents.width
        H = terrain.contents.height

        # Validate end position
        if not (0 <= end_x < W and 0 <= end_y < H):
            raise ValueError(f"End position ({end_x}, {end_y}) out of bounds "
                             f"for grid {W}x{H}")

        try:
            walk_np = mix_backtrace(
                dp_matrix, T, kernel_context, end_x, end_y
            )

            if walk_np is None:
                raise RuntimeError("Backtrace returned null path")

            logger.info(f"Successfully backtraced walk segment to ({end_x}, {end_y})")
            return np.array(walk_np)

        except Exception as e:
            logger.error(f"Failed to backtrace walk segment: {e}")
            raise

    @staticmethod
    def generate_utilization_distribution(DP_Matrix: Any,
                                          T: Any,
                                          kernel_context: Any,
                                          end_x: Any,
                                          end_y: Any):

        return mix_utilization_distribution(
            DP_Matrix, T, kernel_context, end_x, end_y
        )

    @staticmethod
    def generate_multistep_walk(steps: List[Tuple[int, int]], T: int,
                                kernel_context, plot: bool = False,
                                plot_title: str = "Correlated Walk on terrain with multiple steps") -> np.ndarray:
        if len(steps) < 2:
            raise ValueError("At least two steps are required for multistep walk")

        full_path = np.empty((0, 2))
        terrain = kernel_context.contents.terrain

        for i in range(len(steps) - 1):
            print(f"iteration {i} of {len(steps) - 1}", end="\r")
            start_x, start_y = steps[i]
            end_x, end_y = steps[i + 1]

            # Generate segment
            dp_matrix = WalkerHelper.generate_single_segment(
                kernel_context, T, start_x, start_y
            )

            # Backtrace segment
            segment = WalkerHelper.backtrace_single_segment(
                dp_matrix, T, terrain, end_x, end_y
            )
            full_path = np.vstack((full_path, segment[:-1]))

        if plot:
            plot_combined_terrain(terrain, full_path, steps=steps, title=plot_title)

        return full_path

    @staticmethod
    def set_custom_kernel(base_kernel: Optional[np.ndarray] = None, S: int = None) -> MatrixPtr:
        kernel_width, kernel_height = base_kernel.shape if base_kernel is not None else (2 * S + 1, 2 * S + 1)
        try:
            if base_kernel is not None:
                if kernel_width != 2 * S + 1 or kernel_height != 2 * S + 1:
                    raise ValueError(
                        "Custom kernel must have dimensions 2S+1x2S+1. Stepsize and passed Array are contradictory")
                return kernel_from_array(base_kernel, kernel_width, kernel_height)
            else:
                return create_gaussian_kernel(kernel_width, kernel_height, sigma=S / 2.0, scale=1, x_offset=0,
                                              y_offset=0)

        except Exception as e:
            logger.error(f"Failed to set kernel: {e}")
            raise
