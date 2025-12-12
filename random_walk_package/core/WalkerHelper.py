import logging
from typing import Optional, Any, Tuple, List

import numpy as np
import pandas as pd

from random_walk_package import create_gaussian_kernel, MatrixPtr
from random_walk_package.bindings.data_structures.kernels import kernel_from_array
from random_walk_package.bindings.mixed_walk import mix_backtrace, mix_walk
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
    def generate_single_segment(terrain: Any, start_x: int, start_y: int, T: int,
                                kernel_mapping: Any, tensor_map: Any, use_serialization: bool = False,
                                dp_folder: Optional[str] = None) -> Any:
        """Generate a single walk segment from terrain data.

        Args:
            terrain: Terrain map
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            T: Time steps
            kernel_mapping: Kernel mapping parameters
            tensor_map: Tensor map for terrain
            use_serialization: Whether to use serialized data
            dp_folder: Folder for serialized data

        Returns:
            DP matrix for the walk segment
        """
        W = terrain.contents.width
        H = terrain.contents.height

        # Validate start position
        if not (0 <= start_x < W and 0 <= start_y < H):
            raise ValueError(f"Start position ({start_x}, {start_y}) out of bounds "
                             f"for terrain {W}x{H}")

        try:
            dp_matrix = mix_walk(
                W, H, terrain, tensor_map, T,
                start_x, start_y, use_serialization, True, dp_folder, kernel_mapping
            )
            logger.info(f"Successfully generated walk segment from terrain, start=({start_x}, {start_y})")
            return dp_matrix
        except Exception as e:
            logger.error(f"Failed to generate walk segment: {e}")
            raise

    @staticmethod
    def backtrace_single_segment(dp_matrix: Any, T: int, tensor_map: Any, terrain: Any,
                                 end_x: int, end_y: int, kernel_mapping: Any,
                                 use_serialization: bool = False) -> np.ndarray:
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
        W = terrain.contents.width
        H = terrain.contents.height

        # Validate end position
        if not (0 <= end_x < W and 0 <= end_y < H):
            raise ValueError(f"End position ({end_x}, {end_y}) out of bounds "
                             f"for grid {W}x{H}")

        try:
            walk_np = mix_backtrace(
                dp_matrix, T, tensor_map, terrain,
                end_x, end_y, use_serialization, "", "", kernel_mapping
            )

            if walk_np is None:
                raise RuntimeError("Backtrace returned null path")

            logger.info(f"Successfully backtraced walk segment to ({end_x}, {end_y})")
            return np.array(walk_np)

        except Exception as e:
            logger.error(f"Failed to backtrace walk segment: {e}")
            raise

    @staticmethod
    def interpolate_timestamps(start_t, end_t, n_points):
        """Returns a list of timestamps of length n_points, inclusive."""
        return pd.date_range(start=start_t, end=end_t, periods=n_points).to_list()

    @staticmethod
    def generate_multistep_walk(terrain: Any, steps: List[Tuple[int, int]], T: int,
                                kernel_mapping: Any, tensor_map: Any, plot: bool = False,
                                plot_title: str = "Correlated Walk on terrain with multiple steps") -> np.ndarray:
        """Generate a multistep walk from terrain data.

        Args:
            terrain: Terrain map
            steps: List of step points as (x, y) tuples
            T: Time steps
            kernel_mapping: Kernel mapping parameters
            tensor_map: Tensor map for terrain
            plot: Whether to plot the walk
            plot_title: Title of the plot

        Returns:
            numpy array of walk points
        """
        if len(steps) < 2:
            raise ValueError("At least two steps are required for multistep walk")

        full_path = np.empty((0, 2))

        for i in range(len(steps) - 1):
            start_x, start_y = steps[i]
            end_x, end_y = steps[i + 1]

            # Generate segment
            dp_matrix = WalkerHelper.generate_single_segment(
                terrain, start_x, start_y, T, kernel_mapping, tensor_map
            )

            # Backtrace segment
            segment = WalkerHelper.backtrace_single_segment(
                dp_matrix, T, tensor_map, terrain, end_x, end_y, kernel_mapping
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

    @staticmethod
    def create_timed_df(steps_df, geodetic_path_df, animal_id, idx, segment_boundaries):
        rows = []
        for i in range(len(idx) - 1):
            t_start = steps_df.loc[idx[i], "time"]
            t_end = steps_df.loc[idx[i + 1], "time"]

            a = segment_boundaries[i]
            b = segment_boundaries[i + 1]
            # slice; ensure we don't go out of range
            seg_df = geodetic_path_df.iloc[a:b].copy()
            n = len(seg_df)
            if n == 0:
                continue

            seg_df["time"] = WalkerHelper.interpolate_timestamps(t_start, t_end, n)
            seg_df["traj_id"] = animal_id
            rows.append(seg_df)

        # Also add final single-point segment if necessary (from last observation)
        # if the last segment boundary didn't include the final appended grid point, include it
        if segment_boundaries[-1] < len(geodetic_path_df):
            last_seg = geodetic_path_df.iloc[segment_boundaries[-1]:].copy()
            if len(last_seg) > 0:
                t_last = steps_df.loc[idx[-1], "time"]
                last_seg["time"] = [t_last] * len(last_seg)
                last_seg["traj_id"] = animal_id
                rows.append(last_seg)

        if len(rows) == 0:
            # fallback: create a single point at original observation 0
            lon, lat = geodetic_path_df.loc[0, ["longitude", "latitude"]]
            t0 = steps_df.loc[idx[0], "time"]
            rows = [pd.DataFrame([{"longitude": lon, "latitude": lat, "time": t0, "traj_id": animal_id}])]
        return rows
