import typing
import weakref

import numpy.typing as npt

from random_walk_package import matrix_free, tensor_free, biased_walk_init, biased_walk_backtrace, matrix_new
from random_walk_package.bindings.correlated_walk import *
from random_walk_package.bindings.plotter import plot_walk
from random_walk_package.core.WalkerHelper import *


class BiasedWalker:
    def __init__(self, S: int = 3, W: int = 101, H: int = 101, T: int = 50,
                 kernel: Optional[Any] = None):
        """
            Biased Walker class.

            Args:
                S (int): Stepsize, meaning the distance that can be moved in the grid in one step.
                W (int): Width of the grid.
                H (int): Height of the grid.
                T (int): Number of time steps.
                kernel: Optional Kernel which defines transition probabilities between grid cells, if not provided, a standard bivariate Gaussian kernel is used.
        """
        self.S = S
        self.W = W
        self.H = H
        self.T = T
        self.kernels = kernel if kernel is not None else matrix_new(2 * S + 1, 2 * S + 1)
        self.dp_matrix = None
        self.biases = None
        # State tracking
        self._is_initialized = False  # is the DP matrix initialized?
        self._finalizer = weakref.finalize(self, self._cleanup_resources)

    def _validate_parameters(self) -> None:
        """Validate that all required parameters are set."""
        WalkerHelper.validate_parameters(self.T, self.W, self.H, self.S)

    def _cleanup_resources(self) -> None:
        """Clean up all C resources."""
        resources = ["kernels", "dp_matrix", "biases"]
        for resource in resources:
            if getattr(self, resource, None) is not None:
                try:
                    if resource == "kernels":
                        matrix_free(self.kernels)
                    elif resource == "dp_matrix":
                        tensor_free(self.dp_matrix)
                    elif resource == "biases":
                        dll.free_biases(self.biases)
                    setattr(self, resource, None)
                except Exception as e:
                    logger.warning(f"Failed to free {resource}: {e}")

    @property
    def is_ready_for_backtrace(self) -> bool:
        """Check if walker is ready for the generation of a backtrace (walk array)."""
        return self.dp_matrix is not None and self.biases is not None

    from typing import Union, Sequence, Tuple

    def generate(self, S=None, base_kernel: Optional[np.ndarray] = None,
                 bias_offsets: Union[
                     Sequence[Tuple[float, float]], Sequence[Sequence[float]], npt.NDArray[np.float64]] = None,
                 rotations: typing.Union[Sequence[float], npt.NDArray[np.float64]] = None,
                 start_x=None, start_y=None):
        if start_x is None or start_y is None:
            start_x, start_y = self.W // 2, self.H // 2

        self.S = S if S is not None else self.S

        if base_kernel is not None:
            matrix_free(self.kernels)
            self.kernels = WalkerHelper.set_custom_kernel(base_kernel, self.S)

        self._validate_parameters()
        # Clean up previous DP matrix
        if self.dp_matrix is not None:
            tensor_free(self.dp_matrix)
        kernel_width = 2 * self.S + 1
        try:
            self.dp_matrix, self.biases = biased_walk_init(matrix_ptr=self.kernels, W=self.W, H=self.H,
                                                           offsets=bias_offsets, rotations=rotations, start_x=start_x,
                                                           start_y=start_y)
            self._is_initialized = True
            logger.info(f"Successfully generated walk, start=({start_x}, {start_y})")
        except Exception as e:
            self._is_initialized = False
            logger.error(f"Failed to generate walk: {e}")
            raise

    def backtrace(self, end_x, end_y, plot=False, plot_title="Biased Walk"):
        if not self.is_ready_for_backtrace:
            raise ValueError('Initialize walk first before calling backtrace. Call generate first.')

        # Validate end position
        WalkerHelper.validate_point_location(end_x, end_y, self.W, self.H)

        try:
            walk_np = biased_walk_backtrace(self.dp_matrix, self.biases, self.kernels, end_x, end_y)
            if walk_np is None:
                raise RuntimeError("Backtrace returned null path")

            logger.info(f"Successfully backtraced walk to ({end_x}, {end_y})")
            if plot:
                plot_walk(walk_np, self.W, self.H, plot_title)
            return walk_np

        except Exception as e:
            logger.error(f"Failed to backtrace walk: {e}")
            raise

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
