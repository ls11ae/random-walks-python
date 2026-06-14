from pathlib import Path
import tempfile

import numpy as np

from randomwalks.bindings.correlated_walk import CorrelatedWalkBinding
from randomwalks.bindings.data_structures.Kernels import KernelFactory
from randomwalks.bindings.plotter import plot_terrain_walk
from randomwalks.core.WalkerHelper import WalkerHelper
from randomwalks.serialization import walk_to_json


class CorrelatedWalker:
    def __init__(self, *, D=4, S=3, W=101, H=101, T=50, kernels=None,
                 use_serialization=False, serialization_dir=None):
        self.D = D
        self.S = S
        self.W = W
        self.H = H
        self.T = T
        self.kernels = kernels
        self.dp_matrix = None
        self.dp_folder = None
        self._dp_serialized = False
        self._dp_tempdir = None
        self._default_use_serialization = bool(use_serialization or serialization_dir is not None)
        self._default_serialization_dir = serialization_dir
        self.last_walk = None

        if self.kernels is None:
            self.set_kernel()

    def set_kernel(self, kernel=None, *, D=None, S=None,
                   angle_diffusivity=0.3, length_diffusivity=1.0):
        if self.kernels is not None and hasattr(self.kernels, "free"):
            self.kernels.free()

        if D is not None:
            self.D = D
        if S is not None:
            self.S = S

        if kernel is None:
            width = 2 * self.S + 1
            self.kernels = KernelFactory.correlated(
                width,
                self.D,
                angle_diffusivity=angle_diffusivity,
                length_diffusivity=length_diffusivity,
            )
        else:
            kernel = np.asarray(kernel, dtype=np.float64)
            if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                raise ValueError("Correlated kernel base must be a square 2D array")
            self.S = kernel.shape[0] // 2
            self.kernels = KernelFactory.correlated_from_matrix(kernel, self.D)
        return self.kernels

    def generate(self, start=None, *, use_serialization=None, serialization_dir=None):
        WalkerHelper.validate_dimensions(self.W, self.H, self.T, self.S, self.D)
        start = (self.W // 2, self.H // 2) if start is None else start
        start_x, start_y = WalkerHelper.validate_point(start, self.W, self.H, name="start")

        self._clear_dp()
        use_serialization, output_folder = self._prepare_persistent_dp_serialization(
            use_serialization,
            serialization_dir,
        )
        self.dp_matrix = CorrelatedWalkBinding.generate(
            self.kernels,
            self.W,
            self.H,
            self.T,
            start_x,
            start_y,
            use_serialization=use_serialization,
            output_folder=output_folder,
        )
        return self.dp_matrix

    def backtrace(self, end, *, direction=0, plot=False, title="Correlated walk"):
        if self.dp_matrix is None and not self._dp_serialized:
            raise ValueError("Call generate() before backtrace()")
        end_x, end_y = WalkerHelper.validate_point(end, self.W, self.H, name="end")
        walk = CorrelatedWalkBinding.backtrace(
            self.dp_matrix,
            self.kernels,
            self.T,
            end_x,
            end_y,
            direction=direction,
            use_serialization=self._dp_serialized,
            dp_folder=self.dp_folder,
        )
        self.last_walk = walk
        if plot:
            self.plot(title=title)
        return walk

    def walk(self, start, end, *, direction=0, plot=False, title="Correlated walk",
             use_serialization=None, serialization_dir=None):
        self.generate(
            start,
            use_serialization=use_serialization,
            serialization_dir=serialization_dir,
        )
        return self.backtrace(end, direction=direction, plot=plot, title=title)

    def multistep_walk(self, steps, *, direction=0, plot=False, title="Correlated walk",
                       use_serialization=None, serialization_dir=None):
        steps = WalkerHelper.validate_steps(steps, self.W, self.H)
        use_serialization, dp_folder, tempdir = self._temporary_dp_serialization(
            use_serialization,
            serialization_dir,
        )
        try:
            walk = CorrelatedWalkBinding.multi_step(
                self.W,
                self.H,
                self.T,
                self.kernels,
                steps,
                direction=direction,
                use_serialization=use_serialization,
                dp_folder=dp_folder,
            )
            self.last_walk = walk
            if plot:
                self.plot(title=title)
            return walk
        finally:
            if tempdir is not None:
                tempdir.cleanup()

    def plot(self, *, walk=None, title=None, show=True, ax=None):
        return plot_terrain_walk(
            walk=self.last_walk if walk is None else walk,
            width=self.W,
            height=self.H,
            title=title,
            show=show,
            ax=ax,
        )

    def to_json(self, json_file, *, walk=None, steps=None, terrain=None, metadata=None):
        walk = self.last_walk if walk is None else walk
        if walk is None:
            raise ValueError("No walk available. Call walk() or provide walk= first.")
        return walk_to_json(
            walk,
            json_file,
            steps=steps,
            terrain=terrain,
            width=self.W,
            height=self.H,
            metadata=metadata,
        )

    save_walk = to_json

    def _prepare_persistent_dp_serialization(self, use_serialization, serialization_dir):
        use_serialization = self._resolve_use_serialization(use_serialization, serialization_dir)
        if not use_serialization:
            return False, None

        directory = serialization_dir if serialization_dir is not None else self._default_serialization_dir
        if directory is None:
            self._dp_tempdir = tempfile.TemporaryDirectory(prefix="rw_correlated_dp_")
            directory = self._dp_tempdir.name
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.dp_folder = str(path)
        self._dp_serialized = True
        return True, self.dp_folder

    def _temporary_dp_serialization(self, use_serialization, serialization_dir):
        use_serialization = self._resolve_use_serialization(use_serialization, serialization_dir)
        if not use_serialization:
            return False, None, None

        directory = serialization_dir if serialization_dir is not None else self._default_serialization_dir
        tempdir = None
        if directory is None:
            tempdir = tempfile.TemporaryDirectory(prefix="rw_correlated_dp_")
            directory = tempdir.name
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return True, str(path), tempdir

    def _resolve_use_serialization(self, use_serialization, serialization_dir):
        if use_serialization is None:
            return self._default_use_serialization or serialization_dir is not None
        return bool(use_serialization or serialization_dir is not None)

    def _clear_dp(self):
        if self.dp_matrix is not None:
            self.dp_matrix.free()
        if self._dp_tempdir is not None:
            self._dp_tempdir.cleanup()
        self.dp_matrix = None
        self.dp_folder = None
        self._dp_serialized = False
        self._dp_tempdir = None

    def close(self):
        self._clear_dp()
        if self.kernels is not None and hasattr(self.kernels, "free"):
            self.kernels.free()
        self.kernels = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        self.close()
