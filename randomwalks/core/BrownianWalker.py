import numpy as np

from randomwalks.bindings.brownian_walk import BrownianWalkBinding
from randomwalks.bindings.data_structures.Matrix import MatrixHandle
from randomwalks.bindings.plotter import plot_terrain_walk
from randomwalks.core.WalkerHelper import WalkerHelper
from randomwalks.serialization import walk_to_json


class BrownianWalker:
    def __init__(self, *, S=3, W=101, H=101, T=50, kernel=None, sigma=3.0):
        self.S = S
        self.W = W
        self.H = H
        self.T = T
        self.sigma = sigma
        self.kernel = kernel
        self.dp_matrix = None
        self.last_walk = None

        if self.kernel is None:
            self.set_kernel()

    def set_kernel(self, kernel=None, *, sigma=None, S=None):
        if self.kernel is not None and hasattr(self.kernel, "free"):
            self.kernel.free()

        if S is not None:
            self.S = S
        if sigma is not None:
            self.sigma = sigma

        if kernel is None:
            size = 2 * self.S + 1
            self.kernel = MatrixHandle.gaussian(size, size, self.sigma)
        else:
            self.kernel = MatrixHandle.from_numpy(np.asarray(kernel, dtype=np.float64))
            self.S = self.kernel.contents.width // 2
        return self.kernel

    def generate(self, start=None):
        WalkerHelper.validate_dimensions(self.W, self.H, self.T, self.S)
        start = (self.W // 2, self.H // 2) if start is None else start
        start_x, start_y = WalkerHelper.validate_point(start, self.W, self.H, name="start")

        if self.dp_matrix is not None:
            self.dp_matrix.free()
        self.dp_matrix = BrownianWalkBinding.generate(self.kernel, self.W, self.H, self.T, start_x, start_y)
        return self.dp_matrix

    def backtrace(self, end, *, plot=False, title="Brownian walk"):
        if self.dp_matrix is None:
            raise ValueError("Call generate() before backtrace()")
        end_x, end_y = WalkerHelper.validate_point(end, self.W, self.H, name="end")
        walk = BrownianWalkBinding.backtrace(self.dp_matrix, self.kernel, end_x, end_y)
        self.last_walk = walk
        if plot:
            self.plot(title=title)
        return walk

    def walk(self, start, end, *, plot=False, title="Brownian walk"):
        self.generate(start)
        return self.backtrace(end, plot=plot, title=title)

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

    def close(self):
        if self.dp_matrix is not None:
            self.dp_matrix.free()
        if self.kernel is not None:
            self.kernel.free()
        self.dp_matrix = None
        self.kernel = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        self.close()
