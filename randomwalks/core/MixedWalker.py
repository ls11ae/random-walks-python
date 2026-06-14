from pathlib import Path
import tempfile

import numpy as np

from randomwalks.bindings.data_structures.KernelContext import KernelContextHandle
from randomwalks.bindings.data_structures.KernelMapping import KernelMapping
from randomwalks.bindings.data_structures.Terrain import TerrainMapHandle
from randomwalks.bindings.data_structures.types import Reachability, ComputationMode
from randomwalks.bindings.mixed_walk import MixedWalkBinding
from randomwalks.bindings.plotter import plot_terrain_walk
from randomwalks.core.WalkerHelper import WalkerHelper
from randomwalks.serialization import walk_to_json


class MixedWalker:
    def __init__(self, terrain: TerrainMapHandle,
                 mapping: KernelMapping = None,
                 *,
                 T=50,
                 reachability: Reachability = Reachability.SOFT,
                 context: ComputationMode = ComputationMode.KERNEL_POOL,
                 serialization_dir=None):
        if not isinstance(terrain, TerrainMapHandle):
            terrain = TerrainMapHandle.from_ptr(terrain, owned=False)

        self.terrain = terrain
        self.terrain._owned = False
        self.T = T
        self.mapping = mapping if mapping is not None else KernelMapping.mesa_default(terrain)
        if mapping is None:
            self.mapping._owned = False
        self.reachability = reachability
        self.context_mode = context
        self._requested_serialization_dir = serialization_dir
        self._context_tempdir = None
        self.serialization_dir = None
        self.kernel_context = self._create_context(context)
        self.last_walk = None
        self.last_ud = None

    @classmethod
    def from_file(cls, terrain_file, *, delim=" ", mapping=None, reachability=Reachability.SOFT,
                  context=ComputationMode.KERNEL_POOL, T=50, serialization_dir=None):
        terrain = TerrainMapHandle.from_file(terrain_file, delim)
        return cls(
            terrain,
            mapping=mapping,
            T=T,
            reachability=reachability,
            context=context,
            serialization_dir=serialization_dir,
        )

    def _init_movebank_pipeline(
        self,
        data,
        kernel_mapping,
        resolution,
        out_directory,
        time_col="timestamp",
        lon_col="location-long",
        lat_col="location-lat",
        id_col="individual-local-identifier",
        crs="EPSG:4326",
        is_marine=False,
        movement_policy=None,
        reference_speed=None,
    ):
        self.data = data
        self.time_col = time_col
        self.lon_col = lon_col
        self.lat_col = lat_col
        self.id_col = id_col
        self.crs = crs
        self.resolution = resolution
        self.out_directory = out_directory
        self.animal_proc = None
        self.mapping = kernel_mapping
        self.is_marine = bool(is_marine)
        self.movement_policy = movement_policy
        self.reference_speed = reference_speed

    def process_movebank_data(self):
        from randomwalks.core.AnimalMovement import AnimalMovementProcessor

        if not hasattr(self, "data"):
            raise ValueError("Movebank data was not configured for this walker")

        self.animal_proc = AnimalMovementProcessor(
            data=self.data,
            time_col=self.time_col,
            lon_col=self.lon_col,
            lat_col=self.lat_col,
            id_col=self.id_col,
            target_crs=self.crs,
            movement_policy=self.movement_policy,
            reference_speed=self.reference_speed,
        )
        self.animal_proc.create_landcover_data_txt(
            resolution=self.resolution,
            is_marine=self.is_marine,
            out_directory=self.out_directory,
        )
        return self.animal_proc

    _process_movebank_data = process_movebank_data

    @property
    def width(self):
        return self.terrain.width

    @property
    def height(self):
        return self.terrain.height

    def _create_context(self, mode: ComputationMode):
        if mode == ComputationMode.KERNEL_POOL:
            return KernelContextHandle.pool(self.mapping, self.reachability)
        if mode == ComputationMode.ON_THE_FLY:
            return KernelContextHandle.on_fly(self.mapping, self.reachability)
        if mode == ComputationMode.SERIALIZATION:
            return KernelContextHandle.serialization(
                self.mapping,
                self.reachability,
                self._prepare_context_serialization_dir(),
            )
        raise ValueError("value between 0 and 2 - ComputationMode enum")

    def __segment_dp(self, start, *, T=None):
        T = self._resolve_T(T)
        start_x, start_y = WalkerHelper.validate_point(start, self.width, self.height, name="start")
        return MixedWalkBinding.walk(self.kernel_context, T, start_x, start_y)

    def __segment(self, start, end, *, T=None):
        end_x, end_y = WalkerHelper.validate_point(end, self.width, self.height, name="end")
        dp = self.__segment_dp(start, T=T)
        try:
            return MixedWalkBinding.backtrace(dp, self.kernel_context, end_x, end_y)
        finally:
            dp.free()

    def __segment_utilization_distribution(self, start, end, T):
        T = self._resolve_T(T)
        end_x, end_y = WalkerHelper.validate_point(end, self.width, self.height, name="end")
        dp = self.__segment_dp(start, T=T)
        try:
            ud_tensor = MixedWalkBinding.utilization_distribution(dp, self.kernel_context, end_x, end_y)
            try:
                return ud_tensor.to_numpy_sum(self.width, self.height, average=True)
            finally:
                ud_tensor.free()
        finally:
            dp.free()

    def walk(self, steps, T=None, plot=False, title="Mixed walk"):
        T = self._resolve_T(T)
        steps = WalkerHelper.validate_steps(steps, self.width, self.height)
        full_path = np.empty((0, 2), dtype=np.int64)
        for start, end in zip(steps, steps[1:]):
            segment = self.__segment(start, end, T=T)
            full_path = WalkerHelper.append_segment(full_path, segment)
        self.last_walk = full_path
        if plot:
            self.plot(walk=full_path, steps=steps, title=title)
        return full_path

    def utilization_distribution(self, steps, *, T=None, plot=False, title="Mixed utilization distribution"):
        T = self._resolve_T(T)
        steps = WalkerHelper.validate_steps(steps, self.width, self.height)
        total = np.zeros((self.height, self.width), dtype=np.float64)

        for start, end in zip(steps, steps[1:]):
            segment_ud = self.__segment_utilization_distribution(start, end, T=T)
            if np.isnan(segment_ud).any():
                continue
            total += segment_ud

        self.last_ud = total
        if plot:
            self.plot(ud=total, steps=steps, title=title)
        return total

    def walk_with_ud(self, steps, *, T=None, plot=False, title="Mixed walk"):
        T = self._resolve_T(T)
        walk = self.walk(steps, T=T)
        ud = self.utilization_distribution(steps, T=T)
        if plot:
            self.plot(walk=walk, steps=steps, ud=ud, title=title)
        return walk, ud

    def plot(self, *, walk=None, steps=None, ud=None, title=None, show=True, ax=None):
        return plot_terrain_walk(
            terrain=self.terrain,
            walk=self.last_walk if walk is None else walk,
            steps=steps,
            ud=self.last_ud if ud is None else ud,
            title=title,
            show=show,
            ax=ax,
        )

    def to_json(self, json_file, *, walk=None, steps=None, ud=None, metadata=None):
        walk = self.last_walk if walk is None else walk
        ud = self.last_ud if ud is None else ud
        if walk is None:
            raise ValueError("No walk available. Call walk() or provide walk= first.")
        return walk_to_json(
            walk,
            json_file,
            steps=steps,
            terrain=self.terrain,
            ud=ud,
            width=self.width,
            height=self.height,
            metadata=metadata,
        )

    save_walk = to_json

    def _resolve_T(self, T):
        T = self.T if T is None else T
        WalkerHelper.validate_dimensions(self.width, self.height, T)
        return T

    def _prepare_context_serialization_dir(self):
        directory = self._requested_serialization_dir
        if directory is None:
            self._context_tempdir = tempfile.TemporaryDirectory(prefix="rw_mixed_context_")
            directory = self._context_tempdir.name
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.serialization_dir = str(path)
        return self.serialization_dir

    def close(self):
        if getattr(self, "kernel_context", None):
            self.kernel_context.free()
        if getattr(self, "_context_tempdir", None) is not None:
            self._context_tempdir.cleanup()
        self.kernel_context = None
        self.mapping = None
        self.terrain = None
        self._context_tempdir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        self.close()
