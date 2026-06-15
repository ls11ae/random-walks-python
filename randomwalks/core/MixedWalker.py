from pathlib import Path
import subprocess
import tempfile

import numpy as np

from randomwalks.bindings.data_structures.KernelContext import KernelContextHandle
from randomwalks.bindings.data_structures.KernelMapping import KernelMapping
from randomwalks.bindings.data_structures.Terrain import TerrainMapHandle
from randomwalks.bindings.data_structures.types import ComputationMode, Reachability
from randomwalks.bindings.mixed_walk import MixedWalkBinding
from randomwalks.bindings.plotter import plot_terrain_walk
from randomwalks.core.WalkerHelper import WalkerHelper


class MixedWalker:
    def __init__(
            self,
            data,
            resolution=400,
            out_directory=None,
            *,
            time_col="timestamp",
            lon_col="location-long",
            lat_col="location-lat",
            id_col="individual-local-identifier",
            crs="EPSG:4326",
            is_marine=False,
            movement_policy=None,
            reference_speed=None,
            reachability: Reachability = Reachability.SOFT,
            context: ComputationMode = ComputationMode.KERNEL_POOL,
            serialization_dir=None,
    ):
        if out_directory is None:
            out_directory = serialization_dir

        self.data = _coerce_trajectory_collection(
            data,
            time_col=time_col,
            lon_col=lon_col,
            lat_col=lat_col,
            id_col=id_col,
            target_crs=crs,
        )
        self.time_col = self.data.t
        self.lon_col = lon_col
        self.lat_col = lat_col
        self.id_col = self.data.get_traj_id_col()
        self.crs = crs
        self.resolution = resolution
        self.out_directory = out_directory
        self.animal_proc = None
        self.is_marine = bool(is_marine)
        self.movement_policy = movement_policy
        self.reference_speed = reference_speed
        self.reachability = reachability
        self.context_mode = context
        self._requested_serialization_dir = serialization_dir
        self._context_tempdir = None
        self.serialization_dir = None

    @classmethod
    def from_movebank_data(cls, data, **kwargs):
        return cls(data, **kwargs)

    def _init_movebank_pipeline(
            self,
            data,
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
        self.data = _coerce_trajectory_collection(
            data,
            time_col=time_col,
            lon_col=lon_col,
            lat_col=lat_col,
            id_col=id_col,
            target_crs=crs,
        )
        self.time_col = self.data.t
        self.lon_col = lon_col
        self.lat_col = lat_col
        self.id_col = self.data.get_traj_id_col()
        self.crs = crs
        self.resolution = resolution
        self.out_directory = out_directory
        self.animal_proc = None
        self.is_marine = bool(is_marine)
        self.movement_policy = movement_policy
        self.reference_speed = reference_speed

    def process_movebank_data(self):
        from randomwalks.core.AnimalMovement import AnimalMovementProcessor

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

    @staticmethod
    def has_cuda():
        try:
            out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
            return b"CUDA" in out or b"NVIDIA" in out
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @staticmethod
    def preprocess_mixed_gpu(*args, **kwargs):
        pass

    @staticmethod
    def mixed_walk_gpu(*args, **kwargs):
        pass

    @staticmethod
    def free_kernel_pool(*args, **kwargs):
        pass

    def generate_walks(self, mapping=None, serialization_dir=None, movement_policy=None):
        import geopandas as gpd
        import movingpandas as mpd
        import pandas as pd

        self.process_movebank_data()
        steps_dict = self.animal_proc.create_movement_data_dict()
        per_animal_gdfs = []
        context_mode = ComputationMode.SERIALIZATION if serialization_dir is not None else self.context_mode
        movement_policy = movement_policy or self.movement_policy

        for animal_id, trajectory in steps_dict.items():
            print(f"Generating walks for {animal_id}")
            terrain_map = TerrainMapHandle.from_file(self._terrain_path(animal_id), delim=" ")
            mapping = KernelMapping.mesa_default() if mapping is None else mapping
            context = self._context_for_mapping(terrain_map, mapping, context_mode, serialization_dir)

            try:
                steps = trajectory.df
                steps_df = steps_dict[animal_id].df
                idx = steps_df.index
                full_path = []
                segment_boundaries = [0]

                for i in range(len(idx) - 1):
                    print(
                        f"[{i + 1}/{len(idx)}] Generating walk for {animal_id} at {steps['time'].iloc[i]} - {steps['time'].iloc[i + 1]}:")
                    start_x, start_y = int(steps["grid_x"].iloc[i]), int(steps["grid_y"].iloc[i])
                    end_x, end_y = int(steps["grid_x"].iloc[i + 1]), int(steps["grid_y"].iloc[i + 1])

                    if start_x == end_x and start_y == end_y:
                        segment = [(start_x, start_y)]
                    else:
                        T = self._resolve_segment_T(
                            (start_x, start_y),
                            (end_x, end_y),
                            steps["time"].iloc[i],
                            steps["time"].iloc[i + 1],
                            movement_policy,
                        )
                        segment = MixedWalkBinding.single_state_walk(
                            context,
                            T,
                            start_x,
                            start_y,
                            end_x,
                            end_y,
                        )
                        if segment is None:
                            segment = [(start_x, start_y), (end_x, end_y)]

                    full_path.extend(segment[:-1] if len(segment) > 1 else segment)
                    segment_boundaries.append(len(full_path))

                # plot_terrain_walk(terrain=terrain_map, walk=full_path, title=f"{animal_id}", show=True)

                final_gdf = self.animal_proc.movebank_path_to_gdf(full_path, steps_df, animal_id, idx,
                                                                  segment_boundaries)
                if final_gdf is not None:
                    per_animal_gdfs.append(final_gdf)
            finally:
                context.free()
                terrain_map.free()

        if len(per_animal_gdfs) == 0:
            empty = gpd.GeoDataFrame(columns=[self.id_col, "time", "geometry"], geometry="geometry", crs="EPSG:4326")
            return mpd.TrajectoryCollection(empty, traj_id_col=self.id_col, t="time")

        combined = pd.concat(per_animal_gdfs, ignore_index=True)
        combined_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
        combined_gdf["time"] = pd.to_datetime(combined_gdf["time"])
        return mpd.TrajectoryCollection(combined_gdf, traj_id_col=self.id_col, t="time")

    def _context_for_mapping(self, terrain, mapping, mode, serialization_dir=None):
        if mode == ComputationMode.KERNEL_POOL:
            return KernelContextHandle.pool(terrain, mapping, self.reachability)
        if mode == ComputationMode.ON_THE_FLY:
            return KernelContextHandle.on_fly(terrain, mapping, self.reachability)
        if mode == ComputationMode.SERIALIZATION:
            directory = serialization_dir or self._prepare_context_serialization_dir()
            return KernelContextHandle.serialization(terrain, mapping, self.reachability, directory)
        raise ValueError("value between 0 and 2 - ComputationMode enum")

    def _terrain_path(self, animal_id):
        return self.animal_proc.terrain_paths.get(animal_id) or self.animal_proc.terrain_paths[str(animal_id)]

    def _resolve_segment_T(self, start, end, start_time, end_time, movement_policy):
        if movement_policy is not None:
            T, _ = movement_policy.resolve(
                start_point=start,
                end_point=end,
                start_time=start_time,
                end_time=end_time,
                reference_speed=self.reference_speed,
            )
            return int(T)

        manhattan = abs(start[0] - end[0]) + abs(start[1] - end[1])
        return 5 if manhattan < 5 else int(manhattan)

    @staticmethod
    def generate_custom_walks(
            terrain,
            steps,
            T,
            mappings=None,
            reachability: Reachability = Reachability.SOFT,
            context: ComputationMode = ComputationMode.KERNEL_POOL,
            serialization_dir=None,
            plot=False,
            plot_title="Mixed Walk",
    ):
        mapping = KernelMapping.mesa_default() if mappings is None else mappings
        kernel_context = _custom_context(terrain, mapping, reachability, context, serialization_dir)

        try:
            full_path = np.empty((0, 2), dtype=np.int64)
            steps = WalkerHelper.validate_steps(steps, terrain.width, terrain.height)
            for start, end in zip(steps, steps[1:]):
                segment = _custom_segment(kernel_context, terrain.width, terrain.height, start, end, T)
                full_path = WalkerHelper.append_segment(full_path, segment)

            if plot:
                plot_terrain_walk(terrain=terrain, walk=full_path, steps=steps, title=plot_title)
            return full_path
        finally:
            kernel_context.free()

    @staticmethod
    def generate_custom_utilization_distribution(
            terrain,
            steps,
            T,
            kernel_mapping=None,
            *,
            reachability: Reachability = Reachability.SOFT,
            context: ComputationMode = ComputationMode.KERNEL_POOL,
            serialization_dir=None,
            plot=False,
            plot_title="Mixed utilization distribution",
    ):
        mapping = KernelMapping.mesa_default() if kernel_mapping is None else kernel_mapping
        kernel_context = _custom_context(terrain, mapping, reachability, context, serialization_dir)

        try:
            steps = WalkerHelper.validate_steps(steps, terrain.width, terrain.height)
            total = np.zeros((terrain.height, terrain.width), dtype=np.float64)
            for start, end in zip(steps, steps[1:]):
                segment_ud = _custom_utilization_distribution(
                    kernel_context,
                    terrain.width,
                    terrain.height,
                    start,
                    end,
                    T,
                )
                if not np.isnan(segment_ud).any():
                    total += segment_ud

            if plot:
                plot_terrain_walk(terrain=terrain, steps=steps, ud=total, title=plot_title)
            return total
        finally:
            kernel_context.free()

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
        if getattr(self, "_context_tempdir", None) is not None:
            self._context_tempdir.cleanup()
        self._context_tempdir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        self.close()


def _coerce_trajectory_collection(*args, **kwargs):
    from randomwalks.core.AnimalMovement import coerce_trajectory_collection
    return coerce_trajectory_collection(*args, **kwargs)


def _custom_context(terrain, mapping, reachability, mode, serialization_dir):
    if mode == ComputationMode.KERNEL_POOL:
        return KernelContextHandle.pool(terrain, mapping, reachability)
    if mode == ComputationMode.ON_THE_FLY:
        return KernelContextHandle.on_fly(terrain, mapping, reachability)
    if mode == ComputationMode.SERIALIZATION:
        if serialization_dir is None:
            raise ValueError("serialization_dir is required for custom serialized walks")
        return KernelContextHandle.serialization(terrain, mapping, reachability, serialization_dir)
    raise ValueError("value between 0 and 2 - ComputationMode enum")


def _custom_segment(kernel_context, width, height, start, end, T):
    start_x, start_y = WalkerHelper.validate_point(start, width, height, name="start")
    end_x, end_y = WalkerHelper.validate_point(end, width, height, name="end")
    dp = MixedWalkBinding.walk(kernel_context, T, start_x, start_y)
    try:
        return MixedWalkBinding.backtrace(dp, kernel_context, end_x, end_y)
    finally:
        dp.free()


def _custom_utilization_distribution(kernel_context, width, height, start, end, T):
    start_x, start_y = WalkerHelper.validate_point(start, width, height, name="start")
    end_x, end_y = WalkerHelper.validate_point(end, width, height, name="end")
    dp = MixedWalkBinding.walk(kernel_context, T, start_x, start_y)
    try:
        ud_tensor = MixedWalkBinding.utilization_distribution(dp, kernel_context, end_x, end_y)
        try:
            return ud_tensor.to_numpy_sum(width, height, average=True)
        finally:
            ud_tensor.free()
    finally:
        dp.free()
