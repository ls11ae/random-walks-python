from __future__ import annotations

import os

from randomwalks.bindings.data_structures.EnvWeights import EnvWeights
from randomwalks.bindings.data_structures.KernelMapping import KernelMapping
from randomwalks.bindings.data_structures.Terrain import TerrainMapHandle
from randomwalks.bindings.mixed_walk import MixedWalkBinding
from randomwalks.core.MixedWalker import MixedWalker
from randomwalks.core.MovementPolicy import MovementPolicy, TimeStepPolicy


class MixedTimeWalker(MixedWalker):
    def __init__(self, data, env_data, resolution, out_directory, env_samples,
                 kernel_resolver=None,
                 time_col="timestamp",
                 lon_col="location-long",
                 lat_col="location-lat",
                 id_col="tag-local-identifier",
                 crs="EPSG:4326",
                 is_marine=False,
                 movement_policy=None,
                 reference_speed=None):
        movement_policy = movement_policy or TimeStepPolicy(timestep_s=3600)
        super().__init__(
            data,
            resolution=resolution,
            out_directory=out_directory,
            time_col=time_col,
            lon_col=lon_col,
            lat_col=lat_col,
            id_col=id_col,
            crs=crs,
            is_marine=is_marine,
            movement_policy=movement_policy,
            reference_speed=reference_speed,
        )
        self.env_data = env_data
        self.env_paths: dict[tuple[str, str, str], str] = {}
        self.kernel_resolver = kernel_resolver
        self.env_samples = env_samples
        self.process_movebank_data()

    def process_movebank_data(self):
        super().process_movebank_data()
        self.animal_proc.env_samples = self.env_samples
        kernel_dir = os.path.join(self.out_directory, "kernels")
        self.env_paths = self.animal_proc.kernel_params_per_animal_binary(
            env_path=self.env_data,
            kernel_resolver=self.kernel_resolver,
            time_stamp="time",
            lon="longitude",
            lat="latitude",
            out_directory=kernel_dir,
        )
        print("[PREPROCESSING] kernel params loaded")
        return self.animal_proc

    _process_movebank_data = process_movebank_data

    def generate_walks(
            self,
            mapping: KernelMapping | None = None,
            env_weights: EnvWeights | None = None,
            movement_policy: MovementPolicy | None = None,
    ):
        import geopandas as gpd
        import movingpandas as mpd
        import pandas as pd

        if env_weights is None:
            env_weights = EnvWeights.bias_only()
            owns_env_weights = True
        else:
            owns_env_weights = False

        movement_policy = movement_policy or self.movement_policy or TimeStepPolicy(timestep_s=1800)
        steps_dict = self.animal_proc.create_movement_data_dict()
        per_animal_gdfs = []

        try:
            for animal_id, trajectory in steps_dict.items():
                terrain_path = self._terrain_path(animal_id)
                terrain_map = TerrainMapHandle.from_file(terrain_path, delim=" ")
                mapping = mapping or KernelMapping.mesa_default()

                try:
                    steps = trajectory.df
                    full_path = []
                    steps_df = steps_dict[animal_id].df
                    idx = steps_df.index
                    segment_boundaries = [0]

                    for i in range(len(idx) - 1):
                        start_x, start_y = int(steps["grid_x"].iloc[i]), int(steps["grid_y"].iloc[i])
                        end_x, end_y = int(steps["grid_x"].iloc[i + 1]), int(steps["grid_y"].iloc[i + 1])
                        start_date, end_date = steps["time"].iloc[i], steps["time"].iloc[i + 1]

                        if start_x == end_x and start_y == end_y:
                            segment = [(start_x, start_y)]
                            full_path.extend(segment)
                            segment_boundaries.append(len(full_path))
                            continue

                        T, _ = movement_policy.resolve(
                            start_point=[start_x, start_y],
                            end_point=[end_x, end_y],
                            start_time=start_date,
                            end_time=end_date,
                            reference_speed=self.reference_speed,
                            movement_diffusivity=2,
                        )

                        ts = pd.Timestamp(start_date).strftime("%Y%m%dT%H")
                        te = pd.Timestamp(end_date).strftime("%Y%m%dT%H")
                        env_path = self.env_paths.get((str(animal_id), ts, te))
                        if env_path is None:
                            segment = [(start_x, start_y), (end_x, end_y)]
                        else:
                            segment = MixedWalkBinding.time_walk_env_binary(
                                T=T,
                                mapping=mapping,
                                terrain=terrain_map,
                                env_binary_path=env_path,
                                env_weights=env_weights,
                                start_point=(start_x, start_y),
                                end_point=(end_x, end_y),
                                start_time=start_date,
                                end_time=end_date,
                            )
                            if segment is None:
                                segment = [(start_x, start_y), (end_x, end_y)]

                        full_path.extend(segment[:-1] if len(segment) > 1 else segment)
                        segment_boundaries.append(len(full_path))

                    final_gdf = self.animal_proc.movebank_path_to_gdf(full_path, steps_df, animal_id, idx,
                                                                      segment_boundaries)
                    if final_gdf is not None:
                        per_animal_gdfs.append(final_gdf)
                finally:
                    terrain_map.free()

        finally:
            if owns_env_weights:
                env_weights.free()

        if len(per_animal_gdfs) == 0:
            return mpd.TrajectoryCollection(gpd.GeoDataFrame(columns=["geometry"]), traj_id_col=self.id_col, t="time")

        combined = pd.concat(per_animal_gdfs, ignore_index=True)
        combined_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
        combined_gdf["time"] = pd.to_datetime(combined_gdf["time"])
        return mpd.TrajectoryCollection(combined_gdf, traj_id_col=self.id_col, t="time")


__all__ = ["MixedTimeWalker"]
