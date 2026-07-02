from __future__ import annotations

from enum import IntEnum
import os

import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd
from environmentcma.crs import grid_shape_from_bbox, utm_to_grid, grid_to_geo_walk, padded_utm_bbox
from segmentationcma.core import bbox_of_segment
from shapely import Point

from randomwalks.bindings.data_structures.KernelContext import KernelContextHandle
from randomwalks.bindings.data_structures.KernelMapping import KernelMapping
from randomwalks.bindings.data_structures.Terrain import Animal, MesaLandcover, TerrainMapHandle, BarrierMode
from randomwalks.bindings.data_structures.types import Reachability
from randomwalks.bindings.mixed_walk import MixedWalkBinding
from randomwalks.core.KernelFactory import StateAnnotationMethod
from randomwalks.core.MixedWalker import MixedWalker
from randomwalks.core.WalkerHelper import WalkerHelper
from randomwalks.bindings.data_structures.Kernels import kernel_array, kernel_for_grid, kernel_state_value, \
    normalize_kernel
from randomwalks.bindings.step_segments import segments_for_steps
from randomwalks.move_apps_patch import apply_moveapps_id_dtype_patch, debug_patch_state, force_tc_id_object_inplace, \
    merge_traj_collections


class StateDependentWalker(MixedWalker):
    def __init__(self, data, animal_type, resolution, out_directory, movement_policy, barriers: list[int] | None = None,
                 time_col="timestamp", lon_col="location-long", lat_col="location-lat",
                 id_col="individual_local_identifier", crs="EPSG:4326"):
        self.state_kernels = None
        self.animal = _coerce_animal(animal_type)
        self.n_hmm_states = 0
        self.dt_tolerance = 1
        self.rnge = 0
        self.kernel_state_col = "state"
        self.annotation_result = None
        self.barrier_mode = BarrierMode.AVOID
        self.barriers = barriers

        self.is_brownian = False
        is_marine = self.animal in (Animal.MARINE, Animal.AIRBORNE)

        apply_moveapps_id_dtype_patch()
        debug_patch_state()
        self.original_data = None
        if isinstance(data, mpd.TrajectoryCollection):
            force_tc_id_object_inplace(data)
            import copy
            data_copy = copy.deepcopy(data)
            self.original_data = data_copy

        super().__init__(
            data,
            resolution=resolution,
            out_directory=out_directory,
            time_col=time_col,
            lon_col=lon_col,
            lat_col=lat_col,
            id_col=id_col,
            crs=crs,
            movement_policy=movement_policy,
            is_marine=is_marine,
        )

    def annotate_behavior(
            self,
            method: StateAnnotationMethod | str = StateAnnotationMethod.HMM,
            features=None,
            num_states=3,
            penalty=10.0,
            plot_path=None,
    ):
        super()._process_movebank_data(create_landcover=True)

        self.n_hmm_states = num_states
        if self.original_data is None:
            self.original_data = self.animal_proc.traj_coll
        self.annotation_result = self.animal_proc.annotate_behavior(
            method=method,
            features=features,
            num_states=num_states,
            penalty=penalty,
            plot_path=plot_path,
        )
        return self.annotation_result

    def get_kernels(
            self,
            dt_tolerance,
            rnge,
            state_col="state",
            is_brownian=False,
            plot_dir=None,
            density_config=None,
            density_preset=None,
            density_method=None,
            density_model=None,
            n_components=None,
            covariance_type=None,
            reg_covar=None,
            reg_covariance=None,
    ):
        if self.animal_proc is None or self.animal_proc.annotation_result is None:
            raise ValueError("Call annotate_behavior() before get_kernels().")
        self.dt_tolerance = dt_tolerance
        self.rnge = rnge
        self.is_brownian = is_brownian
        self.kernel_state_col = state_col

        if plot_dir:
            os.makedirs(os.path.dirname(plot_dir), exist_ok=True)

        cor_zs, brw_zs = self.animal_proc.generate_state_kernels(
            state_col=state_col,
            dt_tolerance=self.dt_tolerance,
            rnge=self.rnge,
            out_dir=plot_dir or self.out_directory,
            density_config=density_config,
            density_preset=density_preset,
            density_method=density_method,
            density_model=density_model,
            n_components=n_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            reg_covariance=reg_covariance,
        )
        selected_kernels = brw_zs if is_brownian and self.animal != Animal.AIRBORNE else cor_zs
        state_kernels = {}
        for state_kernel in selected_kernels:
            kernel = kernel_array(state_kernel)
            if kernel is None or np.sum(kernel) == 0:
                continue
            state_kernels[kernel_state_value(state_kernel)] = normalize_kernel(kernel)

        if not state_kernels:
            raise ValueError("No usable state kernels were generated.")
        self.state_kernels = state_kernels
        return state_kernels

    def __get_steps(self):
        return {traj.id: traj.df for traj in self.animal_proc.traj.trajectories}

    def generate_walks(self, mapping=None, amount=1,
                       max_cell_size=10, barrier_mode: BarrierMode = BarrierMode.AVOID):
        self.barrier_mode = barrier_mode
        if not self.animal_proc.terrain_paths:
            self.animal_proc.create_landcover_data_txt(
                resolution=self.resolution,
                is_marine=self.is_marine,
                out_directory=self.out_directory,
            )
        return self._randomwalks_core(mapping=None, serialization_dir=self.serialization_dir, amount_of_walks=amount,
                                      ud=False, save_plots=False, max_cell_size=max_cell_size)

    def _randomwalks_core(
            self,
            mapping=None,
            serialization_dir=None,
            amount_of_walks=1,
            ud=False,
            movement_policy=None,
            save_plots=False,
            max_cell_size=10,
    ):
        py_kernels = self.state_kernels
        t_col = self.original_data.t
        id_col = self.original_data.get_traj_id_col()
        movement_policy = self.movement_policy

        steps_dict = self.__get_steps()
        per_animal_gdfs = []

        animal_index = 0
        for animal_id, steps in steps_dict.items():
            animal_index += 1
            segments = segments_for_steps(steps, max_cell_size=max_cell_size, resolution=self.resolution)
            animal_rows = []

            for segment in segments:
                seg_start, seg_end = segment
                min_lon, min_lat, max_lon, max_lat = bbox_of_segment(steps, segment)
                utm_bbox, zone, hemi, epsg, fwd, inv = padded_utm_bbox(
                    min_lon,
                    min_lat,
                    max_lon,
                    max_lat,
                    padding=0.2,
                    max_cell_size=max_cell_size,
                )
                min_utm_x, min_utm_y, max_utm_x, max_utm_y = utm_bbox
                nx, ny = grid_shape_from_bbox(utm_bbox, self.resolution)

                if self.animal == Animal.AIRBORNE:
                    terrain = TerrainMapHandle.single_value(MesaLandcover.GRASSLAND, nx, ny)
                else:
                    min_lon_pad, min_lat_pad = inv.transform(min_utm_x, min_utm_y)
                    max_lon_pad, max_lat_pad = inv.transform(max_utm_x, max_utm_y)
                    terrain = TerrainMapHandle.landcover_to_discrete(
                        file_path=str(self.animal_proc.terrain_TIFFs[str(animal_id)]),
                        res_x=nx,
                        res_y=ny,
                        min_lon=min_lon_pad,
                        min_lat=min_lat_pad,
                        max_lon=max_lon_pad,
                        max_lat=max_lat_pad,
                    )

                cell_size = min(max((max_utm_x - min_utm_x) / max(nx, 1), 1.0), max_cell_size)

                try:
                    for st_idx in range(seg_start, seg_end):
                        print(
                            f"{animal_id} [{animal_index}|{len(steps_dict)}]: Processing step {st_idx}/{seg_end} ")
                        en_idx = st_idx + 1
                        start_lon = steps.geometry.iloc[st_idx].x
                        start_lat = steps.geometry.iloc[st_idx].y
                        end_lon = steps.geometry.iloc[en_idx].x
                        end_lat = steps.geometry.iloc[en_idx].y
                        start_time, end_time = steps.index[st_idx], steps.index[en_idx]
                        state = steps[self.kernel_state_col].iloc[st_idx]

                        st_utm_x, st_utm_y = fwd.transform(start_lon, start_lat)
                        en_utm_x, en_utm_y = fwd.transform(end_lon, end_lat)
                        start_x, start_y = utm_to_grid(nx, ny, utm_bbox, st_utm_x, st_utm_y)
                        end_x, end_y = utm_to_grid(nx, ny, utm_bbox, en_utm_x, en_utm_y)

                        if start_x == end_x and start_y == end_y:
                            animal_rows.append(
                                {
                                    id_col: animal_id,
                                    t_col: start_time,
                                    "geometry": Point(start_lon, start_lat),
                                    "segment_id": int(steps["segment_id"].iloc[st_idx]),
                                }
                            )
                            continue

                        T, S = movement_policy.resolve((start_x, start_y), (end_x, end_y), start_time, end_time)
                        T = int(np.ceil(T * 1.5))
                        directions = 1 if self.is_brownian else 8
                        base_kernel = py_kernels.get(state, next(iter(py_kernels.values())))
                        grid_kernel = kernel_for_grid(
                            base_kernel,
                            S,
                            cell_size,
                            self.rnge,
                            resample=WalkerHelper.resample_kernel_to_grid,
                            preserve_full_kernel=self.animal in (Animal.AIRBORNE, Animal.MARINE),
                        )
                        mapping = KernelMapping.from_state_kernel(
                            terrain,
                            grid_kernel,
                            directions,
                            forbidden_terrains=self.barriers,
                        )
                        context = KernelContextHandle.pool(terrain, mapping, Reachability.RELAXED)

                        try:
                            segment_points = MixedWalkBinding.single_state_walk(
                                context,
                                T,
                                start_x,
                                start_y,
                                end_x,
                                end_y,
                            )
                        finally:
                            context.free()
                            mapping.free()

                        if segment_points is None:
                            segment_points = [(start_x, start_y), (end_x, end_y)]
                        # plot_terrain_walk(terrain=terrain, walk=segment_points)
                        geo_walk = grid_to_geo_walk(segment_points, utm_bbox, nx, ny, inv)
                        times = pd.date_range(start=start_time, end=end_time, periods=len(geo_walk))
                        for (lon, lat), t in zip(geo_walk, times):
                            animal_rows.append(
                                {
                                    id_col: animal_id,
                                    t_col: t,
                                    "geometry": Point(lon, lat),
                                    "segment_id": int(steps["segment_id"].iloc[st_idx]),
                                }
                            )
                finally:
                    terrain.free()

            if animal_rows:
                animal_gdf = gpd.GeoDataFrame(animal_rows, geometry="geometry", crs="EPSG:4326")
                per_animal_gdfs.append(animal_gdf)

        if len(per_animal_gdfs) == 0:
            return mpd.TrajectoryCollection(gpd.GeoDataFrame(columns=["geometry"]), traj_id_col=id_col, t=t_col)

        combined_gdf = gpd.GeoDataFrame(pd.concat(per_animal_gdfs, ignore_index=True), geometry="geometry",
                                        crs="EPSG:4326")
        combined_gdf[t_col] = pd.to_datetime(combined_gdf[t_col])
        return merge_traj_collections(self.original_data, combined_gdf)

    def generate_state_timeline(self):
        pass


def _coerce_animal(animal_type):
    if isinstance(animal_type, Animal):
        return animal_type
    if isinstance(animal_type, IntEnum):
        return Animal(int(animal_type))
    if isinstance(animal_type, str):
        return Animal[animal_type.upper()]
    return Animal(animal_type)


__all__ = ["StateDependentWalker"]
