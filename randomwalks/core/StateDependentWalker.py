from __future__ import annotations

from enum import IntEnum
import json
import os
from pathlib import Path

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
from randomwalks.bindings.data_structures.Terrain import (
    _terrain_neighborhood_window,
    _trajectory_points_in_window,
    plot_terrain_neighborhood,
)
from randomwalks.bindings.data_structures.types import Reachability
from randomwalks.bindings.mixed_walk import MixedWalkBinding
from randomwalks.core.KernelFactory import StateAnnotationMethod
from randomwalks.core.MixedWalker import MixedWalker
from randomwalks.core.WalkerHelper import WalkerHelper
from randomwalks.bindings.data_structures.Kernels import kernel_array, kernel_for_grid, kernel_state_value, \
    normalize_kernel
from randomwalks.bindings.step_segments import segments_for_steps, terrain_pair_weights_from_neighborhoods
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
        self.state_kernel_metadata = {}
        self.annotation_result = None
        self.barrier_mode = BarrierMode.AVOID
        self.barriers = barriers

        self.is_brownian = False
        self.kernel_neighborhood_paths = []
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
            mass_percentile=0.99,
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
            mass_percentile=mass_percentile,
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
        state_kernel_metadata = {}
        for state_kernel in selected_kernels:
            state = kernel_state_value(state_kernel)
            kernel, metadata = _kernel_and_metadata(state_kernel, rnge, mass_percentile)
            if kernel is None or np.sum(kernel) == 0:
                continue
            state_kernels[state] = normalize_kernel(kernel)
            state_kernel_metadata[state] = metadata

        if not state_kernels:
            raise ValueError("No usable state kernels were generated.")
        self.state_kernels = state_kernels
        self.state_kernel_metadata = state_kernel_metadata
        return state_kernels

    def save_kernel_neighborhoods(
            self,
            kernels=None,
            *,
            out_dir=None,
            state_col=None,
            min_terrain_types=2,
            min_points=2,
            plot=True,
            save_matrix=True,
            mass_percentile=0.99,
    ):
        """
        Save terrain neighborhoods for trajectory steps after state kernels exist.

        For each trajectory point t, this uses the kernel belonging to the
        point's state, reads the terrain GeoTIFF in that kernel radius, and only
        saves the neighborhood when:
        - more than one terrain value is present in the neighborhood, and
        - at least min_points from t-1, t, and t+1 fall inside the neighborhood.
        """
        kernels = kernels if kernels is not None else self.state_kernels
        if not kernels:
            raise ValueError("Call get_kernels() before save_kernel_neighborhoods().")
        if self.animal_proc is None or self.animal_proc.traj is None:
            raise ValueError("Call annotate_behavior() before save_kernel_neighborhoods().")
        if not self.animal_proc.terrain_TIFFs:
            raise ValueError("No terrain GeoTIFFs are available.")

        state_col = state_col or self.kernel_state_col
        out_dir = Path(out_dir or Path(self.out_directory or ".") / "kernel_neighborhoods")
        out_dir.mkdir(parents=True, exist_ok=True)

        import matplotlib.pyplot as plt
        import rasterio

        saved = []
        for i, traj in enumerate(self.animal_proc.traj.trajectories):
            print(f"Saving neighborhoods for: {i}/{len(self.animal_proc.traj.trajectories)}\n")
            animal_id = str(traj.id)
            tiff_path = self.animal_proc.terrain_TIFFs.get(animal_id)
            if tiff_path is None:
                tiff_path = self.animal_proc.terrain_TIFFs.get(traj.id)
            if tiff_path is None:
                continue

            steps = traj.df
            if state_col not in steps.columns or "geometry" not in steps:
                continue

            steps_gdf = gpd.GeoDataFrame(steps.copy(), geometry="geometry", crs=getattr(steps, "crs", None) or self.crs)
            with rasterio.open(tiff_path) as src:
                tiff_steps = steps_gdf.to_crs(src.crs) if src.crs is not None else steps_gdf
                for step_index in range(len(tiff_steps)):
                    print(f"  {step_index}/{len(tiff_steps)}\n")
                    state = steps.iloc[step_index][state_col]
                    kernel = _kernel_for_state(kernels, state)
                    if kernel is None:
                        continue

                    kernel_metadata = _kernel_metadata_for_state(self.state_kernel_metadata, state)
                    radius_m = _kernel_radius_m(
                        kernel,
                        metadata=kernel_metadata,
                        fallback_radius=self.rnge,
                        mass_percentile=mass_percentile,
                    )
                    focal = tiff_steps.geometry.iloc[step_index]
                    if focal is None or focal.is_empty:
                        continue

                    candidate_indices = [
                        index
                        for index in (step_index - 1, step_index, step_index + 1)
                        if 0 <= index < len(tiff_steps)
                    ]
                    candidate_points = [
                        (tiff_steps.geometry.iloc[index].x, tiff_steps.geometry.iloc[index].y)
                        for index in candidate_indices
                        if tiff_steps.geometry.iloc[index] is not None
                           and not tiff_steps.geometry.iloc[index].is_empty
                    ]

                    focal_row, focal_col, window = _terrain_neighborhood_window(
                        src,
                        focal.x,
                        focal.y,
                        radius_m,
                        coordinate_space="world",
                    )
                    matrix = src.read(
                        1,
                        window=window,
                        boundless=True,
                        fill_value=src.nodata if src.nodata is not None else 0,
                    )
                    selected_points = _trajectory_points_in_window(
                        candidate_points,
                        src,
                        window,
                        "world",
                    )
                    terrain_values = _terrain_values(matrix, src.nodata)

                    if len(terrain_values) < min_terrain_types or len(selected_points) < min_points:
                        continue

                    obs_dx = None
                    obs_dy = None
                    observed_endpoint_pixel = None
                    if step_index + 1 < len(tiff_steps):
                        observed = tiff_steps.geometry.iloc[step_index + 1]
                        if observed is not None and not observed.is_empty:
                            observed_row, observed_col = src.index(observed.x, observed.y)
                            obs_dx = int(observed_col - focal_col)
                            obs_dy = int(observed_row - focal_row)
                            observed_endpoint_pixel = [
                                int(observed_col - window.col_off),
                                int(observed_row - window.row_off),
                            ]

                    stem = _safe_filename(f"{animal_id}_step_{step_index}_state_{state}_r{int(round(radius_m))}")
                    animal_dir = out_dir / _safe_filename(animal_id)
                    animal_dir.mkdir(parents=True, exist_ok=True)
                    matrix_path = animal_dir / f"{stem}.npy"
                    metadata_path = animal_dir / f"{stem}.json"
                    plot_path = animal_dir / f"{stem}.png"

                    if save_matrix:
                        np.save(matrix_path, matrix)

                    metadata = {
                        "animal_id": animal_id,
                        "step_index": int(step_index),
                        "state": _json_scalar(state),
                        "radius_m": float(radius_m),
                        "kernel_radius_cells": _json_scalar(
                            kernel_metadata.get("radius_cells")
                            if kernel_metadata
                            else None
                        ),
                        "kernel_retained_mass": _json_scalar(
                            kernel_metadata.get("retained_mass")
                            if kernel_metadata
                            else None
                        ),
                        "kernel_mass_percentile": _json_scalar(
                            kernel_metadata.get("mass_percentile")
                            if kernel_metadata
                            else mass_percentile
                        ),
                        "terrain_values": terrain_values,
                        "point_count": len(selected_points),
                        "candidate_step_indices": [int(index) for index in candidate_indices],
                        "focal_pixel": [
                            int(focal_col - window.col_off),
                            int(focal_row - window.row_off),
                        ],
                        "observed_endpoint_pixel": observed_endpoint_pixel,
                        "obs_dx": obs_dx,
                        "obs_dy": obs_dy,
                        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                        "tiff_path": str(tiff_path),
                        "matrix_path": str(matrix_path) if save_matrix else None,
                        "plot_path": str(plot_path) if plot else None,
                    }
                    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

                    if plot:
                        ax = plot_terrain_neighborhood(
                            tiff_path,
                            focal.x,
                            focal.y,
                            radius_m=radius_m,
                            trajectory_points=candidate_points,
                            coordinate_space="world",
                            show=False,
                        )
                        ax.set_title(
                            f"{animal_id} step {step_index}, state={state}, r={radius_m:g}m"
                        )
                        ax.figure.savefig(plot_path, bbox_inches="tight", dpi=150)
                        plt.close(ax.figure)

                    saved.append(metadata)

        self.kernel_neighborhood_paths = saved
        return saved

    def estimate_terrain_pair_weights(
            self,
            neighborhoods=None,
            kernels=None,
            *,
            out_dir=None,
            terrain_values=None,
            exclude_terrain_values=(0,),
            lambda_=1.0,
            log_clip=None,
            lo=0.5,
            hi=1.5,
            count_self_transitions=True,
            save_heatmaps=True,
            verbose=True,
    ):
        kernels = kernels if kernels is not None else self.state_kernels
        if not kernels:
            raise ValueError("Call get_kernels() before estimate_terrain_pair_weights().")

        neighborhoods = neighborhoods if neighborhoods is not None else self.kernel_neighborhood_paths
        if not neighborhoods:
            neighborhoods = self.save_kernel_neighborhoods(kernels=kernels, plot=False, save_matrix=True)

        out_dir = Path(out_dir or Path(self.out_directory or ".") / "terrain_pair_weights")
        return terrain_pair_weights_from_neighborhoods(
            neighborhoods,
            kernels,
            out_dir=out_dir,
            terrain_values=terrain_values,
            exclude_terrain_values=exclude_terrain_values,
            lambda_=lambda_,
            log_clip=log_clip,
            lo=lo,
            hi=hi,
            count_self_transitions=count_self_transitions,
            save_heatmaps=save_heatmaps,
            verbose=verbose,
        )

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


def _kernel_for_state(kernels, state):
    if state in kernels:
        return kernels[state]
    for key, kernel in kernels.items():
        if _state_key_equal(key, state):
            return kernel
    return None


def _kernel_metadata_for_state(metadata_by_state, state):
    if not metadata_by_state:
        return None
    if state in metadata_by_state:
        return metadata_by_state[state]
    for key, metadata in metadata_by_state.items():
        if _state_key_equal(key, state):
            return metadata
    return None


def _state_key_equal(left, right):
    if left == right:
        return True
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def _kernel_and_metadata(state_kernel, fallback_radius, mass_percentile):
    kernel = kernel_array(state_kernel)
    metadata = {
        "rnge": getattr(state_kernel, "rnge", None),
        "reso": getattr(state_kernel, "reso", None),
        "dx": getattr(state_kernel, "dx", None),
        "radius_cells": getattr(state_kernel, "radius_cells", None),
        "retained_mass": getattr(state_kernel, "retained_mass", None),
        "mass_percentile": getattr(state_kernel, "mass_percentile", None),
    }
    if metadata["rnge"] is not None and metadata["mass_percentile"] is not None:
        return kernel, metadata

    try:
        from kernelcma.postprocessing import clip_density_to_mass

        clipped = clip_density_to_mass(kernel, fallback_radius, mass_percentile=mass_percentile)
        metadata = {
            "rnge": clipped.rnge,
            "reso": clipped.reso,
            "dx": clipped.dx,
            "radius_cells": clipped.radius_cells,
            "retained_mass": clipped.retained_mass,
            "mass_percentile": clipped.mass_percentile,
        }
        return clipped.Z, metadata
    except Exception:
        return kernel, metadata


def _kernel_radius_m(kernel, *, metadata=None, fallback_radius=None, mass_percentile=0.99):
    if metadata and metadata.get("rnge") is not None:
        return float(metadata["rnge"])

    if fallback_radius is not None:
        try:
            from kernelcma.postprocessing import clip_density_to_mass

            return float(
                clip_density_to_mass(
                    kernel,
                    fallback_radius,
                    mass_percentile=mass_percentile,
                ).rnge
            )
        except Exception:
            pass

    kernel = np.asarray(kernel)
    if kernel.ndim >= 2 and min(kernel.shape[-2:]) > 1:
        return float(min(kernel.shape[-2], kernel.shape[-1]) // 2)
    return float(fallback_radius or 0)


def _terrain_values(matrix, nodata):
    values = np.unique(matrix)
    result = []
    for value in values:
        if nodata is not None and value == nodata:
            continue
        result.append(_json_scalar(value))
    return result


def _json_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _safe_filename(value):
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(value)
    ).strip("_")
    return safe or "value"


__all__ = ["StateDependentWalker"]
