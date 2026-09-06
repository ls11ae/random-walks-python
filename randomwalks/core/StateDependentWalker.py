from __future__ import annotations

import json
import os
from pathlib import Path

import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd
from environmentcma.crs import grid_shape_from_bbox, utm_to_grid, padded_utm_bbox
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
from randomwalks.bindings.mixed_walk import MixedWalkBinding
from randomwalks.bindings.plotter import plot_terrain_walk
from randomwalks.core.StateKernelHelper import (
    _is_unmodelled_state,
    _json_scalar,
    _kernel_and_metadata,
    _kernel_for_state,
    _kernel_is_clipped_to_mass,
    _kernel_metadata_for_state,
    _kernel_radius_m,
    _kernel_source_radius,
    _state_has_kernel,
    _state_key_equal,
    _state_for_step,
    _terrain_values,
)
from randomwalks.core.StateWalkGrid import (
    _as_epsg_crs,
    _cap_step_radius_to_grid,
    _coarsen_grid_to_cell_size,
    _grid_node_cell_size,
    _grid_node_edge_bounds,
    _grid_walk_to_geographic,
    _marine_rw_grid_terrain,
    _padded_projected_point_bounds,
)
from randomwalks.core.StateWalkerConfig import (
    UnmodelledStatePolicy,
    _coerce_animal,
    _default_barrier_mode,
    _default_barriers,
    _every_nth_point,
    _hard_barrier_endpoints,
    _interpolation_stride_folder,
    _ordinal_suffix,
    _reachability_for_segment,
    _resolve_barrier_mode,
    _state_walk_progress_message,
    _validate_interpolation_stride,
    _validate_max_time_steps,
    _validate_state_fill_gap,
)
from randomwalks.core.KernelFactory import StateAnnotationMethod
from randomwalks.core.MixedWalker import (
    WALK_TRAJ_ID_COL,
    MixedWalker,
    _annotate_walk_version,
    _validate_walk_amount,
)
from randomwalks.core.UtilizationDistribution import (
    UtilizationDistributionGrid,
    _safe_filename,
    combine_grid_utilization_distributions,
    utilization_distribution_from_forward_density,
)
from randomwalks.core.WalkerHelper import WalkerHelper
from randomwalks.bindings.data_structures.Kernels import kernel_state_value, normalize_kernel
from randomwalks.bindings.step_segments import segments_for_steps, terrain_pair_weights_from_neighborhoods
from randomwalks.move_apps_patch import apply_moveapps_id_dtype_patch, debug_patch_state, force_tc_id_object_inplace, \
    merge_traj_collections

DEFAULT_MAX_TIME_STEPS = 1000

_coerce_grid_walk = WalkerHelper.coerce_grid_walk
_normalize_grid_walk = WalkerHelper.normalize_grid_walk
_runtime_kernel = WalkerHelper.runtime_kernel
_validate_policy_resolution = WalkerHelper.validate_policy_resolution
_validate_rw_grid_coordinates = WalkerHelper.validate_grid_paths


class StateDependentWalker(MixedWalker):
    def __init__(self, data, animal_type, resolution, out_directory, movement_policy, barriers: list[int] | None = None,
                 time_col="timestamp", lon_col="location-long", lat_col="location-lat",
                 id_col="individual_local_identifier", crs="EPSG:4326", n=1, cuda=False):
        self.n = _validate_interpolation_stride(n)
        self.cuda = bool(cuda)
        self.interpolation_stride = self.n
        self.interpolation_point_counts = {}
        self.state_kernels = None
        self.animal = _coerce_animal(animal_type)
        self.n_hmm_states = 0
        self.dt_tolerance = 1
        self.rnge = 0
        self.kernel_state_col = "state"
        self.state_kernel_metadata = {}
        self.annotation_result = None
        self.runtime_kernel_resampling = []
        self.skipped_endpoint_pairs = []
        self.rw_grid_plot_paths = {}
        self.barrier_mode = _default_barrier_mode(self.animal)
        self.barriers = _default_barriers(self.animal) if barriers is None else list(barriers)

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
        self._configure_generation_output_directories()

    def _configure_generation_output_directories(self):
        output_root = Path(self.out_directory or ".")
        suffix = _interpolation_stride_folder(self.n)
        self.interpolation_output_suffix = suffix
        self.ud_plots_directory = output_root / "ud_plots"
        self.walks_directory = output_root / "walks"
        if suffix is not None:
            self.ud_plots_directory /= suffix
            self.walks_directory /= suffix

    def _ud_output_directory_for_run(self, save_plots):
        if save_plots in (False, None, True):
            return getattr(
                self,
                "ud_plots_directory",
                Path(getattr(self, "out_directory", ".") or ".") / "ud_plots",
            )
        output_dir = Path(save_plots)
        suffix = getattr(self, "interpolation_output_suffix", None)
        if suffix is not None:
            output_dir /= suffix
        return output_dir

    def annotate_behavior(
            self,
            method: StateAnnotationMethod | str = StateAnnotationMethod.HMM,
            features=None,
            num_states=3,
            penalty=10.0,
            plot_path=None,
    ):
        super()._process_movebank_data(create_landcover=True)

        if plot_path is None:
            states_directory = getattr(
                self,
                "states_directory",
                Path(self.out_directory or ".") / "states",
            )
            plot_path = states_directory / "states.png"
        if plot_path:
            Path(plot_path).parent.mkdir(parents=True, exist_ok=True)

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
            dt_tolerance=None,
            rnge=None,
            reso=None,
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
            dt_model_s=None,
            time_factor=None,
            kernel_config=None,
    ):
        if self.animal_proc is None or self.animal_proc.annotation_result is None:
            raise ValueError("Call annotate_behavior() before get_kernels().")
        if kernel_config is not None:
            if dt_tolerance is not None or rnge is not None or dt_model_s is not None or time_factor is not None:
                raise ValueError("Pass either kernel_config or individual kernel parameters, not both.")
            dt_tolerance = kernel_config.dt_tolerance
            rnge = kernel_config.range_m
            reso = kernel_config.resolution
            mass_percentile = kernel_config.mass_percentile
            dt_model_s = kernel_config.timestep_s
            time_factor = kernel_config.time_factor
            density_config = dict(kernel_config.density) or None
            is_brownian = kernel_config.is_brownian
        if dt_tolerance is None or rnge is None:
            raise ValueError("dt_tolerance and rnge are required without a KernelConfig.")
        self.dt_tolerance = dt_tolerance
        self.rnge = rnge
        self.is_brownian = is_brownian
        self.kernel_state_col = state_col

        if plot_dir is None:
            kernels_directory = getattr(
                self,
                "kernels_directory",
                Path(self.out_directory or ".") / "kernels",
            )
            plot_dir = kernels_directory / "kernels.png"
        if plot_dir:
            Path(plot_dir).parent.mkdir(parents=True, exist_ok=True)

        cor_zs, brw_zs = self.animal_proc.generate_state_kernels(
            state_col=state_col,
            dt_tolerance=self.dt_tolerance,
            rnge=self.rnge,
            reso=reso,
            out_dir=plot_dir or None,
            dt_model_s=dt_model_s,
            time_factor=time_factor,
            mass_percentile=mass_percentile,
            density_config=density_config,
            density_preset=density_preset,
            density_method=density_method,
            density_model=density_model,
            n_components=n_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            reg_covariance=reg_covariance,
            is_brownian=is_brownian,
        )
        selected_kernels = brw_zs if is_brownian else cor_zs
        state_kernels = {}
        state_kernel_metadata = {}
        for state_kernel in selected_kernels:
            state = kernel_state_value(state_kernel)
            kernel, metadata = _kernel_and_metadata(state_kernel, rnge, mass_percentile)
            if kernel is None or np.sum(kernel) == 0:
                continue
            if metadata.get("dt_model_s") is None and is_brownian and dt_model_s is not None:
                metadata["dt_model_s"] = float(dt_model_s)
            if metadata.get("dt_model_s") is None and not is_brownian:
                raise ValueError(
                    "Correlated kernel is missing its inferred native sampling interval."
                )
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
        steps = {}
        counts = {}
        for trajectory in self.animal_proc.traj.trajectories:
            full = trajectory.df
            sampled = _every_nth_point(full, self.n)
            steps[trajectory.id] = sampled
            counts[str(trajectory.id)] = {
                "full": len(full),
                "interpolation": len(sampled),
                "n": self.n,
            }
        self.interpolation_point_counts = counts
        return steps

    def generate_walks(
            self,
            mapping=None,
            amount=1,
            max_cell_size=10,
            barrier_mode: BarrierMode | None = None,
            unmodelled_state_policy: UnmodelledStatePolicy | str = UnmodelledStatePolicy.SKIP,
            max_state_fill_gap=None,
            max_time_steps=DEFAULT_MAX_TIME_STEPS,
            save_plots=True,
            save_intermediate_plots=False,
    ):
        self.barrier_mode = _resolve_barrier_mode(
            getattr(self, "animal", Animal.TERRESTRIAL),
            barrier_mode,
        )
        if not self.animal_proc.terrain_paths:
            self.animal_proc.create_landcover_data_txt(
                resolution=self.resolution,
                is_marine=self.is_marine,
                out_directory=self.out_directory,
            )
        return self._randomwalks_core(
            mapping=None,
            serialization_dir=self.serialization_dir,
            amount_of_walks=amount,
            ud=False,
            save_plots=save_plots,
            save_intermediate_plots=save_intermediate_plots,
            max_cell_size=max_cell_size,
            max_time_steps=max_time_steps,
            unmodelled_state_policy=unmodelled_state_policy,
            max_state_fill_gap=max_state_fill_gap,
        )

    def generate_utilization_distribution(
            self,
            mapping=None,
            serialization_dir=None,
            sample_walks=0,
            save_plots=True,
            max_cell_size=10,
            barrier_mode: BarrierMode | None = None,
            kernels=None,
            kernel_ranges=None,
            unmodelled_state_policy: UnmodelledStatePolicy | str = UnmodelledStatePolicy.SKIP,
            max_state_fill_gap=None,
            max_time_steps=DEFAULT_MAX_TIME_STEPS,
            save_intermediate_plots=False,
            combined_cell_size=None,
            target_cell_size=None,
            save_combined_map=True,
    ):
        """Generate state-dependent UDs on one common grid per animal.

        All state kernels are treated as physical density rasters. Internally
        fitted kernels use the per-state range and model-timestep metadata from
        ``get_kernels()``; external ``kernels`` require ``kernel_ranges`` in
        metres. The movement policy receives that physical range, the realized
        random-walk cell size, and the state model timestep, and returns the
        final unbounded ``(T, S)`` before interpolation. The complete kernel is
        then mapped to radius ``S`` without clipping or further adjustment.
        Endpoint pairs whose resolved ``T`` exceeds ``max_time_steps`` are
        skipped before either walk interpolation or UD calculation. Set the
        guard to ``None`` to disable it; the movement policy's ``T`` itself is
        never modified.
        Steps touching an unmodelled state are skipped by default. ``previous``
        uses the closest earlier modelled state only when
        ``max_state_fill_gap`` is not exceeded.
        Final UD maps and PNGs follow ``save_plots``. Per-segment RW-grid plots
        are separate and disabled unless ``save_intermediate_plots`` is true.
        ``combined_cell_size`` fixes the final per-animal raster resolution;
        leave it ``None`` for the historical finest-segment resolution.
        ``target_cell_size`` coarsens each native RW grid towards that physical
        node spacing (without exceeding ``max_cell_size``). This retains the
        physical kernel range while avoiding unnecessarily large runtime
        kernels and barrier contexts.
        ``save_combined_map=False`` skips only the additional all-animal HTML.
        """
        sample_walks = _validate_walk_amount(sample_walks, allow_zero=True, name="sample_walks")
        self.barrier_mode = _resolve_barrier_mode(
            getattr(self, "animal", Animal.TERRESTRIAL),
            barrier_mode,
        )
        return self._randomwalks_core(
            mapping=mapping,
            serialization_dir=serialization_dir,
            amount_of_walks=sample_walks,
            ud=True,
            save_plots=save_plots,
            save_intermediate_plots=save_intermediate_plots,
            max_cell_size=max_cell_size,
            max_time_steps=max_time_steps,
            kernels=kernels,
            kernel_ranges=kernel_ranges,
            unmodelled_state_policy=unmodelled_state_policy,
            max_state_fill_gap=max_state_fill_gap,
            combined_cell_size=combined_cell_size,
            target_cell_size=target_cell_size,
            save_combined_map=save_combined_map,
        )

    def _randomwalks_core(
            self,
            mapping=None,
            serialization_dir=None,
            amount_of_walks=1,
            ud=False,
            movement_policy=None,
            save_plots=False,
            save_intermediate_plots=False,
            max_cell_size=10,
            max_time_steps=DEFAULT_MAX_TIME_STEPS,
            kernels=None,
            kernel_ranges=None,
            unmodelled_state_policy: UnmodelledStatePolicy | str = UnmodelledStatePolicy.SKIP,
            max_state_fill_gap=None,
            combined_cell_size=None,
            target_cell_size=None,
            save_combined_map=True,
    ):
        external_kernels = kernels is not None
        py_kernels = self.state_kernels if kernels is None else kernels
        if not py_kernels:
            raise ValueError("State kernels are required; call get_kernels() or pass kernels by state.")
        if external_kernels and not kernel_ranges:
            raise ValueError("kernel_ranges by state are required for externally supplied physical kernels.")
        unmodelled_state_policy = UnmodelledStatePolicy(unmodelled_state_policy)
        max_state_fill_gap = _validate_state_fill_gap(unmodelled_state_policy, max_state_fill_gap)
        max_time_steps = _validate_max_time_steps(max_time_steps)
        if target_cell_size is not None:
            target_cell_size = float(target_cell_size)
            if not np.isfinite(target_cell_size) or target_cell_size <= 0:
                raise ValueError("target_cell_size must be positive and finite.")
            if target_cell_size > float(max_cell_size):
                raise ValueError("target_cell_size must not exceed max_cell_size.")
        self.runtime_kernel_resampling = []
        self.skipped_endpoint_pairs = []
        self.rw_grid_plot_paths = {}
        t_col = self.original_data.t
        id_col = self.original_data.get_traj_id_col()
        movement_policy = self.movement_policy

        steps_dict = self.__get_steps()
        per_animal_gdfs = []
        if ud:
            self._reset_utilization_distributions()

        animal_index = 0
        for animal_id, steps in steps_dict.items():
            animal_index += 1
            full_observed_points = self.animal_proc.traj.get_trajectory(animal_id).df
            segments = segments_for_steps(steps, max_cell_size=max_cell_size, resolution=self.resolution)
            endpoint_pair_count = max(0, len(steps) - 1)
            animal_rows_by_walk = [[] for _ in range(amount_of_walks)]
            animal_ud_grids = []

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
                segment_utm_points = [
                    fwd.transform(
                        steps.geometry.iloc[point_idx].x,
                        steps.geometry.iloc[point_idx].y,
                    )
                    for point_idx in range(seg_start, seg_end + 1)
                ]
                # Projected extrema can occur between the four geographic BBox
                # corners on long, highly distorted segments. Bound the actual
                # observations so no valid endpoint maps outside the RW grid.
                utm_bbox = _padded_projected_point_bounds(
                    segment_utm_points,
                    padding=0.2,
                    minimum_padding=max_cell_size,
                )
                min_utm_x, min_utm_y, max_utm_x, max_utm_y = utm_bbox
                nx, ny = grid_shape_from_bbox(utm_bbox, self.resolution)
                full_resolution_shape = (nx, ny)
                if target_cell_size is not None:
                    nx, ny = _coarsen_grid_to_cell_size(
                        utm_bbox,
                        nx,
                        ny,
                        target_cell_size,
                    )

                if self.is_marine:
                    min_lon_pad, min_lat_pad = inv.transform(min_utm_x, min_utm_y)
                    max_lon_pad, max_lat_pad = inv.transform(max_utm_x, max_utm_y)
                    terrain = _marine_rw_grid_terrain(
                        nx,
                        ny,
                        utm_bbox,
                        epsg,
                        (min_lon_pad, min_lat_pad, max_lon_pad, max_lat_pad),
                    )
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

                if target_cell_size is None:
                    # Preserve the historical grid convention unless explicit
                    # physical coarsening was requested.
                    cell_size = min(
                        max((max_utm_x - min_utm_x) / max(nx, 1), 1.0),
                        max_cell_size,
                    )
                else:
                    cell_size = _grid_node_cell_size(utm_bbox, nx, ny)
                    print(
                        f"RW grid {full_resolution_shape[0]}x{full_resolution_shape[1]} "
                        f"coarsened to {nx}x{ny} at {cell_size:g} m/cell "
                        f"(target {target_cell_size:g} m)"
                    )
                segment_ud = np.zeros((ny, nx), dtype=np.float32) if ud else None
                segment_plot_walks = []
                segment_grid_steps = []
                for point_utm_x, point_utm_y in segment_utm_points:
                    segment_grid_steps.append(
                        utm_to_grid(nx, ny, utm_bbox, point_utm_x, point_utm_y)
                    )
                # A state/S context is invariant across endpoint pairs on this
                # terrain segment. Reuse fitted and externally supplied kernels
                # alike instead of rebuilding the native kernel map each time.
                context_cache = {}
                terrain_values = {int(value) for value in terrain.unique_values()}
                segment_barriers = [
                    barrier for barrier in self.barriers
                    if int(barrier) in terrain_values
                ]
                segment_reachability = _reachability_for_segment(
                    self.barrier_mode,
                    segment_barriers,
                    terrain_values,
                )

                try:
                    for st_idx in range(seg_start, seg_end):
                        en_idx = st_idx + 1
                        start_lon = steps.geometry.iloc[st_idx].x
                        start_lat = steps.geometry.iloc[st_idx].y
                        end_lon = steps.geometry.iloc[en_idx].x
                        end_lat = steps.geometry.iloc[en_idx].y
                        start_time, end_time = steps.index[st_idx], steps.index[en_idx]
                        state = _state_for_step(
                            steps,
                            st_idx,
                            state_col=self.kernel_state_col,
                            kernels=py_kernels,
                            policy=unmodelled_state_policy,
                            max_fill_gap=max_state_fill_gap,
                        )
                        if state is None:
                            print(
                                _state_walk_progress_message(
                                    animal_id,
                                    animal_index,
                                    len(steps_dict),
                                    st_idx + 1,
                                    endpoint_pair_count,
                                    status="skipped: unmodelled state",
                                )
                            )
                            continue

                        st_utm_x, st_utm_y = segment_utm_points[st_idx - seg_start]
                        en_utm_x, en_utm_y = segment_utm_points[en_idx - seg_start]
                        start_x, start_y = utm_to_grid(nx, ny, utm_bbox, st_utm_x, st_utm_y)
                        end_x, end_y = utm_to_grid(nx, ny, utm_bbox, en_utm_x, en_utm_y)

                        barrier_endpoints = (
                            _hard_barrier_endpoints(
                                terrain,
                                segment_barriers,
                                (start_x, start_y),
                                (end_x, end_y),
                                self.barrier_mode,
                            )
                            if self.is_marine
                            else ()
                        )
                        if barrier_endpoints:
                            record = {
                                "animal_id": str(animal_id),
                                "endpoint_pair": int(st_idx + 1),
                                "segment": [int(seg_start), int(seg_end)],
                                "start_time": str(start_time),
                                "end_time": str(end_time),
                                "start_grid": [int(start_x), int(start_y)],
                                "end_grid": [int(end_x), int(end_y)],
                                "barrier_endpoints": list(barrier_endpoints),
                                "reason": "marine_hard_endpoint_on_barrier",
                            }
                            self.skipped_endpoint_pairs.append(record)
                            print(
                                _state_walk_progress_message(
                                    animal_id,
                                    animal_index,
                                    len(steps_dict),
                                    st_idx + 1,
                                    endpoint_pair_count,
                                    state=state,
                                    status=(
                                        "skipped: HARD marine endpoint on barrier "
                                        f"({', '.join(barrier_endpoints)})"
                                    ),
                                )
                            )
                            continue

                        directions = 1 if self.is_brownian else 8
                        base_kernel = _kernel_for_state(py_kernels, state)
                        if base_kernel is None:
                            raise KeyError(f"No kernel was supplied for state {state!r}.")
                        kernel_metadata = (
                            {}
                            if external_kernels
                            else (_kernel_metadata_for_state(self.state_kernel_metadata, state) or {})
                        )
                        kernel_range = (
                            _kernel_for_state(kernel_ranges, state)
                            if external_kernels
                            else _kernel_radius_m(
                                base_kernel,
                                metadata=kernel_metadata,
                                fallback_radius=self.rnge,
                            )
                        )
                        if kernel_range is None or not np.isfinite(kernel_range) or float(kernel_range) <= 0:
                            raise KeyError(f"No physical kernel range was supplied for state {state!r}.")
                        kernel_timestep_s = kernel_metadata.get("dt_model_s")
                        T, S = movement_policy.resolve_for_kernel(
                            (start_x, start_y),
                            (end_x, end_y),
                            start_time,
                            end_time,
                            state=state,
                            kernel=base_kernel,
                            kernel_range_m=float(kernel_range),
                            kernel_timestep_s=kernel_timestep_s,
                            cell_size_m=float(cell_size),
                        )
                        T, S = _validate_policy_resolution(T, S)
                        requested_T, requested_S = T, S
                        T, S = _cap_step_radius_to_grid(
                            T=T,
                            S=S,
                            start_point=(start_x, start_y),
                            end_point=(end_x, end_y),
                            width=nx,
                            height=ny,
                        )

                        same_grid_cell = start_x == end_x and start_y == end_y
                        exceeds_time_guard = (
                                max_time_steps is not None and T > max_time_steps
                        )
                        progress_status = None
                        if same_grid_cell:
                            progress_status = "same grid cell; no walk"
                        elif exceeds_time_guard:
                            progress_status = (
                                f"skipped: T exceeds max_time_steps={max_time_steps}"
                            )
                        print(
                            _state_walk_progress_message(
                                animal_id,
                                animal_index,
                                len(steps_dict),
                                st_idx + 1,
                                endpoint_pair_count,
                                state=state,
                                S=S,
                                T=T,
                                status=progress_status,
                            )
                        )
                        if same_grid_cell:
                            segment_plot_walks.extend(
                                [[(start_x, start_y)] for _ in range(max(1, amount_of_walks))]
                            )
                            if ud and amount_of_walks == 0:
                                continue
                            for animal_rows in animal_rows_by_walk:
                                animal_rows.append({
                                    id_col: animal_id,
                                    t_col: start_time,
                                    "geometry": Point(start_lon, start_lat),
                                    "segment_id": int(steps["segment_id"].iloc[st_idx]),
                                })
                            continue
                        if exceeds_time_guard:
                            continue

                        cache_key = (str(state), int(S), directions)
                        cached_context = context_cache.get(cache_key)
                        if cached_context is None:
                            grid_kernel = _runtime_kernel(
                                base_kernel,
                                S,
                                cell_size,
                                self.rnge,
                                externally_supplied=external_kernels,
                            )
                            resampling_record = {
                                "animal_id": str(animal_id),
                                "segment": [int(seg_start), int(seg_end)],
                                "state": _json_scalar(state),
                                "range_m": float(kernel_range),
                                "model_timestep_s": (
                                    None if kernel_timestep_s is None else float(kernel_timestep_s)
                                ),
                                "rw_cell_size_m": float(cell_size),
                                "rw_target_cell_size_m": (
                                    None if target_cell_size is None else float(target_cell_size)
                                ),
                                "requested_time_steps": int(requested_T),
                                "time_steps": int(T),
                                "time_steps_were_increased": bool(T != requested_T),
                                "requested_step_radius_cells": int(requested_S),
                                "step_radius_cells": int(S),
                                "step_radius_was_capped": bool(S != requested_S),
                                "source_shape": list(np.shape(base_kernel)),
                                "runtime_shape": list(np.shape(grid_kernel)),
                            }
                            self.runtime_kernel_resampling.append(resampling_record)
                            print(
                                f"State {state}: mapped physical kernel {np.shape(base_kernel)} "
                                f"over {float(kernel_range):g} m to RW grid {np.shape(grid_kernel)} "
                                f"at {float(cell_size):g} m/cell (S={int(S)}, T={int(T)})"
                            )
                            mapping = KernelMapping.from_state_kernel(
                                terrain,
                                grid_kernel,
                                directions,
                                forbidden_terrains=segment_barriers,
                            )
                            try:
                                context = KernelContextHandle.pool(
                                    terrain,
                                    mapping,
                                    segment_reachability,
                                )
                            except Exception:
                                mapping.free()
                                raise
                            context_cache[cache_key] = (mapping, context)
                        else:
                            mapping, context = cached_context

                        sampled_segment_points = []
                        if ud:
                            forward_density = MixedWalkBinding.walk(
                                context, T, start_x, start_y, cuda=self.cuda
                            )
                            try:
                                step_ud = utilization_distribution_from_forward_density(
                                    forward_density,
                                    context,
                                    nx,
                                    ny,
                                    end_x,
                                    end_y,
                                    cuda=self.cuda,
                                )
                                if step_ud is not None:
                                    segment_ud += step_ud
                                for _ in range(amount_of_walks):
                                    sampled_segment_points.append(MixedWalkBinding.backtrace(
                                        forward_density,
                                        context,
                                        end_x,
                                        end_y,
                                    ))
                                if amount_of_walks == 0 and save_intermediate_plots:
                                    sampled_segment_points.append(MixedWalkBinding.backtrace(
                                        forward_density,
                                        context,
                                        end_x,
                                        end_y,
                                    ))
                            finally:
                                forward_density.free()
                        else:
                            for _ in range(amount_of_walks):
                                if self.cuda:
                                    forward_density = MixedWalkBinding.walk(
                                        context, T, start_x, start_y, cuda=True
                                    )
                                    try:
                                        sampled_segment_points.append(MixedWalkBinding.backtrace(
                                            forward_density,
                                            context,
                                            end_x,
                                            end_y,
                                        ))
                                    finally:
                                        forward_density.free()
                                else:
                                    sampled_segment_points.append(
                                        MixedWalkBinding.single_state_walk(
                                            context,
                                            T,
                                            start_x,
                                            start_y,
                                            end_x,
                                            end_y,
                                        )
                                    )

                        if ud and amount_of_walks == 0:
                            for segment_points in sampled_segment_points:
                                plot_walk = _coerce_grid_walk(segment_points)
                                if plot_walk is not None:
                                    segment_plot_walks.append(plot_walk)
                            continue
                        for walk_index, segment_points in enumerate(sampled_segment_points):
                            segment_points = _normalize_grid_walk(
                                segment_points,
                                (start_x, start_y),
                                (end_x, end_y),
                            )
                            segment_plot_walks.append(segment_points)
                            geo_walk = _grid_walk_to_geographic(segment_points, utm_bbox, nx, ny, inv)
                            times = pd.date_range(start=start_time, end=end_time, periods=len(geo_walk))
                            for (lon, lat), t in zip(geo_walk, times):
                                animal_rows_by_walk[walk_index].append(
                                    {
                                        id_col: animal_id,
                                        t_col: t,
                                        "geometry": Point(lon, lat),
                                        "segment_id": int(steps["segment_id"].iloc[st_idx]),
                                    }
                                )

                    if save_intermediate_plots:
                        self._save_rw_grid_plots(
                            animal_id,
                            segment,
                            terrain=terrain,
                            walks=segment_plot_walks,
                            steps=segment_grid_steps,
                            utilization_distribution=segment_ud,
                            save_plots=(
                                save_plots
                                if save_plots not in (False, None)
                                else True
                            ),
                        )
                finally:
                    for mapping, context in context_cache.values():
                        context.free()
                        mapping.free()
                    terrain.free()

                if ud and segment_ud.sum() > 0:
                    ud_bounds = _grid_node_edge_bounds(utm_bbox, nx, ny)
                    animal_ud_grids.append(
                        UtilizationDistributionGrid(
                            segment_ud,
                            ud_bounds,
                            cell_size=(
                                (ud_bounds[2] - ud_bounds[0]) / nx,
                                (ud_bounds[3] - ud_bounds[1]) / ny,
                            ),
                            crs=_as_epsg_crs(epsg),
                        )
                    )

            animal_path_gdfs = []
            for walk_index, animal_rows in enumerate(animal_rows_by_walk):
                if not animal_rows:
                    continue
                animal_gdf = gpd.GeoDataFrame(animal_rows, geometry="geometry", crs="EPSG:4326")
                if amount_of_walks > 1:
                    animal_gdf = _annotate_walk_version(animal_gdf, animal_id, walk_index + 1)
                per_animal_gdfs.append(animal_gdf)
                animal_path_gdfs.append(animal_gdf)

            if ud:
                combined_ud = combine_grid_utilization_distributions(
                    animal_ud_grids,
                    target_cell_size=combined_cell_size,
                )
                # The combined grid owns its data; release all segment grids
                # before creating maps or benchmark serialization copies.
                animal_ud_grids.clear()
                if combined_ud is not None:
                    combined_ud = self._store_utilization_distribution(animal_id, combined_ud)
                    if save_plots:
                        ud_output_dir = self._ud_output_directory_for_run(save_plots)
                        self._save_utilization_distribution_map(
                            animal_id,
                            animal_path_gdfs,
                            combined_ud,
                            output_dir=ud_output_dir,
                            observed_points=full_observed_points,
                        )
                        self._save_utilization_distribution_png(
                            animal_id,
                            observed_points=full_observed_points,
                            utilization_distribution=combined_ud,
                            save_plots=ud_output_dir,
                            smoothing_metres=max_cell_size,
                        )

        if ud and save_plots and save_combined_map:
            self._save_combined_utilization_distribution_map(
                per_animal_gdfs,
                output_dir=self._ud_output_directory_for_run(save_plots),
                observed_points=self.animal_proc.traj.to_point_gdf(),
            )

        if ud and amount_of_walks == 0:
            return self.original_data

        if len(per_animal_gdfs) == 0:
            result_id_col = WALK_TRAJ_ID_COL if amount_of_walks > 1 else id_col
            columns = [id_col, t_col, "geometry"]
            if result_id_col not in columns:
                columns.append(result_id_col)
            empty = gpd.GeoDataFrame(columns=columns, geometry="geometry", crs="EPSG:4326")
            return mpd.TrajectoryCollection(empty, traj_id_col=result_id_col, t=t_col)

        combined_gdf = gpd.GeoDataFrame(pd.concat(per_animal_gdfs, ignore_index=True), geometry="geometry",
                                        crs="EPSG:4326")
        combined_gdf[t_col] = pd.to_datetime(combined_gdf[t_col])
        if amount_of_walks > 1:
            return mpd.TrajectoryCollection(
                combined_gdf,
                traj_id_col=WALK_TRAJ_ID_COL,
                t=t_col,
                crs="EPSG:4326",
            )
        return merge_traj_collections(self.original_data, combined_gdf)

    def _save_rw_grid_plots(
            self,
            animal_id,
            segment,
            *,
            terrain,
            walks,
            steps,
            utilization_distribution,
            save_plots=True,
    ):
        """Save Matplotlib walk and UD plots against the actual local RW terrain grid."""
        if not save_plots:
            return []

        output_dir = self._ud_output_directory_for_run(save_plots)
        output_dir.mkdir(parents=True, exist_ok=True)
        seg_start, seg_end = map(int, segment)
        stem = f"{_safe_filename(animal_id)}_rw_grid_{seg_start}_{seg_end}"
        saved = []
        terrain_shape = (terrain.height, terrain.width)
        _validate_rw_grid_coordinates(walks, terrain.width, terrain.height, "walk")
        _validate_rw_grid_coordinates([steps], terrain.width, terrain.height, "observed step")

        walk_path = output_dir / f"{stem}_walks.png"
        plot_terrain_walk(
            terrain=terrain,
            walk=(None if not walks else walks[0] if len(walks) == 1 else walks),
            steps=steps,
            title=f"{animal_id} random walks (points {seg_start}-{seg_end})",
            show=False,
            save_path=walk_path,
        )
        saved.append(walk_path)

        ud_array = None if utilization_distribution is None else np.asarray(utilization_distribution)
        if ud_array is not None and ud_array.shape != terrain_shape:
            raise ValueError(
                f"RW-grid UD shape {ud_array.shape} does not match terrain shape {terrain_shape}."
            )
        if ud_array is not None and ud_array.size and np.isfinite(ud_array).all():
            ud_path = output_dir / f"{stem}_UD.png"
            plot_terrain_walk(
                terrain=terrain,
                steps=steps,
                ud=ud_array,
                title=f"{animal_id} utilization distribution (points {seg_start}-{seg_end})",
                show=False,
                save_path=ud_path,
            )
            saved.append(ud_path)

        key = str(animal_id)
        self.rw_grid_plot_paths.setdefault(key, []).extend(str(path) for path in saved)
        for path in saved:
            print(f"Saved RW-grid plot to {path}")
        return saved


__all__ = ["StateDependentWalker", "UnmodelledStatePolicy"]
