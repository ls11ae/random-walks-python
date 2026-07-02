import os
import subprocess
import tempfile
from pathlib import Path

import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd

from randomwalks.bindings.data_structures.KernelContext import KernelContextHandle
from randomwalks.bindings.data_structures.KernelMapping import KernelMapping
from randomwalks.bindings.data_structures.Terrain import TerrainMapHandle
from randomwalks.bindings.data_structures.types import ComputationMode, Reachability
from randomwalks.bindings.mixed_walk import MixedWalkBinding
from randomwalks.bindings.plotter import plot_terrain_walk
from randomwalks.bindings.walk_visualization import LeafletGridOverlay, walk_to_osm
from randomwalks.core.WalkerHelper import WalkerHelper

WALK_BASE_ID_COL = "_rw_walk_base_id"
WALK_VERSION_COL = "_rw_walk_version"
WALK_TRAJ_ID_COL = "_rw_walk_id"


class MixedWalker:
    def __init__(
            self,
            data,
            resolution,
            out_directory,
            *,
            time_col="timestamp",
            lon_col="location-long",
            lat_col="location-lat",
            id_col="individual_local_identifier",
            crs="EPSG:4326",
            is_marine=False,
            movement_policy=None,
            reference_speed=None,
            reachability: Reachability = Reachability.RELAXED,
            context: ComputationMode = ComputationMode.KERNEL_POOL,
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
        self.reachability = reachability
        self.context_mode = context
        self._context_tempdir = None
        self.serialization_dir = os.path.join(self.out_directory, "serialization")
        self.utilization_distributions = {}
        self.utilization_distribution_paths = {}
        self.plot_paths = {}

    def _process_movebank_data(self, create_landcover=True):
        from randomwalks.core.AnimalMovement import AnimalMovementProcessor
        from environmentcma import traj_utm

        self.animal_proc = AnimalMovementProcessor(
            data=self.data,
            time_col=self.time_col,
            lon_col=self.lon_col,
            lat_col=self.lat_col,
            id_col=self.id_col,
            target_crs=self.crs,
            movement_policy=self.movement_policy,
            reference_speed=self.reference_speed,
            coerce_data=False,
        )
        if create_landcover:
            self.animal_proc.create_landcover_data_txt(
                resolution=self.resolution,
                is_marine=self.is_marine,
                out_directory=self.out_directory,
            )
        self.data = self.animal_proc.traj
        for traj in self.animal_proc.traj.trajectories:
            utm_traj = traj_utm(traj)
            traj.df["utm_x"] = utm_traj.df.geometry.x.to_numpy()
            traj.df["utm_y"] = utm_traj.df.geometry.y.to_numpy()
            traj.df["utm_crs"] = utm_traj.crs.to_string()
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

    def generate_walks(self, mapping=None, serialization_dir=None, amount=1, save_plots=False):
        amount = _validate_walk_amount(amount, allow_zero=False)
        return self._randomwalks_core(
            mapping,
            serialization_dir,
            amount_of_walks=amount,
            ud=False,
            save_plots=save_plots,
        )

    def generate_utilization_distribution(self, mapping=None, serialization_dir=None, sample_walks=0, save_plots=False):
        sample_walks = _validate_walk_amount(sample_walks, allow_zero=True, name="sample_walks")
        return self._randomwalks_core(
            mapping,
            serialization_dir,
            amount_of_walks=sample_walks,
            ud=True,
            save_plots=save_plots,
        )

    def _randomwalks_core(
            self,
            mapping=None,
            serialization_dir=None,
            amount_of_walks=1,
            ud=False,
            movement_policy=None,
            save_plots=False,
    ):
        amount_of_walks = _validate_walk_amount(amount_of_walks, allow_zero=ud, name="amount_of_walks")
        movement_policy = movement_policy or self.movement_policy
        self._process_movebank_data()
        steps_dict = {traj.id: traj.df for traj in self.animal_proc.traj.trajectories}
        per_animal_gdfs = []
        context_mode = ComputationMode.SERIALIZATION if serialization_dir is not None else self.context_mode
        self.utilization_distributions = {}
        self.utilization_distribution_paths = {}
        self.plot_paths = {}

        for animal_id, steps in steps_dict.items():
            print(f"Generating walks for {animal_id}")
            terrain_map = TerrainMapHandle.from_file(self._terrain_path(animal_id), delim=" ")
            mapping = mapping or KernelMapping.mesa_default()
            context = self._context_for_mapping(terrain_map, mapping, context_mode, serialization_dir)

            try:
                steps_df = steps
                idx = steps_df.index
                full_paths = [[] for _ in range(amount_of_walks)]
                segment_boundaries = [[0] for _ in range(amount_of_walks)]
                total = np.zeros((terrain_map.height, terrain_map.width), dtype=np.float64)

                for i in range(len(idx) - 1):
                    print(
                        f"[{i + 1}/{len(idx)}] Generating walk for {animal_id} at {idx[i]} - {idx[i + 1]}:")
                    start_x, start_y = int(steps["grid_x"].iloc[i]), int(steps["grid_y"].iloc[i])
                    end_x, end_y = int(steps["grid_x"].iloc[i + 1]), int(steps["grid_y"].iloc[i + 1])
                    sampled_segments = []

                    if start_x == end_x and start_y == end_y:
                        sampled_segments = [[(start_x, start_y)] for _ in range(amount_of_walks)]
                    else:
                        T = self._resolve_segment_T(
                            (start_x, start_y),
                            (end_x, end_y),
                            idx[i],
                            idx[i + 1],
                            movement_policy,
                        )
                        print(f"T: {T}")
                        print(f"Calculate forward density:")
                        forward_density = MixedWalkBinding.walk(
                            context,
                            T,
                            start_x,
                            start_y,
                        )
                        try:
                            for _ in range(amount_of_walks):
                                segment = MixedWalkBinding.backtrace(forward_density, context, end_x, end_y)
                                segment = _normalize_walk_segment(
                                    segment,
                                    (start_x, start_y),
                                    (end_x, end_y),
                                )
                                sampled_segments.append(segment)

                            if ud:
                                print(f"Calculate utilization distribution:")
                                segment_ud_handle = MixedWalkBinding.utilization_distribution(
                                    dp_matrix=forward_density,
                                    kernel_context=context,
                                    end_x=end_x,
                                    end_y=end_y,
                                )
                                try:
                                    segment_ud = segment_ud_handle.to_numpy_sum(
                                        terrain_map.width,
                                        terrain_map.height,
                                        average=True,
                                    )
                                except ValueError:
                                    segment_ud = None
                                finally:
                                    segment_ud_handle.free()

                                if segment_ud is not None and not np.isnan(segment_ud).any():
                                    total += segment_ud
                        finally:
                            forward_density.free()

                    for walk_index, segment in enumerate(sampled_segments):
                        full_paths[walk_index].extend(segment[:-1] if len(segment) > 1 else segment)
                        segment_boundaries[walk_index].append(len(full_paths[walk_index]))

                path_gdfs = []
                for walk_index, full_path in enumerate(full_paths):
                    final_gdf = self.animal_proc.movebank_path_to_gdf(
                        full_path,
                        steps_df,
                        animal_id,
                        idx,
                        segment_boundaries[walk_index],
                    )
                    if final_gdf is not None:
                        if amount_of_walks > 1:
                            final_gdf = _annotate_walk_version(final_gdf, animal_id, walk_index + 1)
                        path_gdfs.append(final_gdf)
                        per_animal_gdfs.append(final_gdf)

                if ud:
                    self.utilization_distributions[str(animal_id)] = total.copy()
                    self._save_utilization_distribution_map(animal_id, path_gdfs, total)
                if save_plots:
                    self._save_animal_plot(
                        animal_id,
                        terrain_map,
                        steps_df,
                        full_paths,
                        total if ud else None,
                        save_plots,
                    )
            finally:
                context.free()
                terrain_map.free()

        if ud:
            self._save_combined_utilization_distribution_map(per_animal_gdfs)

        if ud and amount_of_walks == 0:
            return self.data

        if len(per_animal_gdfs) == 0:
            traj_id_col = WALK_TRAJ_ID_COL if amount_of_walks > 1 else self.id_col
            empty_columns = [self.id_col, "time", "geometry"]
            if traj_id_col not in empty_columns:
                empty_columns.append(traj_id_col)
            empty = gpd.GeoDataFrame(columns=empty_columns, geometry="geometry", crs="EPSG:4326")
            return mpd.TrajectoryCollection(empty, traj_id_col=traj_id_col, t="time")

        combined = pd.concat(per_animal_gdfs, ignore_index=True)
        combined_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
        combined_gdf["time"] = pd.to_datetime(combined_gdf["time"])
        traj_id_col = WALK_TRAJ_ID_COL if amount_of_walks > 1 else self.id_col
        return mpd.TrajectoryCollection(combined_gdf, traj_id_col=traj_id_col, t="time")

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

    def _save_utilization_distribution_map(self, animal_id, path_gdfs, utilization_distribution):
        output_dir = Path(self.out_directory or ".")
        output_dir.mkdir(parents=True, exist_ok=True)

        key = str(animal_id)
        overlay = LeafletGridOverlay(
            grid=np.asarray(utilization_distribution, dtype=float),
            bounds=self.animal_proc.bbox_geo(animal_id),
            name=f"{key} utilization distribution",
        )
        coords_by_version = _coords_by_walk_version(key, path_gdfs)

        if not coords_by_version:
            min_lon, min_lat, max_lon, max_lat = self.animal_proc.bbox_geo(animal_id)
            center = [((min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0)]
            out_file = walk_to_osm(
                center,
                animal_id=key,
                walk_path=str(output_dir),
                map_filename=f"{key}_UD.html",
                utilization_distribution_overlays={"walk": overlay},
                draw_walk=False,
            )
            self.utilization_distribution_paths[key] = out_file
            print(f"Saved utilization distribution map to {out_file}")
            return Path(out_file)

        if len(coords_by_version) == 1:
            coords = next(iter(coords_by_version.values()))
            walk_data = coords
            overlays = {key: overlay}
        else:
            walk_data = coords_by_version
            overlays = {"walk": overlay}

        out_file = walk_to_osm(
            walk_data,
            animal_id=key,
            walk_path=str(output_dir),
            map_filename=f"{key}_UD.html",
            utilization_distribution_overlays=overlays,
        )
        self.utilization_distribution_paths[key] = out_file
        print(f"Saved utilization distribution map to {out_file}")
        return Path(out_file)

    def _save_combined_utilization_distribution_map(self, path_gdfs):
        print(f"save combined utilization distribution map:")
        combined = _combine_utilization_distributions(
            (
                (self.utilization_distributions[animal_id], self.animal_proc.bbox_geo(animal_id))
                for animal_id in self.utilization_distributions
            )
        )
        if combined is None:
            return None

        combined_grid, combined_bounds = combined
        overlay = LeafletGridOverlay(
            grid=combined_grid,
            bounds=combined_bounds,
            name="All animals utilization distribution",
        )
        walk_data = _primary_coords_by_animal(path_gdfs, self.id_col)

        output_dir = Path(self.out_directory or ".")
        output_dir.mkdir(parents=True, exist_ok=True)

        if walk_data:
            out_file = walk_to_osm(
                walk_data,
                walk_path=str(output_dir),
                map_filename="all_trajectories_walks_ud.html",
                utilization_distribution_overlays={"walk": overlay},
            )
        else:
            min_lon, min_lat, max_lon, max_lat = combined_bounds
            center = [((min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0)]
            out_file = walk_to_osm(
                center,
                animal_id="all",
                walk_path=str(output_dir),
                map_filename="all_trajectories_ud.html",
                utilization_distribution_overlays={"walk": overlay},
                draw_walk=False,
            )

        self.utilization_distribution_paths["all"] = out_file
        print(f"Saved combined utilization distribution map to {out_file}")
        return Path(out_file)

    def _save_animal_plot(self, animal_id, terrain_map, steps_df, full_paths, utilization_distribution, save_plots):
        output_dir = _plot_output_dir(save_plots, self.out_directory)
        if output_dir is None:
            return None

        key = str(animal_id)
        suffix = "ud_walks" if utilization_distribution is not None else "walks"
        out_file = output_dir / f"{_safe_filename(key)}_{suffix}.png"
        original_steps = _grid_steps_from_steps_df(steps_df)
        walk_paths = _plot_walk_paths(full_paths)
        walk = _plot_walk_argument(walk_paths, original_steps)

        plot_terrain_walk(
            terrain=terrain_map,
            walk=walk,
            steps=original_steps,
            ud=utilization_distribution,
            title=f"{key} utilization distribution" if utilization_distribution is not None else f"{key} walks",
            show=False,
            save_path=out_file,
        )
        self.plot_paths[key] = str(out_file)
        print(f"Saved walk plot to {out_file}")
        return out_file

    def _resolve_segment_T(self, start, end, start_time, end_time, movement_policy):
        """if movement_policy is not None:
            T, _ = movement_policy.resolve(
                start_point=start,
                end_point=end,
                start_time=start_time,
                end_time=end_time,
                reference_speed=self.reference_speed,
            )
            return int(T)"""

        manhattan = abs(start[0] - end[0]) + abs(start[1] - end[1])
        return 5 if manhattan < 5 else int(manhattan)

    @staticmethod
    def generate_custom_walks(
            terrain,
            steps,
            T,
            mappings=None,
            reachability: Reachability = Reachability.RELAXED,
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
            reachability: Reachability = Reachability.RELAXED,
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
        directory = self.serialization_dir
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


def _validate_walk_amount(amount, *, allow_zero, name="amount"):
    try:
        amount = int(amount)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    minimum = 0 if allow_zero else 1
    if amount < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return amount


def _plot_output_dir(save_plots, out_directory):
    if save_plots is False or save_plots is None:
        return None
    if save_plots is True:
        output_dir = Path(out_directory or ".") / "plots"
    else:
        output_dir = Path(save_plots)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _safe_filename(value):
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(value)
    ).strip("_")
    return safe or "animal"


def _grid_steps_from_steps_df(steps_df):
    if steps_df is None or len(steps_df) == 0:
        return []
    return [
        (int(x), int(y))
        for x, y in zip(steps_df["grid_x"], steps_df["grid_y"])
    ]


def _plot_walk_paths(full_paths):
    paths = []
    for path in full_paths or []:
        try:
            path = np.asarray(path, dtype=np.int64)
        except (TypeError, ValueError, OverflowError):
            continue
        if path.ndim != 2 or path.shape[1] != 2 or len(path) == 0:
            continue
        paths.append([(int(x), int(y)) for x, y in path])
    return paths


def _plot_walk_argument(walk_paths, fallback_steps):
    if walk_paths:
        return walk_paths[0] if len(walk_paths) == 1 else walk_paths
    if fallback_steps:
        return fallback_steps
    return None


def _walk_version_id(animal_id, version):
    return f"{animal_id}__walk_{int(version)}"


def _annotate_walk_version(gdf, animal_id, version):
    annotated = gdf.copy()
    annotated[WALK_BASE_ID_COL] = str(animal_id)
    annotated[WALK_VERSION_COL] = int(version)
    annotated[WALK_TRAJ_ID_COL] = _walk_version_id(animal_id, version)
    return annotated


def _fallback_segment(start, end):
    return [(int(start[0]), int(start[1])), (int(end[0]), int(end[1]))]


def _normalize_walk_segment(segment, start, end):
    fallback = _fallback_segment(start, end)
    if segment is None:
        return fallback

    try:
        segment = np.asarray(segment, dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        return fallback

    if segment.ndim != 2 or segment.shape[1] != 2 or len(segment) < 2:
        return fallback

    try:
        return [(int(x), int(y)) for x, y in segment]
    except (TypeError, ValueError, OverflowError):
        return fallback


def _coords_by_walk_version(animal_id, path_gdfs):
    if path_gdfs is None:
        return {}
    if isinstance(path_gdfs, gpd.GeoDataFrame):
        path_gdfs = [path_gdfs]

    coords_by_version = {}
    for index, path_gdf in enumerate(path_gdfs, start=1):
        coords = [
            (point.y, point.x)
            for point in path_gdf.geometry
            if point is not None and not point.is_empty
        ]
        if not coords:
            continue

        if WALK_VERSION_COL in path_gdf.columns and len(path_gdf) > 0:
            version = path_gdf[WALK_VERSION_COL].iloc[0]
        else:
            version = index
        label = str(animal_id) if len(path_gdfs) == 1 else f"{animal_id} v{version}"
        coords_by_version[label] = coords
    return coords_by_version


def _primary_coords_by_animal(path_gdfs, id_col):
    if path_gdfs is None:
        return {}
    if isinstance(path_gdfs, gpd.GeoDataFrame):
        path_gdfs = [path_gdfs]

    coords_by_animal = {}
    for path_gdf in path_gdfs:
        if path_gdf is None or len(path_gdf) == 0:
            continue
        if WALK_BASE_ID_COL in path_gdf.columns:
            animal_id = str(path_gdf[WALK_BASE_ID_COL].iloc[0])
        elif id_col in path_gdf.columns:
            animal_id = str(path_gdf[id_col].iloc[0])
        else:
            continue
        if animal_id in coords_by_animal:
            continue

        coords = [
            (point.y, point.x)
            for point in path_gdf.geometry
            if point is not None and not point.is_empty
        ]
        if coords:
            coords_by_animal[animal_id] = coords
    return coords_by_animal


def _combine_utilization_distributions(ud_items):
    prepared = []
    for grid, bounds in ud_items:
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 2:
            continue
        grid = np.clip(grid, 0.0, None)
        if not np.isfinite(grid).all() or grid.sum() <= 0:
            continue

        min_lon, min_lat, max_lon, max_lat = map(float, bounds)
        height, width = grid.shape
        if height <= 0 or width <= 0 or max_lon <= min_lon or max_lat <= min_lat:
            continue

        prepared.append(
            {
                "grid": grid,
                "bounds": (min_lon, min_lat, max_lon, max_lat),
                "dx": (max_lon - min_lon) / width,
                "dy": (max_lat - min_lat) / height,
            }
        )

    if not prepared:
        return None

    min_lon = min(item["bounds"][0] for item in prepared)
    min_lat = min(item["bounds"][1] for item in prepared)
    max_lon = max(item["bounds"][2] for item in prepared)
    max_lat = max(item["bounds"][3] for item in prepared)
    target_dx = min(item["dx"] for item in prepared if item["dx"] > 0)
    target_dy = min(item["dy"] for item in prepared if item["dy"] > 0)
    width = max(1, int(np.ceil((max_lon - min_lon) / target_dx)))
    height = max(1, int(np.ceil((max_lat - min_lat) / target_dy)))
    max_lon = min_lon + width * target_dx
    min_lat = max_lat - height * target_dy

    combined = np.zeros((height, width), dtype=np.float64)
    for item in prepared:
        source = _resample_ud_to_common_grid(
            item["grid"],
            item["bounds"],
            (min_lon, min_lat, max_lon, max_lat),
            (height, width),
        )
        source_total = source.sum()
        if source_total > 0:
            combined += source / source_total

    total = combined.sum()
    if total <= 0:
        return None
    return combined / total, (min_lon, min_lat, max_lon, max_lat)


def _resample_ud_to_common_grid(source, source_bounds, target_bounds, target_shape):
    src_min_lon, src_min_lat, src_max_lon, src_max_lat = source_bounds
    dst_min_lon, dst_min_lat, dst_max_lon, dst_max_lat = target_bounds
    dst_height, dst_width = target_shape
    src_height, src_width = source.shape

    dst_dx = (dst_max_lon - dst_min_lon) / dst_width
    dst_dy = (dst_max_lat - dst_min_lat) / dst_height
    src_dx = (src_max_lon - src_min_lon) / src_width
    src_dy = (src_max_lat - src_min_lat) / src_height

    col_start = max(0, int(np.floor((src_min_lon - dst_min_lon) / dst_dx)))
    col_stop = min(dst_width, int(np.ceil((src_max_lon - dst_min_lon) / dst_dx)))
    row_start = max(0, int(np.floor((dst_max_lat - src_max_lat) / dst_dy)))
    row_stop = min(dst_height, int(np.ceil((dst_max_lat - src_min_lat) / dst_dy)))

    target = np.zeros(target_shape, dtype=np.float64)
    if col_stop <= col_start or row_stop <= row_start:
        return target

    cols = np.arange(col_start, col_stop)
    rows = np.arange(row_start, row_stop)
    lons = dst_min_lon + (cols + 0.5) * dst_dx
    lats = dst_max_lat - (rows + 0.5) * dst_dy

    src_cols = np.floor((lons - src_min_lon) / src_dx).astype(int)
    src_rows = np.floor((src_max_lat - lats) / src_dy).astype(int)
    src_cols = np.clip(src_cols, 0, src_width - 1)
    src_rows = np.clip(src_rows, 0, src_height - 1)
    target[np.ix_(rows, cols)] = source[np.ix_(src_rows, src_cols)]
    return target


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
        segment = MixedWalkBinding.backtrace(dp, kernel_context, end_x, end_y)
        return _normalize_walk_segment(segment, (start_x, start_y), (end_x, end_y))
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
