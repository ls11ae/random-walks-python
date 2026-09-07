from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np

from randomwalks.bindings.mixed_walk import MixedWalkBinding
from randomwalks.bindings.walk_visualization import LeafletGridOverlay, walk_to_osm


@dataclass(frozen=True)
class UtilizationDistributionGrid:
    """A utilization distribution together with its spatial grid metadata."""

    grid: np.ndarray
    bounds: tuple[float, float, float, float]
    cell_size: float | tuple[float, float] | None = None
    crs: object | None = None


class UtilizationDistributionMixin:
    """Shared UD storage and map output used by the walker implementations."""

    def _initialize_utilization_distributions(self):
        self.utilization_distributions = {}
        self.utilization_distribution_paths = {}
        self.utilization_distribution_png_paths = {}
        self.utilization_distribution_bounds = {}
        self.utilization_distribution_cell_sizes = {}
        self.utilization_distribution_crs = {}

    def _reset_utilization_distributions(self):
        self._initialize_utilization_distributions()

    def _store_utilization_distribution(self, animal_id, distribution):
        distribution = _coerce_grid(distribution)
        key = str(animal_id)
        # UDs are normalized later with float64 accumulation. Keeping their
        # large resident rasters as contiguous float32 halves storage and does
        # not create a second copy when the combiner already returned float32.
        grid = np.ascontiguousarray(distribution.grid, dtype=np.float32)
        self.utilization_distributions[key] = grid
        self.utilization_distribution_bounds[key] = distribution.bounds
        self.utilization_distribution_cell_sizes[key] = _canonical_cell_size(_cell_size(distribution))
        self.utilization_distribution_crs[key] = distribution.crs
        return UtilizationDistributionGrid(
            grid,
            distribution.bounds,
            distribution.cell_size,
            distribution.crs,
        )

    def _utilization_distribution_for(self, animal_id):
        key = str(animal_id)
        return UtilizationDistributionGrid(
            grid=self.utilization_distributions[key],
            bounds=self.utilization_distribution_bounds[key],
            cell_size=self.utilization_distribution_cell_sizes[key],
            crs=self.utilization_distribution_crs[key],
        )

    def _save_utilization_distribution_map(
            self,
            animal_id,
            path_gdfs,
            utilization_distribution=None,
            output_dir=None,
            observed_points=None,
    ):
        key = str(animal_id)
        if utilization_distribution is None:
            distribution = self._utilization_distribution_for(key)
        elif isinstance(utilization_distribution, UtilizationDistributionGrid):
            distribution = utilization_distribution
        else:
            distribution = UtilizationDistributionGrid(
                grid=np.asarray(utilization_distribution, dtype=float),
                bounds=self.animal_proc.bbox_geo(animal_id),
                crs="EPSG:4326",
            )

        display_grid, display_bounds = _leaflet_grid(distribution)
        output_dir = Path(output_dir or Path(self.out_directory or ".") / "ud_plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        overlay = LeafletGridOverlay(
            grid=display_grid,
            bounds=display_bounds,
            name=f"{key} utilization distribution",
        )
        coords_by_version = _coords_by_walk_version(key, path_gdfs)
        observed_coords = _leaflet_coords_from_points(observed_points)
        coastline_geojson = (
            _marine_coastline_geojson(display_bounds)
            if getattr(self, "is_marine", False)
            else None
        )

        if not coords_by_version:
            min_lon, min_lat, max_lon, max_lat = display_bounds
            center = [((min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0)]
            out_file = walk_to_osm(
                center,
                animal_id=key,
                walk_path=str(output_dir),
                map_filename=f"{key}_UD.html",
                utilization_distribution_overlays={"walk": overlay},
                draw_walk=False,
                observed_coords=observed_coords,
                coastline_geojson=coastline_geojson,
            )
        else:
            if len(coords_by_version) == 1:
                walk_data = next(iter(coords_by_version.values()))
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
                observed_coords=observed_coords,
                coastline_geojson=coastline_geojson,
            )

        self.utilization_distribution_paths[key] = out_file
        print(f"Saved utilization distribution map to {out_file}")
        return Path(out_file)

    def _save_utilization_distribution_png(
            self,
            animal_id,
            observed_points=None,
            utilization_distribution=None,
            save_plots=True,
            smoothing_metres=None,
    ):
        """Save an adehabitatHR-style volume-UD PNG for one animal.

        The optional smoothing is applied only to this display. It does not
        modify the utilization distribution stored on the walker.
        """
        output_dir = _png_output_directory(save_plots, getattr(self, "out_directory", None))
        if output_dir is None:
            return None

        key = str(animal_id)
        if utilization_distribution is None:
            distribution = self._utilization_distribution_for(key)
        else:
            distribution = _coerce_grid(utilization_distribution)
        grid = np.clip(np.asarray(distribution.grid, dtype=np.float32), 0.0, None)
        if grid.ndim != 2 or not np.isfinite(grid).all() or grid.sum() <= 0:
            return None
        grid /= grid.sum()

        sigma = _display_smoothing_sigma(distribution, smoothing_metres)
        if sigma is not None:
            from scipy.ndimage import gaussian_filter
            grid = gaussian_filter(grid, sigma=sigma, mode="constant")
            if grid.sum() > 0:
                grid /= grid.sum()
        volume = _volume_percent_grid(grid)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        min_x, min_y, max_x, max_y = distribution.bounds
        height, width = grid.shape
        cell_size_x = (max_x - min_x) / width
        cell_size_y = (max_y - min_y) / height
        x = min_x + (np.arange(width) + 0.5) * cell_size_x
        y = max_y - (np.arange(height) + 0.5) * cell_size_y

        fig, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
        axis.imshow(
            volume,
            extent=(min_x, max_x, min_y, max_y),
            origin="upper",
            cmap="inferno_r",
            vmin=0.0,
            vmax=100.0,
            interpolation="nearest",
        )
        contour = axis.contour(
            x,
            y,
            volume,
            levels=(50.0, 95.0),
            colors=("white", "cyan"),
            linewidths=(1.2, 1.8),
        )
        axis.clabel(contour, fmt="%d", fontsize=7)

        coastline_drawn = False
        if getattr(self, "is_marine", False):
            coastline = _marine_land_for_distribution(distribution)
            if coastline is not None and not coastline.empty:
                coastline.boundary.plot(
                    ax=axis,
                    color="#66A64B",
                    linewidth=1.2,
                    zorder=4,
                )
                coastline_drawn = True

        points = _points_in_distribution_crs(observed_points, distribution.crs)
        if points is not None and len(points):
            finite = np.isfinite(np.column_stack((points.geometry.x, points.geometry.y))).all(axis=1)
            points = points.loc[finite]
            axis.scatter(
                points.geometry.x,
                points.geometry.y,
                s=9,
                c="white",
                edgecolors="black",
                alpha=0.9,
                linewidths=0.25,
                zorder=5,
            )

        handles = [
            Line2D([0], [0], color="white", linewidth=1.2, label="50% isopleth"),
            Line2D([0], [0], color="cyan", linewidth=1.8, label="95% isopleth"),
        ]
        if coastline_drawn:
            handles.append(Line2D(
                [0], [0], color="#66A64B", linewidth=1.2,
                label="coastline (marine barrier)",
            ))
        if points is not None and len(points):
            handles.append(Line2D(
                [0], [0], marker="o", linestyle="none", color="white",
                markersize=4, label="observed fixes",
            ))
        axis.legend(handles=handles, loc="upper left", frameon=False, labelcolor="white")
        title = f"{key} utilization distribution"
        if sigma is not None:
            title += f" ({float(smoothing_metres):g} m display smoothing)"
        axis.set(
            title=title,
            xlabel="Easting" if not _is_geographic_crs(distribution.crs) else "Longitude",
            ylabel="Northing" if not _is_geographic_crs(distribution.crs) else "Latitude",
            xlim=(min_x, max_x),
            ylim=(min_y, max_y),
            aspect="equal",
        )
        axis.ticklabel_format(style="plain", axis="both", useOffset=False)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{_safe_filename(key)}_UD.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        self.utilization_distribution_png_paths[key] = str(output_path)
        print(f"Saved utilization distribution PNG to {output_path}")
        return output_path

    def _save_combined_utilization_distribution_map(
            self,
            path_gdfs,
            output_dir=None,
            observed_points=None,
    ):
        print("save combined utilization distribution map:")
        display_distributions = []
        for animal_id in self.utilization_distributions:
            grid, bounds = _leaflet_grid(self._utilization_distribution_for(animal_id))
            display_distributions.append((grid, bounds))
        combined = combine_utilization_distributions(display_distributions)
        if combined is None:
            return None

        combined_grid, combined_bounds = combined
        overlay = LeafletGridOverlay(
            grid=combined_grid,
            bounds=combined_bounds,
            name="All animals utilization distribution",
        )
        walk_data = _primary_coords_by_animal(path_gdfs, self.id_col)
        output_dir = Path(output_dir or Path(self.out_directory or ".") / "ud_plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        observed_coords = _leaflet_coords_from_points(observed_points)
        coastline_geojson = (
            _marine_coastline_geojson(combined_bounds)
            if getattr(self, "is_marine", False)
            else None
        )

        if walk_data:
            out_file = walk_to_osm(
                walk_data,
                walk_path=str(output_dir),
                map_filename="all_trajectories_walks_ud.html",
                utilization_distribution_overlays={"walk": overlay},
                observed_coords=observed_coords,
                coastline_geojson=coastline_geojson,
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
                observed_coords=observed_coords,
                coastline_geojson=coastline_geojson,
            )

        self.utilization_distribution_paths["all"] = out_file
        print(f"Saved combined utilization distribution map to {out_file}")
        return Path(out_file)


def utilization_distribution_from_forward_density(
        forward_density,
        kernel_context,
        width,
        height,
        end_x,
        end_y,
        *,
        cuda=False,
):
    """Convert a forward-density tensor into a finite 2D UD, if possible."""
    ud_handle = MixedWalkBinding.utilization_distribution_sum(
        dp_matrix=forward_density,
        kernel_context=kernel_context,
        end_x=end_x,
        end_y=end_y,
        cuda=cuda,
    )
    try:
        try:
            distribution = ud_handle.to_numpy()
        except ValueError:
            return None
    finally:
        ud_handle.free()

    distribution = np.asarray(distribution, dtype=np.float32)
    if distribution.shape != (height, width) or not np.isfinite(distribution).all():
        return None
    return distribution


def combine_grid_utilization_distributions(
        distributions,
        *,
        target_cell_size=None,
        target_crs=None,
        normalize_sources=False,
        normalize_result=False,
        dtype=np.float32,
        resample_chunk_rows=256,
):
    """Resample and add spatial UDs without full-grid coordinate temporaries.

    ``target_cell_size`` should be set for large, heterogeneous segment grids;
    otherwise the historical finest-input-cell behavior is retained. Raster
    values default to float32 while totals and normalization use float64.
    """
    prepared = []
    for distribution in distributions:
        try:
            distribution = _coerce_grid(distribution)
        except (TypeError, ValueError):
            continue
        grid = np.asarray(distribution.grid)
        if (
            grid.ndim != 2
            or not np.issubdtype(grid.dtype, np.number)
            or not np.isfinite(grid).all()
            or _positive_sum(grid) <= 0
        ):
            continue
        if not _valid_bounds(distribution.bounds):
            continue
        prepared.append(UtilizationDistributionGrid(
            grid,
            distribution.bounds,
            distribution.cell_size,
            distribution.crs,
        ))

    if not prepared:
        return None

    target_crs = target_crs if target_crs is not None else prepared[0].crs
    transformed_bounds = [_transform_bounds(item.bounds, item.crs, target_crs) for item in prepared]
    if target_cell_size is None:
        sizes = [_cell_size_in_crs(item, target_crs) for item in prepared]
        target_dx = min(size[0] for size in sizes)
        target_dy = min(size[1] for size in sizes)
    else:
        target_dx, target_dy = _size_pair(target_cell_size)

    min_x = min(bounds[0] for bounds in transformed_bounds)
    min_y = min(bounds[1] for bounds in transformed_bounds)
    max_x = max(bounds[2] for bounds in transformed_bounds)
    max_y = max(bounds[3] for bounds in transformed_bounds)
    width = max(1, int(np.ceil((max_x - min_x) / target_dx)))
    height = max(1, int(np.ceil((max_y - min_y) / target_dy)))
    target_bounds = (min_x, max_y - height * target_dy, min_x + width * target_dx, max_y)
    combined = np.zeros((height, width), dtype=dtype)

    for item in prepared:
        _add_resampled_utilization_distribution(
            combined,
            item,
            target_bounds=target_bounds,
            target_crs=target_crs,
            preserve_mass=True,
            normalize=normalize_sources,
            chunk_rows=resample_chunk_rows,
        )

    total = combined.sum(dtype=np.float64)
    if total <= 0:
        return None
    if normalize_result:
        combined /= total
    return UtilizationDistributionGrid(
        combined,
        target_bounds,
        _canonical_cell_size((target_dx, target_dy)),
        target_crs,
    )


def combine_utilization_distributions(ud_items):
    """Backward-compatible equal-weight combination used for Leaflet overlays."""
    combined = combine_grid_utilization_distributions(
        ud_items,
        normalize_sources=True,
        normalize_result=True,
    )
    if combined is None:
        return None
    return combined.grid, combined.bounds


def resample_utilization_distribution(
        distribution,
        *,
        target_bounds,
        target_shape,
        target_crs=None,
        preserve_mass=True,
        dtype=np.float32,
        chunk_rows=256,
):
    dst_height, dst_width = map(int, target_shape)
    target = np.zeros((dst_height, dst_width), dtype=dtype)
    _add_resampled_utilization_distribution(
        target,
        distribution,
        target_bounds=target_bounds,
        target_crs=target_crs,
        preserve_mass=preserve_mass,
        normalize=False,
        chunk_rows=chunk_rows,
    )
    return target


def _add_resampled_utilization_distribution(
        target,
        distribution,
        *,
        target_bounds,
        target_crs=None,
        preserve_mass=True,
        normalize=False,
        chunk_rows=256,
):
    """Resample one UD into ``target`` using bounded row chunks.

    A short first pass obtains the resampled mass needed for exact source-mass
    preservation. The second pass writes directly into the combined raster, so
    no second target-sized array exists at any point.
    """
    distribution = _coerce_grid(distribution)
    source_total = _positive_sum(distribution.grid)
    if source_total <= 0:
        return 0.0

    sampled_total = 0.0
    for _, _, _, sampled in _resampled_utilization_chunks(
            distribution,
            target_bounds=target_bounds,
            target_shape=target.shape,
            target_crs=target_crs,
            chunk_rows=chunk_rows,
    ):
        sampled_total += float(sampled.sum(dtype=np.float64))
    if sampled_total <= 0:
        return 0.0

    if normalize:
        scale = 1.0 / sampled_total
    elif preserve_mass:
        scale = source_total / sampled_total
    else:
        scale = 1.0

    for row_start, row_stop, valid, sampled in _resampled_utilization_chunks(
            distribution,
            target_bounds=target_bounds,
            target_shape=target.shape,
            target_crs=target_crs,
            chunk_rows=chunk_rows,
    ):
        target_chunk = target[row_start:row_stop]
        target_chunk[valid] += (sampled * scale).astype(target.dtype, copy=False)
    return sampled_total * scale


def _resampled_utilization_chunks(
        distribution,
        *,
        target_bounds,
        target_shape,
        target_crs,
        chunk_rows,
):
    source = np.asarray(distribution.grid)
    src_min_x, src_min_y, src_max_x, src_max_y = distribution.bounds
    dst_min_x, dst_min_y, dst_max_x, dst_max_y = map(float, target_bounds)
    dst_height, dst_width = map(int, target_shape)
    src_height, src_width = source.shape
    chunk_rows = _positive_chunk_rows(chunk_rows)

    dst_dx = (dst_max_x - dst_min_x) / dst_width
    dst_dy = (dst_max_y - dst_min_y) / dst_height
    target_x = dst_min_x + (np.arange(dst_width) + 0.5) * dst_dx
    src_dx = (src_max_x - src_min_x) / src_width
    src_dy = (src_max_y - src_min_y) / src_height

    for row_start in range(0, dst_height, chunk_rows):
        row_stop = min(dst_height, row_start + chunk_rows)
        rows = np.arange(row_start, row_stop)
        target_y = dst_max_y - (rows + 0.5) * dst_dy
        xs, ys = np.meshgrid(target_x, target_y)
        xs, ys = _transform_coordinates(xs, ys, target_crs, distribution.crs)
        valid = (
            (xs >= src_min_x)
            & (xs < src_max_x)
            & (ys >= src_min_y)
            & (ys < src_max_y)
        )
        if not valid.any():
            yield row_start, row_stop, valid, np.empty(0, dtype=source.dtype)
            continue

        src_cols = np.floor((xs - src_min_x) / src_dx).astype(np.int32)
        src_rows = np.floor((src_max_y - ys) / src_dy).astype(np.int32)
        np.clip(src_cols, 0, src_width - 1, out=src_cols)
        np.clip(src_rows, 0, src_height - 1, out=src_rows)
        sampled = np.maximum(source[src_rows[valid], src_cols[valid]], 0)
        yield row_start, row_stop, valid, sampled


def _resample_ud_to_common_grid(source, source_bounds, target_bounds, target_shape):
    """Compatibility wrapper for the former MixedWalker private helper."""
    return resample_utilization_distribution(
        (source, source_bounds),
        target_bounds=target_bounds,
        target_shape=target_shape,
        preserve_mass=False,
    )


def _coerce_grid(distribution):
    if isinstance(distribution, UtilizationDistributionGrid):
        grid = np.asarray(distribution.grid)
        bounds = tuple(map(float, distribution.bounds))
        return UtilizationDistributionGrid(grid, bounds, distribution.cell_size, distribution.crs)
    if not isinstance(distribution, (tuple, list)) or len(distribution) < 2:
        raise TypeError("A UD must be UtilizationDistributionGrid or a (grid, bounds) tuple")
    grid, bounds = distribution[:2]
    cell_size = distribution[2] if len(distribution) > 2 else None
    crs = distribution[3] if len(distribution) > 3 else None
    return UtilizationDistributionGrid(np.asarray(grid), tuple(map(float, bounds)), cell_size, crs)


def _positive_chunk_rows(value):
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError("chunk_rows must be a positive integer")
    return int(value)


def _positive_sum(values):
    """Sum a normally non-negative raster without allocating a clipped copy."""
    values = np.asarray(values)
    if values.size == 0:
        return 0.0
    if float(values.min()) >= 0:
        return float(values.sum(dtype=np.float64))
    return float(values[values > 0].sum(dtype=np.float64))


def _valid_bounds(bounds):
    min_x, min_y, max_x, max_y = bounds
    return max_x > min_x and max_y > min_y and np.isfinite(bounds).all()


def _size_pair(value):
    if isinstance(value, (tuple, list, np.ndarray)):
        dx, dy = map(float, value)
    else:
        dx = dy = float(value)
    if dx <= 0 or dy <= 0 or not np.isfinite((dx, dy)).all():
        raise ValueError("cell size must be finite and greater than zero")
    return dx, dy


def _canonical_cell_size(value):
    dx, dy = _size_pair(value)
    return dx if np.isclose(dx, dy) else (dx, dy)


def _cell_size(distribution):
    if distribution.cell_size is not None:
        return _size_pair(distribution.cell_size)
    height, width = distribution.grid.shape
    min_x, min_y, max_x, max_y = distribution.bounds
    return (max_x - min_x) / width, (max_y - min_y) / height


def _cell_size_in_crs(distribution, target_crs):
    if _same_crs(distribution.crs, target_crs):
        return _cell_size(distribution)
    source_bounds = distribution.bounds
    target_bounds = _transform_bounds(source_bounds, distribution.crs, target_crs)
    height, width = distribution.grid.shape
    return (target_bounds[2] - target_bounds[0]) / width, (target_bounds[3] - target_bounds[1]) / height


def _volume_percent_grid(probability):
    """Cumulative volume of the smallest high-density region per grid cell."""
    flat = np.asarray(probability, dtype=np.float32).ravel()
    order = np.argsort(-flat, kind="stable")
    sorted_probability = flat[order]
    cumulative_percent = np.cumsum(sorted_probability, dtype=np.float64) * 100.0
    group_ends = np.r_[
        np.flatnonzero(sorted_probability[:-1] != sorted_probability[1:]),
        len(sorted_probability) - 1,
    ]
    group_starts = np.r_[0, group_ends[:-1] + 1]
    sorted_volume = np.empty_like(sorted_probability)
    for start, end in zip(group_starts, group_ends):
        sorted_volume[start:end + 1] = cumulative_percent[end]
    volume = np.empty_like(flat)
    volume[order] = sorted_volume
    return volume.reshape(np.shape(probability))


def _display_smoothing_sigma(distribution, smoothing_metres):
    if smoothing_metres is None or float(smoothing_metres) <= 0:
        return None
    dx, dy = _cell_size(distribution)
    if distribution.crs is None:
        return None
    from pyproj import CRS
    crs = CRS.from_user_input(distribution.crs)
    if crs.is_geographic:
        latitude = (distribution.bounds[1] + distribution.bounds[3]) / 2.0
        metres_per_degree_y = 111_320.0
        metres_per_degree_x = max(111_320.0 * np.cos(np.deg2rad(latitude)), 1.0)
        return (
            float(smoothing_metres) / (dy * metres_per_degree_y),
            float(smoothing_metres) / (dx * metres_per_degree_x),
        )
    unit_factor = crs.axis_info[0].unit_conversion_factor if crs.axis_info else 1.0
    unit_factor = float(unit_factor or 1.0)
    return float(smoothing_metres) / (dy * unit_factor), float(smoothing_metres) / (dx * unit_factor)


def _is_geographic_crs(crs):
    if crs is None:
        return False
    from pyproj import CRS
    return CRS.from_user_input(crs).is_geographic


def _points_in_distribution_crs(points, target_crs):
    if points is None:
        return None
    if not isinstance(points, gpd.GeoDataFrame):
        if not hasattr(points, "geometry"):
            return None
        points = gpd.GeoDataFrame(points, geometry="geometry", crs=getattr(points, "crs", None))
    if points.crs is not None and target_crs is not None and not _same_crs(points.crs, target_crs):
        points = points.to_crs(target_crs)
    return points


def _leaflet_coords_from_points(points):
    points = _points_in_distribution_crs(points, "EPSG:4326")
    if points is None or not len(points):
        return []
    return _finite_leaflet_coords(points.geometry)


def _marine_land_wgs84(bounds):
    from environmentcma.ocean_cover import marine_cover_path

    west, south, east, north = map(float, bounds)
    land = gpd.read_file(marine_cover_path()).to_crs("EPSG:4326")
    land.geometry = land.geometry.make_valid()
    return land.clip((west, south, east, north))


def _marine_coastline_geojson(display_bounds):
    land = _marine_land_wgs84(display_bounds)
    return None if land.empty else land.__geo_interface__


def _marine_land_for_distribution(distribution):
    if distribution.crs is None:
        return None
    geographic_bounds = _transform_bounds(
        distribution.bounds,
        distribution.crs,
        "EPSG:4326",
    )
    land = _marine_land_wgs84(geographic_bounds)
    if land.empty or _same_crs(distribution.crs, "EPSG:4326"):
        return land
    return land.to_crs(distribution.crs)


def _png_output_directory(save_plots, out_directory):
    if save_plots is False or save_plots is None:
        return None
    return _ud_output_directory(save_plots, out_directory)


def _ud_output_directory(save_plots, out_directory):
    if save_plots not in (False, None, True):
        return Path(save_plots)
    return Path(out_directory or ".") / "ud_plots"


def _safe_filename(value):
    safe = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in str(value)
    ).strip("_")
    return safe or "animal"


def _same_crs(source_crs, target_crs):
    if source_crs is None or target_crs is None:
        return source_crs is None and target_crs is None
    from pyproj import CRS
    return CRS.from_user_input(source_crs) == CRS.from_user_input(target_crs)


def _transform_coordinates(xs, ys, source_crs, target_crs):
    if _same_crs(source_crs, target_crs):
        return xs, ys
    if source_crs is None or target_crs is None:
        raise ValueError("Both source and target CRS are required when combining grids from different CRSs")
    from pyproj import Transformer
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return transformer.transform(xs, ys)


def _transform_bounds(bounds, source_crs, target_crs):
    if _same_crs(source_crs, target_crs):
        return tuple(map(float, bounds))
    if source_crs is None or target_crs is None:
        raise ValueError("Both source and target CRS are required when transforming bounds")
    from pyproj import Transformer
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return tuple(map(float, transformer.transform_bounds(*bounds, densify_pts=21)))


def _leaflet_grid(distribution):
    """Return a normalized, genuinely georeferenced EPSG:4326 display raster.

    Projected grids cannot be placed accurately by transforming their four
    corners and stretching the unchanged array into a Leaflet rectangle. Grid
    convergence makes that operation a shear. Reprojecting the raster first
    keeps every probability cell aligned with the basemap.
    """
    distribution = _coerce_grid(distribution)
    source = np.clip(np.asarray(distribution.grid, dtype=np.float32), 0.0, None)
    if source.ndim != 2 or not np.isfinite(source).all() or source.sum() <= 0:
        return source, _transform_bounds(distribution.bounds, distribution.crs, "EPSG:4326")

    if _same_crs(distribution.crs, "EPSG:4326"):
        display = source.copy()
        display_bounds = tuple(map(float, distribution.bounds))
    else:
        if distribution.crs is None:
            raise ValueError("A projected utilization distribution requires CRS metadata")
        from rasterio.transform import array_bounds, from_bounds
        from rasterio.warp import Resampling, calculate_default_transform, reproject

        height, width = source.shape
        source_bounds = tuple(map(float, distribution.bounds))
        source_transform = from_bounds(*source_bounds, width=width, height=height)
        target_transform, target_width, target_height = calculate_default_transform(
            distribution.crs,
            "EPSG:4326",
            width,
            height,
            *source_bounds,
        )
        display = np.zeros((target_height, target_width), dtype=np.float32)
        reproject(
            source=source,
            destination=display,
            src_transform=source_transform,
            src_crs=distribution.crs,
            dst_transform=target_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            init_dest_nodata=True,
        )
        display_bounds = tuple(map(float, array_bounds(target_height, target_width, target_transform)))

    display = np.clip(display, 0.0, None)
    total = display.sum()
    if total > 0:
        display /= total
    return display, display_bounds


def _coords_by_walk_version(animal_id, path_gdfs):
    if path_gdfs is None:
        return {}
    if isinstance(path_gdfs, gpd.GeoDataFrame):
        path_gdfs = [path_gdfs]
    coords_by_version = {}
    for index, path_gdf in enumerate(path_gdfs, start=1):
        coords = _finite_leaflet_coords(path_gdf.geometry)
        if not coords:
            continue
        version = path_gdf["_rw_walk_version"].iloc[0] if "_rw_walk_version" in path_gdf.columns else index
        label = str(animal_id) if len(path_gdfs) == 1 else f"{animal_id} v{version}"
        coords_by_version[label] = coords
    return coords_by_version


def _primary_coords_by_animal(path_gdfs, id_col):
    if path_gdfs is None:
        return {}
    if isinstance(path_gdfs, gpd.GeoDataFrame):
        path_gdfs = [path_gdfs]
    result = {}
    for path_gdf in path_gdfs:
        if path_gdf is None or len(path_gdf) == 0:
            continue
        if "_rw_walk_base_id" in path_gdf.columns:
            animal_id = str(path_gdf["_rw_walk_base_id"].iloc[0])
        elif id_col in path_gdf.columns:
            animal_id = str(path_gdf[id_col].iloc[0])
        else:
            continue
        if animal_id not in result:
            coords = _finite_leaflet_coords(path_gdf.geometry)
            if coords:
                result[animal_id] = coords
    return result


def _finite_leaflet_coords(geometries):
    coords = []
    for point in geometries:
        if point is None or point.is_empty:
            continue
        lon, lat = float(point.x), float(point.y)
        if np.isfinite((lon, lat)).all():
            coords.append((lat, lon))
    return coords


__all__ = [
    "UtilizationDistributionGrid",
    "UtilizationDistributionMixin",
    "combine_grid_utilization_distributions",
    "combine_utilization_distributions",
    "resample_utilization_distribution",
    "utilization_distribution_from_forward_density",
]
