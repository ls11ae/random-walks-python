import numpy as np

from randomwalks.bindings.data_structures.Terrain import MesaLandcover, TerrainMapHandle


def _cap_step_radius_to_grid(T, S, start_point, end_point, width, height):
    """Fit a step kernel into the local grid and retain endpoint reachability.

    A radius ``S`` produces a square runtime kernel with side length
    ``2 * S + 1``. Restrict that support to the smaller grid dimension so a
    very small or narrow segment bbox cannot create a kernel much larger than
    the grid it is evaluated on. When the radius is reduced, increase ``T``
    only as far as needed to keep the segment endpoint reachable.
    """
    T = _positive_grid_integer(T, "T")
    S = _positive_grid_integer(S, "S")
    width = _positive_grid_integer(width, "width")
    height = _positive_grid_integer(height, "height")

    # Radius one is the smallest value supported by the random-walk bindings,
    # including grids with a singleton or two-node axis.
    max_grid_radius = max(1, (min(width, height) - 1) // 2)
    capped_S = min(S, max_grid_radius)
    if capped_S == S:
        return T, S

    start_x, start_y = start_point
    end_x, end_y = end_point
    chebyshev_distance = max(
        abs(int(end_x) - int(start_x)),
        abs(int(end_y) - int(start_y)),
    )
    required_T = int(np.ceil(chebyshev_distance / capped_S))
    return max(T, required_T), capped_S


def _coarsen_grid_to_cell_size(utm_bbox, width, height, target_cell_size):
    """Coarsen a grid towards a physical node spacing.

    The returned shape uses the fewest nodes needed for the requested spacing
    when the original grid supports it. The original shape remains an upper
    bound, so this helper coarsens but never refines the configured grid.
    """
    width = _positive_grid_integer(width, "width")
    height = _positive_grid_integer(height, "height")
    try:
        target_cell_size = float(target_cell_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_cell_size must be positive and finite.") from exc
    if not np.isfinite(target_cell_size) or target_cell_size <= 0:
        raise ValueError("target_cell_size must be positive and finite.")

    min_x, min_y, max_x, max_y = map(float, utm_bbox)
    extent_x = max_x - min_x
    extent_y = max_y - min_y
    if not np.isfinite((min_x, min_y, max_x, max_y)).all() or extent_x <= 0 or extent_y <= 0:
        raise ValueError("utm_bbox must contain positive finite extents.")

    target_width = max(2, int(np.ceil(extent_x / target_cell_size)) + 1)
    target_height = max(2, int(np.ceil(extent_y / target_cell_size)) + 1)
    return min(width, target_width), min(height, target_height)


def _grid_node_cell_size(utm_bbox, width, height):
    """Return the larger physical node spacing of a regular UTM grid."""
    width = _positive_grid_integer(width, "width")
    height = _positive_grid_integer(height, "height")
    min_x, min_y, max_x, max_y = map(float, utm_bbox)
    spacing_x = (max_x - min_x) / (width - 1) if width > 1 else max_x - min_x
    spacing_y = (max_y - min_y) / (height - 1) if height > 1 else max_y - min_y
    cell_size = max(spacing_x, spacing_y)
    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("utm_bbox and grid shape must define a positive finite cell size.")
    return float(cell_size)


def _padded_projected_point_bounds(points, *, padding, minimum_padding):
    """Bound every projected segment point, including nonlinear CRS interiors."""
    coordinates = np.asarray(points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not len(coordinates):
        raise ValueError("points must be a non-empty sequence of projected x/y pairs")
    if not np.isfinite(coordinates).all():
        raise ValueError("projected points must be finite")
    min_x, min_y = coordinates.min(axis=0)
    max_x, max_y = coordinates.max(axis=0)
    pad_x = max((max_x - min_x) * float(padding), float(minimum_padding))
    pad_y = max((max_y - min_y) * float(padding), float(minimum_padding))
    return (min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y)


def _marine_rw_grid_terrain(width, height, utm_bbox, epsg, lonlat_bbox):
    """Classify land at every native RW grid node and return that exact grid."""
    import geopandas as gpd
    from environmentcma.ocean_cover import marine_cover_path
    from shapely import intersects_xy

    min_lon, min_lat, max_lon, max_lat = lonlat_bbox
    land = gpd.read_file(marine_cover_path()).to_crs("EPSG:4326")
    land.geometry = land.geometry.make_valid()
    land = land.clip((min_lon, min_lat, max_lon, max_lat))

    if land.empty:
        return TerrainMapHandle.single_value(MesaLandcover.PERMANENT_WATER, width, height)

    land = land.to_crs(_as_epsg_crs(epsg))
    land.geometry = land.geometry.make_valid()
    land_geometry = land.geometry.union_all()

    min_x, min_y, max_x, max_y = map(float, utm_bbox)
    x_coords = np.full(1, (min_x + max_x) / 2.0) if width == 1 else np.linspace(min_x, max_x, width)
    y_coords = np.full(1, (min_y + max_y) / 2.0) if height == 1 else np.linspace(max_y, min_y, height)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    is_land = intersects_xy(land_geometry, grid_x, grid_y)

    terrain = TerrainMapHandle(width=width, height=height)
    for y in range(height):
        for x in range(width):
            terrain.set(
                x,
                y,
                MesaLandcover.TREE_COVER if is_land[y, x] else MesaLandcover.PERMANENT_WATER,
            )
    return terrain


def _as_epsg_crs(epsg):
    value = str(epsg)
    return value if value.upper().startswith("EPSG:") else f"EPSG:{value}"


def _positive_grid_integer(value, name):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(numeric) or numeric < 1 or not numeric.is_integer():
        raise ValueError(f"{name} must be a positive integer.")
    return int(numeric)


def _grid_node_edge_bounds(utm_bbox, width, height):
    """Return raster-cell bounds for a grid whose nodes span ``utm_bbox``.

    On axes with multiple nodes, the bounding-box coordinates describe the
    centres of the outermost cells.  Raster bounds therefore extend by half a
    node spacing at either end.  A singleton axis has its node at the bounding
    box midpoint and keeps the supplied extent as that cell's bounds.
    """
    width = _positive_grid_integer(width, "width")
    height = _positive_grid_integer(height, "height")
    min_x, min_y, max_x, max_y = map(float, utm_bbox)

    half_x_step = (max_x - min_x) / (2.0 * (width - 1)) if width > 1 else 0.0
    half_y_step = (max_y - min_y) / (2.0 * (height - 1)) if height > 1 else 0.0

    return (
        min_x - half_x_step,
        min_y - half_y_step,
        max_x + half_x_step,
        max_y + half_y_step,
    )


def _grid_walk_to_geographic(grid_walk, utm_bbox, width, height, transformer):
    """Convert RW grid-node coordinates to geographic coordinate pairs."""
    width = _positive_grid_integer(width, "width")
    height = _positive_grid_integer(height, "height")
    min_x, min_y, max_x, max_y = map(float, utm_bbox)

    x_midpoint = (min_x + max_x) / 2.0
    y_midpoint = (min_y + max_y) / 2.0
    geographic = []
    for grid_x, grid_y in grid_walk:
        x = (
            x_midpoint
            if width == 1
            else min_x + float(grid_x) * (max_x - min_x) / (width - 1)
        )
        y = (
            y_midpoint
            if height == 1
            else max_y - float(grid_y) * (max_y - min_y) / (height - 1)
        )
        geographic.append(tuple(transformer.transform(x, y)))
    return geographic
