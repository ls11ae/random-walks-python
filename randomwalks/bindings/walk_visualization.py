import os.path
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from math import isclose
from pathlib import Path
from typing import Any, Optional

import folium
import movingpandas as mpd
import numpy as np
from folium.plugins import TimestampedGeoJson

from randomwalks.bindings.data_structures.Terrain import (
    MESA_LANDCOVER_COLORS,
    TerrainMapHandle,
)
from randomwalks.bindings.plotter import ud_isopleth_band_map


class LeafletTiles(str, Enum):
    ESRI_WORLD_IMAGERY = "Esri.WorldImagery"
    ESRI_WORLD_TOPOMAP = "Esri.WorldTopoMap"
    ESRI_WORLD_SHADOW = "Esri.WorldShadow"
    ESRI_NATURAL_EARTH_IMMERSIVE = "Esri.NatGeoWorldMap"
    ESRI_NATURAL_EARTH_VECTOR = "Esri.NatGeoWorldMapVector"
    ESRI_STREETS = "Esri.WorldStreetMap"
    ESRI_LIGHT_GRAY = "Esri.WorldLightGrayCanvas"
    ESRI_DARK_GRAY = "Esri.WorldDarkGrayCanvas"
    ESRI_TOPO_GRAY = "Esri.WorldTopoMap"
    ESRI_TOPO_DARK_GRAY = "Esri.WorldTopoMapDark"
    CARTODB_POSITRON_NO_LABELS = "CartoDB.PositronNoLabels"


@dataclass(frozen=True)
class LeafletGridOverlay:
    """A grid and geodetic footprint that can be drawn as a Leaflet image layer."""

    grid: Any
    bounds: Any
    name: str | None = None
    opacity: float | None = None


_LANDCOVER_TXT_RE = re.compile(
    r"^landcover_(?P<animal_id>.+)_"
    r"(?P<min_lon>-?\d+(?:\.\d+)?)_"
    r"(?P<min_lat>-?\d+(?:\.\d+)?)_"
    r"(?P<max_lon>-?\d+(?:\.\d+)?)_"
    r"(?P<max_lat>-?\d+(?:\.\d+)?)_"
    r"(?P<resolution>\d+)\.txt$"
)

_UD_GRADIENT_STOPS = np.array([
    (255, 255, 153),
    (255, 0, 0),
    (139, 0, 0),
], dtype=float)

_WALK_BASE_ID_COL = "_rw_walk_base_id"
_WALK_VERSION_COL = "_rw_walk_version"


def _color_cycle():
    # A palette of distinct folium-supported colors
    return [
        "red", "blue", "green", "purple", "orange", "darkred", "lightred", "beige",
        "darkblue", "darkgreen", "cadetblue", "darkpurple", "white", "pink",
        "lightblue", "lightgreen", "gray", "black", "lightgray"
    ]


def _make_map(center_point, zoom_start=13):
    # Satellite tiles (Esri World Imagery), matches existing style
    return folium.Map(
        location=center_point,
        zoom_start=zoom_start,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri"
    )


def _terrain_to_leaflet_image(terrain):
    terrain_array = _grid_to_array(terrain, dtype=int)
    image = np.zeros((*terrain_array.shape, 4), dtype=np.uint8)
    for value in np.unique(terrain_array):
        image[terrain_array == value] = _color_to_uint8_rgba(
            MESA_LANDCOVER_COLORS.get(int(value), (0.6, 0.6, 0.6, 0.75))
        )
    return image


def _utilization_distribution_to_leaflet_image(ud):
    band_map = ud_isopleth_band_map(_grid_to_array(ud, dtype=float))
    image = np.zeros((*band_map.shape, 4), dtype=np.uint8)
    levels = np.arange(5, 100, 5)
    for index, level in enumerate(levels):
        mask = band_map == level
        if np.any(mask):
            t = 1.0 - index / (len(levels) - 1)
            image[mask] = (*_ud_gradient_color(t), 230)
    return image


def _grid_to_array(grid, *, dtype):
    if isinstance(grid, TerrainMapHandle):
        array = grid.to_numpy()
    elif hasattr(grid, "to_numpy"):
        array = grid.to_numpy()
    elif isinstance(grid, (str, os.PathLike, Path)):
        array = np.loadtxt(grid, dtype=dtype)
    else:
        array = grid

    array = np.asarray(array, dtype=dtype)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("Leaflet grid overlays require a 2D array or terrain grid")
    return array


def _color_to_uint8_rgba(color):
    rgba = np.asarray(color, dtype=float)
    if rgba.size == 3:
        rgba = np.concatenate([rgba, [1.0]])
    if rgba.size != 4:
        raise ValueError("Colors must be RGB or RGBA")
    if rgba.max() <= 1.0:
        rgba = rgba * 255.0
    return np.clip(np.rint(rgba), 0, 255).astype(np.uint8)


def _ud_gradient_color(t):
    t = float(np.clip(t, 0.0, 1.0))
    scaled = t * (len(_UD_GRADIENT_STOPS) - 1)
    left = min(int(np.floor(scaled)), len(_UD_GRADIENT_STOPS) - 2)
    frac = scaled - left
    color = (1.0 - frac) * _UD_GRADIENT_STOPS[left] + frac * _UD_GRADIENT_STOPS[left + 1]
    return tuple(np.clip(np.rint(color), 0, 255).astype(np.uint8))


def _normalize_leaflet_bounds(bounds):
    if isinstance(bounds, Mapping):
        if {"min_lon", "min_lat", "max_lon", "max_lat"}.issubset(bounds):
            bounds = (
                bounds["min_lon"],
                bounds["min_lat"],
                bounds["max_lon"],
                bounds["max_lat"],
            )
        elif {"west", "south", "east", "north"}.issubset(bounds):
            bounds = (
                bounds["west"],
                bounds["south"],
                bounds["east"],
                bounds["north"],
            )

    array = np.asarray(bounds, dtype=float)
    if array.shape == (4,):
        min_lon, min_lat, max_lon, max_lat = array.tolist()
        return [[min_lat, min_lon], [max_lat, max_lon]]
    if array.shape == (2, 2):
        return array.tolist()
    raise ValueError(
        "Overlay bounds must be (min_lon, min_lat, max_lon, max_lat) "
        "or [[south, west], [north, east]]"
    )


def _add_grid_overlay(m, overlay, *, image_factory, default_name, default_opacity):
    image = image_factory(overlay.grid)
    folium.raster_layers.ImageOverlay(
        image=image,
        bounds=_normalize_leaflet_bounds(overlay.bounds),
        name=overlay.name or default_name,
        opacity=default_opacity if overlay.opacity is None else overlay.opacity,
        origin="upper",
        interactive=True,
        cross_origin=False,
    ).add_to(m)


def _add_grid_overlays_for_id(
        m,
        traj_id,
        terrain_overlays=None,
        utilization_distribution_overlays=None,
        *,
        terrain_opacity=0.6,
        utilization_distribution_opacity=0.7,
):
    key = str(traj_id)
    if terrain_overlays and key in terrain_overlays:
        _add_grid_overlay(
            m,
            terrain_overlays[key],
            image_factory=_terrain_to_leaflet_image,
            default_name=f"{key} terrain",
            default_opacity=terrain_opacity,
        )
    if utilization_distribution_overlays and key in utilization_distribution_overlays:
        _add_grid_overlay(
            m,
            utilization_distribution_overlays[key],
            image_factory=_utilization_distribution_to_leaflet_image,
            default_name=f"{key} utilization distribution",
            default_opacity=utilization_distribution_opacity,
        )


def _resolve_grid_overlay_map(
        overlays,
        overlay_bounds,
        *,
        default_name,
        default_opacity,
        trajectory_ids=None,
):
    if overlays is None or overlays is False:
        return {}
    if overlays is True:
        raise ValueError("Boolean overlay discovery is only supported for terrain_overlays")

    if not isinstance(overlays, Mapping):
        ids = [str(traj_id) for traj_id in trajectory_ids or ["walk"]]
        overlays = {traj_id: overlays for traj_id in ids}

    resolved = {}
    for key, overlay in overlays.items():
        traj_id = str(key)
        if isinstance(overlay, LeafletGridOverlay):
            grid = overlay.grid
            bounds = overlay.bounds
            name = overlay.name
            opacity = overlay.opacity
        elif isinstance(overlay, Mapping) and "grid" in overlay:
            grid = overlay["grid"]
            bounds = overlay.get("bounds")
            name = overlay.get("name")
            opacity = overlay.get("opacity")
        else:
            grid = overlay
            bounds = None
            name = None
            opacity = None

        if bounds is None:
            bounds = _bounds_for_key(overlay_bounds, traj_id)
        if bounds is None:
            raise ValueError(f"Missing overlay bounds for '{traj_id}'")

        resolved[traj_id] = LeafletGridOverlay(
            grid=grid,
            bounds=bounds,
            name=name or f"{traj_id} {default_name}",
            opacity=default_opacity if opacity is None else opacity,
        )
    return resolved


def _bounds_for_key(overlay_bounds, key):
    if overlay_bounds is None:
        return None
    if _looks_like_single_bounds(overlay_bounds):
        return overlay_bounds
    if isinstance(overlay_bounds, Mapping):
        if key in overlay_bounds:
            return overlay_bounds[key]
        str_key = str(key)
        if str_key in overlay_bounds:
            return overlay_bounds[str_key]
        return None
    return overlay_bounds


def _looks_like_single_bounds(bounds):
    if isinstance(bounds, Mapping):
        return (
            {"min_lon", "min_lat", "max_lon", "max_lat"}.issubset(bounds)
            or {"west", "south", "east", "north"}.issubset(bounds)
        )
    try:
        array = np.asarray(bounds, dtype=float)
    except (TypeError, ValueError):
        return False
    return array.shape in {(4,), (2, 2)}


def _resolve_terrain_overlay_map(
        terrain_overlays,
        save_path,
        trajectory_ids,
        overlay_bounds,
        *,
        terrain_resolution=None,
        terrain_opacity=0.6,
):
    if terrain_overlays is True:
        return _discover_terrain_overlays(save_path, trajectory_ids, terrain_resolution, terrain_opacity)
    return _resolve_grid_overlay_map(
        terrain_overlays,
        overlay_bounds,
        default_name="terrain",
        default_opacity=terrain_opacity,
        trajectory_ids=trajectory_ids,
    )


def _discover_terrain_overlays(save_path, trajectory_ids, terrain_resolution=None, terrain_opacity=0.6):
    landcover_dir = Path(save_path) / "landcover"
    if not landcover_dir.exists():
        return {}

    wanted = {str(traj_id) for traj_id in trajectory_ids}
    candidates = {}
    for path in sorted(landcover_dir.glob("landcover_*.txt")):
        parsed = _parse_landcover_grid_path(path)
        if parsed is None:
            continue
        traj_id = parsed["animal_id"]
        resolution = parsed["resolution"]
        if wanted and traj_id not in wanted:
            continue
        if terrain_resolution is not None and resolution != int(terrain_resolution):
            continue
        current = candidates.get(traj_id)
        if current is None or resolution > current["resolution"]:
            candidates[traj_id] = {**parsed, "path": path}

    return {
        traj_id: LeafletGridOverlay(
            grid=info["path"],
            bounds=info["bounds"],
            name=f"{traj_id} terrain",
            opacity=terrain_opacity,
        )
        for traj_id, info in candidates.items()
    }


def _parse_landcover_grid_path(path):
    match = _LANDCOVER_TXT_RE.match(Path(path).name)
    if match is None:
        return None
    groups = match.groupdict()
    bounds = tuple(
        float(groups[name])
        for name in ("min_lon", "min_lat", "max_lon", "max_lat")
    )
    return {
        "animal_id": groups["animal_id"],
        "bounds": bounds,
        "resolution": int(groups["resolution"]),
    }


def _add_start_end_markers(m, coords, color, label_prefix):
    if not coords:
        return
    start = coords[0]
    end = coords[-1]
    folium.Marker(
        location=start,
        tooltip=f"{label_prefix} start",
        icon=folium.Icon(color="green", icon="play", prefix="fa")
    ).add_to(m)
    folium.Marker(
        location=end,
        tooltip=f"{label_prefix} end",
        icon=folium.Icon(color="red", icon="stop", prefix="fa")
    ).add_to(m)


def _find_coord_index(coords, target, tol=1e-5):
    for idx, (lat, lon) in enumerate(coords):
        if isclose(lat, target[0], abs_tol=tol) and isclose(lon, target[1], abs_tol=tol):
            return idx
    return None


def _add_step_boxes(m, coords, steps_for_animal, color):
    if not coords or not steps_for_animal:
        return
    for i, step_coord in enumerate(steps_for_animal, start=1):
        if i == 0 or i == len(steps_for_animal) - 1:
            continue
        idx = _find_coord_index(coords, step_coord)
        if idx is None:
            continue
        latlon = coords[idx]
        folium.Marker(
            location=latlon,
            tooltip=f"Step {i}",
            icon=folium.DivIcon(
                html=(
                    f'<div style="display:inline-block; background: rgba(255,255,255,0.85); '
                    f'border: 1px solid {color}; border-radius: 2px; '
                    f'padding: 1px 3px; font-weight: 400; color: #000; '
                    f'box-shadow: 0 1px 2px rgba(0,0,0,0.3);">{i}</div>'
                )
            )
        ).add_to(m)


def walks_to_osm_multi(
        geodetic_walks: dict[str, list[tuple[float, float]]],
        out_path: str = ".",
        map_filename: str = "walks_map.html",
        step_annotations: dict[str, list[tuple[int, int]]] | None = None,
        zoom_start: int = 13,
        annotated=False,
        terrain_overlays=None,
        utilization_distribution_overlays=None,
        overlay_bounds=None,
        terrain_resolution: int | None = None,
        terrain_opacity: float = 0.6,
        utilization_distribution_opacity: float = 0.7,
        draw_walks: bool = True,
) -> str:
    """
    Create a single map with separate polylines for each animal id in geodetic_walks,
    each with a different color. Marks start and end points. If step_annotations is
    provided (dict[str, list[tuple[int,int]]]), places numbered boxes along the path
    in the order of appearance.

    Returns absolute path to the saved HTML.
    """
    if not geodetic_walks:
        raise ValueError("geodetic_walks is empty.")

    # Determine a reasonable center: use the first walk's first point
    first_key = next(iter(geodetic_walks))
    first_walk = geodetic_walks[first_key]
    if not first_walk:
        raise ValueError(f"First walk for key '{first_key}' has no coordinates.")
    center_point = first_walk[0]

    m = _make_map(center_point, zoom_start=zoom_start)
    terrain_overlay_map = _resolve_terrain_overlay_map(
        terrain_overlays,
        out_path,
        geodetic_walks.keys(),
        overlay_bounds,
        terrain_resolution=terrain_resolution,
        terrain_opacity=terrain_opacity,
    )
    ud_overlay_map = _resolve_grid_overlay_map(
        utilization_distribution_overlays,
        overlay_bounds,
        default_name="utilization distribution",
        default_opacity=utilization_distribution_opacity,
        trajectory_ids=geodetic_walks.keys(),
    )
    _add_grid_overlays_for_id(
        m,
        "walk",
        terrain_overlay_map,
        ud_overlay_map,
        terrain_opacity=terrain_opacity,
        utilization_distribution_opacity=utilization_distribution_opacity,
    )

    colors = _color_cycle()
    color_count = len(colors)

    for idx, (animal_id, coords) in enumerate(geodetic_walks.items()):
        if not coords:
            continue
        color = colors[idx % color_count]

        _add_grid_overlays_for_id(
            m,
            animal_id,
            terrain_overlay_map,
            ud_overlay_map,
            terrain_opacity=terrain_opacity,
            utilization_distribution_opacity=utilization_distribution_opacity,
        )

        if draw_walks:
            folium.PolyLine(coords, color=color, weight=3, tooltip=f"Animal {animal_id}").add_to(m)
            _add_start_end_markers(m, coords, color, label_prefix=f"Animal {animal_id}")

            if annotated and step_annotations and animal_id in step_annotations:
                _add_step_boxes(m, coords, step_annotations[animal_id], color)

    if terrain_overlay_map or ud_overlay_map:
        folium.LayerControl().add_to(m)

    os.makedirs(out_path, exist_ok=True)
    out_file = os.path.join(out_path, map_filename)
    m.save(out_file)
    return os.path.abspath(out_file)


def walk_to_osm(
        walk_coords_or_dict,
        original_coords: Optional[list[tuple[float, float]]] = None,
        animal_id: Optional[str] = None,
        walk_path: str = ".",
        step_annotations: dict[str, list[tuple[int, int]]] | None = None,
        map_filename: Optional[str] = None,
        zoom_start: int = 13,
        annotated: bool = False,
        terrain_overlays=None,
        utilization_distribution_overlays=None,
        overlay_bounds=None,
        terrain_resolution: int | None = None,
        terrain_opacity: float = 0.6,
        utilization_distribution_opacity: float = 0.7,
        draw_walk: bool = True,
):
    """
    Backwards-compatible entry point:
    - If walk_coords_or_dict is a list[tuple[float,float]], render a single-walk map (as before),
      with start/end markers and optional numbered step boxes.
    - If walk_coords_or_dict is a dict[str, list[tuple[float,float]]], render a single map
      containing all walks, each polyline in a different color, marking start/end and
      optional numbered step boxes per animal id from step_annotations.
    """
    # If a dict is passed, create a multi-walk map
    if isinstance(walk_coords_or_dict, dict):
        # map_filename default for multi
        map_name = map_filename or "walks_map.html"
        return walks_to_osm_multi(
            geodetic_walks=walk_coords_or_dict,
            out_path=walk_path,
            map_filename=map_name,
            step_annotations=step_annotations,
            zoom_start=zoom_start,
            annotated=annotated,
            terrain_overlays=terrain_overlays,
            utilization_distribution_overlays=utilization_distribution_overlays,
            overlay_bounds=overlay_bounds,
            terrain_resolution=terrain_resolution,
            terrain_opacity=terrain_opacity,
            utilization_distribution_opacity=utilization_distribution_opacity,
            draw_walks=draw_walk,
        )

    # Otherwise, assume a single coordinate sequence
    walk_coords = walk_coords_or_dict
    if not walk_coords:
        raise ValueError("walk_coords is empty for single-walk rendering.")

    if not animal_id:
        animal_id = "walk"

    start_point = walk_coords[0]
    m = _make_map(start_point, zoom_start=zoom_start)
    overlay_id = str(animal_id)
    terrain_overlay_map = _resolve_terrain_overlay_map(
        terrain_overlays,
        walk_path,
        [overlay_id],
        overlay_bounds,
        terrain_resolution=terrain_resolution,
        terrain_opacity=terrain_opacity,
    )
    ud_overlay_map = _resolve_grid_overlay_map(
        utilization_distribution_overlays,
        overlay_bounds,
        default_name="utilization distribution",
        default_opacity=utilization_distribution_opacity,
        trajectory_ids=[overlay_id],
    )
    if terrain_overlays is not None and overlay_id not in terrain_overlay_map and "walk" in terrain_overlay_map:
        terrain_overlay_map[overlay_id] = terrain_overlay_map["walk"]
    if (
            utilization_distribution_overlays is not None
            and overlay_id not in ud_overlay_map
            and "walk" in ud_overlay_map
    ):
        ud_overlay_map[overlay_id] = ud_overlay_map["walk"]

    _add_grid_overlays_for_id(
        m,
        overlay_id,
        terrain_overlay_map,
        ud_overlay_map,
        terrain_opacity=terrain_opacity,
        utilization_distribution_opacity=utilization_distribution_opacity,
    )

    if draw_walk:
        # Draw the single polyline (keeps the original 'red' default)
        folium.PolyLine(walk_coords, color="red", weight=3).add_to(m)

    if draw_walk and annotated and original_coords is not None:
        coords_list = [tuple(pt) for pt in original_coords.to_list()]  # Serie -> List[tuple]
        for idx, (lon, lat) in enumerate(coords_list, start=1):
            folium.CircleMarker(
                location=(lat, lon),
                radius=5,
                color="cyan",
                fill=True,
                fill_opacity=0.7,
                tooltip=f"Original {idx}"
            ).add_to(m)

    if draw_walk:
        # Start/End markers
        _add_start_end_markers(m, walk_coords, color="red", label_prefix=f"Animal {animal_id}")

    # Optional step boxes: if a dict is provided, try to use by animal_id
    # If a plain list is passed (not per-animal), also accept under the special key or fallback
    steps_for_animal = None
    if isinstance(step_annotations, dict):
        steps_for_animal = step_annotations.get(animal_id)
    elif step_annotations is not None:
        # Not a dict -> ignore (strict typing could be enforced)
        steps_for_animal = None

    if draw_walk and annotated and steps_for_animal:
        _add_step_boxes(m, walk_coords, steps_for_animal, color="red")

    if terrain_overlay_map or ud_overlay_map:
        folium.LayerControl().add_to(m)

    # Save
    os.makedirs(walk_path, exist_ok=True)
    out_name = map_filename or f"{animal_id}_walk_map.html"
    out_file = os.path.join(walk_path, out_name)
    m.save(out_file)
    return os.path.abspath(out_file)


colors = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "lightred", "beige", "darkblue",
    "darkgreen", "cadetblue", "darkpurple", "pink",
]


def _trajectory_base_id(traj):
    df = traj.df
    if _WALK_BASE_ID_COL in df.columns and len(df) > 0:
        return str(df[_WALK_BASE_ID_COL].iloc[0])
    return str(traj.id)


def _trajectory_version(traj):
    df = traj.df
    if _WALK_VERSION_COL in df.columns and len(df) > 0:
        return df[_WALK_VERSION_COL].iloc[0]
    return None


def _trajectory_display_label(traj, base_id, group_size):
    if group_size == 1:
        return str(traj.id)
    version = _trajectory_version(traj)
    if version is None:
        return str(traj.id)
    return f"{base_id} v{version}"


def _group_trajectories_by_base_id(traj_coll):
    groups = {}
    for traj in traj_coll.trajectories:
        groups.setdefault(_trajectory_base_id(traj), []).append(traj)
    return groups


def _trajectory_coords(traj):
    return [(pt.y, pt.x) for pt in traj.df.geometry]


def save_trajectory_coll_leaflet(
        traj_coll: mpd.TrajectoryCollection,
        save_path="walks/",
        tiles=LeafletTiles.ESRI_WORLD_IMAGERY,
        terrain_overlays=None,
        utilization_distribution_overlays=None,
        overlay_bounds=None,
        terrain_resolution: int | None = None,
        terrain_opacity: float = 0.6,
        utilization_distribution_opacity: float = 0.7,
):
    """
    Plot each trajectory in a MovingPandas TrajectoryCollection using Folium.
    Saves:
        - one HTML file per animal; versioned walks are drawn together
        - one combined map with one trajectory per animal

    terrain_overlays=True auto-loads generated landcover text grids from
    save_path/landcover. Explicit overlays can be passed as {trajectory_id:
    grid_or_path} with overlay_bounds, or as {trajectory_id: LeafletGridOverlay}.
    utilization_distribution_overlays accepts the same explicit forms and is
    rendered with the same isopleth bands as plot_terrain_walk.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    trajectory_groups = _group_trajectories_by_base_id(traj_coll)
    trajectory_ids = list(trajectory_groups)
    terrain_overlay_map = _resolve_terrain_overlay_map(
        terrain_overlays,
        save_path,
        trajectory_ids,
        overlay_bounds,
        terrain_resolution=terrain_resolution,
        terrain_opacity=terrain_opacity,
    )
    ud_overlay_map = _resolve_grid_overlay_map(
        utilization_distribution_overlays,
        overlay_bounds,
        default_name="utilization distribution",
        default_opacity=utilization_distribution_opacity,
        trajectory_ids=trajectory_ids,
    )

    if len(traj_coll.trajectories) == 0:
        center = (0.0, 0.0)
    else:
        first_coords = _trajectory_coords(traj_coll.trajectories[0])
        center = first_coords[0] if first_coords else (0.0, 0.0)

    # combined map
    m_all = folium.Map(location=center, zoom_start=14, tiles=tiles, attr="Tiles © Esri")

    for idx, (base_id, trajectories) in enumerate(trajectory_groups.items()):
        color = colors[idx % len(colors)]
        primary_traj = trajectories[0]
        primary_coords = _trajectory_coords(primary_traj)
        if not primary_coords:
            continue

        # save individual animal map
        m_single = folium.Map(location=primary_coords[0], zoom_start=14, tiles=tiles, attr="Tiles © Esri")
        _add_grid_overlays_for_id(
            m_single,
            base_id,
            terrain_overlay_map,
            ud_overlay_map,
            terrain_opacity=terrain_opacity,
            utilization_distribution_opacity=utilization_distribution_opacity,
        )
        for version_idx, traj in enumerate(trajectories):
            traj_id = str(traj.id)
            version_color = color if version_idx == 0 else colors[(idx + version_idx) % len(colors)]
            coords = _trajectory_coords(traj)
            if not coords:
                continue
            label = _trajectory_display_label(traj, base_id, len(trajectories))
            polyline_kwargs = {}
            if len(trajectories) > 1:
                polyline_kwargs["tooltip"] = label
            folium.PolyLine(coords, color=version_color, weight=4, opacity=0.8, **polyline_kwargs).add_to(m_single)
            marker_label = label if len(trajectories) > 1 else traj_id
            folium.Marker(coords[0], tooltip=f"{marker_label} Start").add_to(m_single)
            folium.Marker(coords[-1], tooltip=f"{marker_label} End").add_to(m_single)
        if terrain_overlay_map or ud_overlay_map:
            folium.LayerControl().add_to(m_single)

        m_single.save(str(save_path / f"{base_id}.html"))

        # add to combined map
        folium.PolyLine(primary_coords, color=color, weight=3, opacity=0.8).add_to(m_all)

    out_file = save_path / "all_trajectories.html"
    m_all.save(str(out_file))
    print(f"Trajectories saved to {out_file}")

    return out_file


def save_trajectory_collection_timed(traj_coll, save_path="walks/"):
    """
    Create a TimeDimension animated map for a MovingPandas TrajectoryCollection.
    One animated layer per trajectory.
    """

    if len(traj_coll.trajectories) == 0:
        raise Exception("failed - no trajectories")

    # Center of first trajectory
    p = traj_coll.trajectories[0].df.geometry.iloc[0]
    center = (p.y, p.x)

    m = folium.Map(location=center, zoom_start=14, tiles="Esri.WorldImagery", attr="Tiles © Esri")

    features = []
    ci = 0
    for traj in traj_coll.trajectories:
        traj_id = str(traj.id)

        df = traj.df
        coords = [(pt.x, pt.y) for pt in df.geometry]
        times = df.index.astype(str).tolist()

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
            "properties": {
                "times": times,
                "style": {"color": colors[ci], "weight": 3},
                "icon": "circle",
                "popup": traj_id,
            }
        }
        ci = (ci + 1) % len(colors)
        features.append(feature)

    TimestampedGeoJson(
        {
            "type": "FeatureCollection",
            "features": features,
        },
        period="PT1H",  # 1 second per frame
        add_last_point=True,
        auto_play=False,
        loop=False
    ).add_to(m)

    output = save_path
    m.save(str(output))
    print(f"Trajectories saved to {output}")
    return output
