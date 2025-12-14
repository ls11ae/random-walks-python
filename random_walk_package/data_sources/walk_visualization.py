import os.path
from math import isclose
from pathlib import Path
from typing import Optional

import folium
import movingpandas as mpd
from folium.plugins import TimestampedGeoJson


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
        annotated=False
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

    colors = _color_cycle()
    color_count = len(colors)

    for idx, (animal_id, coords) in enumerate(geodetic_walks.items()):
        if not coords:
            continue
        color = colors[idx % color_count]

        # Draw path
        folium.PolyLine(coords, color=color, weight=3, tooltip=f"Animal {animal_id}").add_to(m)

        # Start/End markers
        _add_start_end_markers(m, coords, color, label_prefix=f"Animal {animal_id}")

        # Optional step boxes (numbered labels)
        if annotated and step_annotations and animal_id in step_annotations:
            _add_step_boxes(m, coords, step_annotations[animal_id], color)

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
        annotated: bool = False
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
            zoom_start=zoom_start
        )

    # Otherwise, assume a single coordinate sequence
    walk_coords = walk_coords_or_dict
    if not walk_coords:
        raise ValueError("walk_coords is empty for single-walk rendering.")

    if not animal_id:
        animal_id = "walk"

    start_point = walk_coords[0]
    m = _make_map(start_point, zoom_start=zoom_start)

    # Draw the single polyline (keeps the original 'red' default)
    folium.PolyLine(walk_coords, color="red", weight=3).add_to(m)

    if annotated and original_coords is not None:
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

    if annotated and steps_for_animal:
        _add_step_boxes(m, walk_coords, steps_for_animal, color="red")

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


def plot_trajectory_collection(traj_coll: mpd.TrajectoryCollection, save_path="walks/"):
    """
    Plot each trajectory in a MovingPandas TrajectoryCollection using Folium.
    Saves:
        - one HTML file per trajectory
        - one combined map with all trajectories
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if len(traj_coll.trajectories) == 0:
        center = (0.0, 0.0)
    else:
        p = traj_coll.trajectories[0].df.geometry.iloc[0]
        center = (p.y, p.x)

    # combined map
    m_all = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    for idx, traj in enumerate(traj_coll.trajectories):
        traj_id = str(traj.id)
        color = colors[idx % len(colors)]
        coords = [(pt.y, pt.x) for pt in traj.df.geometry]

        # save individual trajectory
        m_single = folium.Map(location=coords[0], zoom_start=14, tiles="OpenStreetMap")
        folium.PolyLine(coords, color=color, weight=4, opacity=0.8).add_to(m_single)
        folium.Marker(coords[0], tooltip=f"{traj_id} Start").add_to(m_single)
        folium.Marker(coords[-1], tooltip=f"{traj_id} End").add_to(m_single)

        m_single.save(str(save_path / f"{traj_id}.html"))

        # add to combined map
        folium.PolyLine(coords, color=color, weight=3, opacity=0.8).add_to(m_all)

    out_file = save_path / "all_trajectories.html"
    m_all.save(str(out_file))

    return out_file


def plot_trajectory_collection_timed(traj_coll, save_path="walks/"):
    """
    Create a TimeDimension animated map for a MovingPandas TrajectoryCollection.
    One animated layer per trajectory.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if len(traj_coll.trajectories) == 0:
        return None

    # Center of first trajectory
    p = traj_coll.trajectories[0].df.geometry.iloc[0]
    center = (p.y, p.x)

    m = folium.Map(location=center, zoom_start=14, tiles="Esri.WorldImagery")

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
        ci += 1
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

    output = save_path / "trajectories_timed.html"
    m.save(str(output))
    return output
