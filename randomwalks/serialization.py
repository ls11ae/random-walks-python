from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from randomwalks import TerrainMapHandle


@dataclass
class SerializedWalk:
    width: int
    height: int
    walk: np.ndarray
    steps: np.ndarray | None = None
    terrain: np.ndarray | None = None
    utilization_distribution: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def start(self):
        return tuple(map(int, self.walk[0])) if self.walk.size else None

    @property
    def end(self):
        return tuple(map(int, self.walk[-1])) if self.walk.size else None

    def to_json(self, json_file, *, indent=2):
        return walk_to_json(
            self.walk,
            json_file,
            steps=self.steps,
            terrain=self.terrain,
            ud=self.utilization_distribution,
            width=self.width,
            height=self.height,
            metadata=self.metadata,
            indent=indent,
        )


def walk_to_json(
        walk,
        json_file,
        *,
        steps=None,
        terrain=None,
        terrain_map=None,
        ud=None,
        width=None,
        height=None,
        W=None,
        H=None,
        metadata=None,
        indent=2,
):
    terrain = terrain if terrain is not None else terrain_map
    width = width if width is not None else W
    height = height if height is not None else H

    walk_array = _points_array(walk, name="walk")
    steps_array = None if steps is None else _points_array(steps, name="steps")
    terrain_array = _optional_array(terrain, dtype=int)
    ud_array = _optional_array(ud, dtype=float)

    if terrain_array is not None:
        height, width = terrain_array.shape
    elif width is None or height is None:
        width, height = _infer_dimensions(walk_array, steps_array, ud_array)

    if width is None or height is None:
        raise ValueError("width/height are required when no terrain is provided")

    payload = {
        "Height": int(height),
        "Width": int(width),
        "Start Point": _point_record(walk_array[0]) if walk_array.size else None,
        "End Point": _point_record(walk_array[-1]) if walk_array.size else None,
        "Walk": _point_records(walk_array),
    }
    if steps_array is not None:
        payload["Steps"] = _point_records(steps_array)
    if terrain_array is not None:
        payload["Terrain"] = terrain_array.astype(int).tolist()
    if ud_array is not None:
        payload["Utilization Distribution"] = ud_array.astype(float).tolist()
    if metadata is not None:
        payload["metadata"] = dict(metadata)

    path = Path(json_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent)
        handle.write("\n")
    return path


def walk_from_json(json_file):
    path = Path(json_file)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    walk = _points_array(_get_any(payload, "Walk", "walk"), name="walk")
    steps_raw = _get_any(payload, "Steps", "steps", default=None)
    steps = None if steps_raw is None else _points_array(steps_raw, name="steps")

    terrain_raw = _get_any(payload, "Terrain", "terrain", default=None)
    terrain = None if terrain_raw is None else np.asarray(terrain_raw, dtype=int)

    ud_raw = _get_any(
        payload,
        "Utilization Distribution",
        "utilization_distribution",
        "ud",
        default=None,
    )
    utilization_distribution = None if ud_raw is None else np.asarray(ud_raw, dtype=float)

    width = _get_any(payload, "Width", "width", "W", default=None)
    height = _get_any(payload, "Height", "height", "H", default=None)
    if terrain is not None:
        height, width = terrain.shape
    elif width is None or height is None:
        width, height = _infer_dimensions(walk, steps, utilization_distribution)

    if width is None or height is None:
        raise ValueError("JSON walk does not contain width/height or terrain")

    return SerializedWalk(
        width=int(width),
        height=int(height),
        walk=walk,
        steps=steps,
        terrain=terrain,
        utilization_distribution=utilization_distribution,
        metadata=dict(payload.get("metadata", {})),
    )


def plot_walk_from_json(json_file, *, title=None, show=True, ax=None, show_legend=True):
    from randomwalks import plot_terrain_walk

    serialized = walk_from_json(json_file)
    return plot_terrain_walk(
        terrain=serialized.terrain,
        walk=serialized.walk,
        steps=serialized.steps,
        ud=serialized.utilization_distribution,
        width=serialized.width,
        height=serialized.height,
        title=title,
        show=show,
        ax=ax,
        show_legend=show_legend,
    )


def _optional_array(value, *, dtype):
    if value is None:
        return None
    if isinstance(value, TerrainMapHandle):
        return value.to_numpy()
    if hasattr(value, "to_numpy"):
        return np.asarray(value.to_numpy(), dtype=dtype)
    return np.asarray(value, dtype=dtype)


def _points_array(points, *, name):
    if points is None:
        raise ValueError(f"{name} is required")

    if isinstance(points, SerializedWalk):
        points = points.walk

    if _looks_like_point_records(points):
        array = np.array([(point["x"], point["y"]) for point in points], dtype=np.int64)
    else:
        array = np.asarray(points, dtype=np.int64)

    if array.ndim == 1:
        if array.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        if array.size != 2:
            raise ValueError(f"{name} must be a sequence of (x, y) points")
        array = array.reshape(1, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must be a sequence of (x, y) points")
    return array


def _looks_like_point_records(points):
    return (
            isinstance(points, list)
            and bool(points)
            and isinstance(points[0], dict)
            and "x" in points[0]
            and "y" in points[0]
    )


def _point_record(point):
    return {"x": int(point[0]), "y": int(point[1])}


def _point_records(points):
    return [_point_record(point) for point in np.asarray(points, dtype=np.int64)]


def _infer_dimensions(walk, steps=None, ud=None):
    if ud is not None and ud.ndim >= 2:
        return int(ud.shape[1]), int(ud.shape[0])

    arrays = [array for array in (walk, steps) if array is not None and array.size]
    if not arrays:
        return None, None

    points = np.vstack(arrays)
    return int(points[:, 0].max()) + 1, int(points[:, 1].max()) + 1


def _get_any(payload, *keys, default=None):
    for key in keys:
        if key in payload:
            return payload[key]
    return default
