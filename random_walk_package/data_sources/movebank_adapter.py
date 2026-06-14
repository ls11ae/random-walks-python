"""Compatibility wrapper for bbox helpers."""

from environmentcma.bounds import (
    bbox_to_discrete_space,
    clamp_lonlat_bbox,
    padded_bbox,
)

__all__ = [
    "bbox_to_discrete_space",
    "clamp_lonlat_bbox",
    "padded_bbox",
]
