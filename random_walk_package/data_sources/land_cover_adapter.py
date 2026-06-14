"""Compatibility wrapper for landcover helpers."""

from environmentcma.landcover import (
    fetch_landcover_data,
    landcover_classes,
    landcover_to_discrete_txt,
)

__all__ = [
    "fetch_landcover_data",
    "landcover_classes",
    "landcover_to_discrete_txt",
]
