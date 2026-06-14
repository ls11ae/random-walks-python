"""Compatibility wrapper for HMM state annotation helpers."""

from hmmcma.utils import angle_diff, detect_typical_interval, merge_states_to_gdf

__all__ = [
    "angle_diff",
    "detect_typical_interval",
    "merge_states_to_gdf",
]
