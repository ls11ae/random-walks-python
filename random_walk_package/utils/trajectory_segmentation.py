"""Compatibility wrapper for trajectory segmentation helpers."""

from segmentationcma import (
    Segment,
    SegmentationCriterion,
    UTMDistanceCriterion,
    annotate_segments_dataframe,
    annotate_trajectory_collection,
    bbox_of_segment,
    make_overlapping,
    merge_singletons,
    segment_dataframe,
    segment_trajectory_collection,
    trajectory_segments,
)

__all__ = [
    "Segment",
    "SegmentationCriterion",
    "UTMDistanceCriterion",
    "annotate_segments_dataframe",
    "annotate_trajectory_collection",
    "bbox_of_segment",
    "make_overlapping",
    "merge_singletons",
    "segment_dataframe",
    "segment_trajectory_collection",
    "trajectory_segments",
]
