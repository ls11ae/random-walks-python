from __future__ import annotations


def segments_for_steps(
        steps,
        *,
        max_cell_size: int,
        resolution: int,
        criterion=None,
        segment_col="segment_id",
        merge_single_point_segments=True,
):
    from segmentationcma import UTMDistanceCriterion, annotate_segments_dataframe, make_overlapping, segment_dataframe

    if criterion is None:
        criterion = UTMDistanceCriterion.from_cell_grid(max_cell_size, resolution)

    base_segments = segment_dataframe(
        steps,
        criterion,
        merge_single_point_segments=merge_single_point_segments,
    )
    annotated = annotate_segments_dataframe(steps, segments=base_segments, segment_col=segment_col)
    steps[segment_col] = annotated[segment_col]
    return make_overlapping(base_segments)
