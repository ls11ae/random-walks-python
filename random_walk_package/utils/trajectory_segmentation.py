import numpy as np


def trajectory_segments(steps, max_cell_size, resolution):
    if len(steps) == 0:
        return []
    segments = []
    max_radius = max_cell_size * resolution / 4.0
    start_idx = 0
    ref_x = steps["utm_x"].iloc[0]
    ref_y = steps["utm_y"].iloc[0]

    for i in range(1, len(steps)):
        x = steps["utm_x"].iloc[i]
        y = steps["utm_y"].iloc[i]
        dist = np.hypot(ref_x - x, ref_y - y)
        if dist >= max_radius:
            segments.append((start_idx, i - 1))
            start_idx = i
            ref_x, ref_y = x, y

    segments.append((start_idx, len(steps) - 1))
    return segments

def merge_singletons(segments):
    if not segments:
        return segments
    merged = []
    i = 0
    while i < len(segments):
        s, e = segments[i]
        if s == e:
            if merged:
                ps, pe = merged[-1]
                merged[-1] = (ps, e)
            elif i + 1 < len(segments):
                ns, ne = segments[i + 1]
                merged.append((s, ne))
                i += 1
            else:
                merged.append((s, e))
        else:
            merged.append((s, e))
        i += 1

    return merged

def make_overlapping(segments):
    if not segments:
        return []

    out = [segments[0]]
    for i in range(1, len(segments)):
        _, prev_e = out[-1]
        _, cur_e = segments[i]
        out.append((prev_e, cur_e))
    return out

def bbox_of_segment(steps, segment):
    s, e = segment
    min_lon, min_lat = float("inf"), float("inf")
    max_lon, max_lat = float("-inf"), float("-inf")

    for i in range(s, e + 1):
        lon = steps["geo_x"].iloc[i]
        lat = steps["geo_y"].iloc[i]

        min_lon = min(min_lon, lon)
        min_lat = min(min_lat, lat)
        max_lon = max(max_lon, lon)
        max_lat = max(max_lat, lat)

    return min_lon, min_lat, max_lon, max_lat