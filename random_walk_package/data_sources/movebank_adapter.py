import math


def padded_bbox(min_lon_raw, min_lat_raw, max_lon_raw, max_lat_raw, padding: float = 0.2) -> tuple[
    float, float, float, float]:
    lon_range = max(max_lon_raw - min_lon_raw, 0.0)
    lat_range = max(max_lat_raw - min_lat_raw, 0.0)

    # Handle degenerate cases by inflating a tiny range (avoids division by zero later)
    if lon_range == 0.0:
        lon_range = 1e-6
        min_lon_raw -= lon_range / 2.0
        max_lon_raw += lon_range / 2.0
    if lat_range == 0.0:
        lat_range = 1e-6
        min_lat_raw -= lat_range / 2.0
        max_lat_raw += lat_range / 2.0

    lon_pad = lon_range * padding
    lat_pad = lat_range * padding

    min_lon = min_lon_raw - lon_pad
    max_lon = max_lon_raw + lon_pad
    min_lat = min_lat_raw - lat_pad
    max_lat = max_lat_raw + lat_pad

    return min_lon, min_lat, max_lon, max_lat


def clamp_lonlat_bbox(bbox):
    min_x, min_y, max_x, max_y = bbox
    return (
        max(min_x, -180.0),
        max(min_y, -90.0),
        min(max_x, 180.0),
        min(max_y, 90.0),
    )


def bbox_to_discrete_space(bbox: tuple[float, float, float, float], samples: int) -> tuple[int, int, int, int]:
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat

    # Guard tiny/zero extents
    lon_range = lon_range if lon_range != 0.0 else 1e-12
    lat_range = lat_range if lat_range != 0.0 else 1e-12

    # Use ground-distance-aware aspect ratio (meters), not degrees
    mean_lat = (min_lat + max_lat) / 2.0
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(mean_lat))

    width_m = lon_range * meters_per_deg_lon
    height_m = lat_range * meters_per_deg_lat
    aspect_ratio_m = width_m / height_m

    # Optional: also compute degree aspect for logging/diagnostics
    aspect_ratio_deg = lon_range / lat_range
    print(f"aspect ratio (deg): {aspect_ratio_deg}")
    print(f"aspect ratio (meters): {aspect_ratio_m}")

    # Size grid using metric aspect so it matches GeoTIFF appearance
    if width_m >= height_m:
        x_res = samples
        y_res = max(1, round(samples / aspect_ratio_m))
    else:
        y_res = samples
        x_res = max(1, round(samples * aspect_ratio_m))

    return 0, 0, x_res, y_res
