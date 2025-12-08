import math

import pandas as pd


def get_unique_animal_ids(df: pd.DataFrame) -> list:
    """Return list of unique animal IDs from the DataFrame."""
    return df['tag-local-identifier'].unique().tolist()


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


def get_bounding_boxes_per_animal(df: pd.DataFrame, padding: float = 0.2) -> dict[
    str, tuple[float, float, float, float]]:
    """Return per-animal padded bounding boxes as a dict:
    { animal_id: (min_lon, min_lat, max_lon, max_lat) }.
    `padding` is the fraction to add on each side (e.g., 0.05 = 5%).
    """
    result: dict[str, tuple[float, float, float, float]] = {}

    if 'tag-local-identifier' not in df.columns:
        print("No tag-local-identifier column found in DataFrame")
        return result  # No animal IDs available

    for animal_id, group in df.groupby('tag-local-identifier'):
        print(f"Processing animal {animal_id}")
        coords = group[['location-long', 'location-lat']].dropna()
        if coords.empty:
            print(f"No valid coordinates found for animal {animal_id}")
            continue

        min_lon_raw = coords['location-long'].min()
        max_lon_raw = coords['location-long'].max()
        min_lat_raw = coords['location-lat'].min()
        max_lat_raw = coords['location-lat'].max()

        print(
            f"Raw bounds for {animal_id}: lon={min_lon_raw:.6f} to {max_lon_raw:.6f}, lat={min_lat_raw:.6f} to {max_lat_raw:.6f}")
        min_lon, min_lat, max_lon, max_lat = padded_bbox(min_lon_raw, min_lat_raw, max_lon_raw, max_lat_raw, padding)
        print(f"Final bounds for {animal_id}: lon={min_lon:.6f} to {max_lon:.6f}, lat={min_lat:.6f} to {max_lat:.6f}")
        result[str(animal_id)] = (min_lon, min_lat, max_lon, max_lat)

    print(f"Processed {len(result)} animals")
    return result


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


def bbox_dict_to_discrete_space(
        bbox_dict: dict[str, tuple[float, float, float, float]],
        samples: int
) -> dict[str, tuple[int, int, int, int]]:
    """
    Convert a dict of per-animal bboxes into per-animal discrete space parameters.

    Args:
        bbox_dict: { animal_id: (min_lon, min_lat, max_lon, max_lat) }
        samples: Target resolution along the longer ground-distance dimension.


    Returns:
        { animal_id: (0, 0, x_res, y_res) }
    """
    result: dict[str, tuple[int, int, int, int]] = {}
    for animal_id, bbox in bbox_dict.items():
        result[str(animal_id)] = bbox_to_discrete_space(bbox, samples)
    return result


def get_start_end_dates(df, animal_id: str):
    # Filter the DataFrame first
    mask = df['tag-local-identifier'] == animal_id
    filtered_timestamps = df.loc[mask, 'timestamp']

    # Convert to datetime and get min/max
    timestamps = pd.to_datetime(filtered_timestamps)
    start_date = timestamps.min().strftime('%Y-%m-%d')
    end_date = timestamps.max().strftime('%Y-%m-%d')

    return start_date, end_date


def get_animal_coordinates(df: pd.DataFrame, animal_id: str, epsg_code: str,
                           samples: int = None, width=100, height=100,
                           bbox_utm: tuple[float, float, float, float] | None = None, time_stamped=False):
    """
    Return mapped grid coordinates for a specific animal.
    If samples is provided, returns equidistant samples.
    If bbox_utm is provided, use it (must be in UTM!).
    """
    from pyproj import Transformer

    # prepare transformer
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)

    # filter and clean
    animal_df = df[df['tag-local-identifier'] == animal_id]
    clean_df = animal_df[['timestamp', 'location-long', 'location-lat']].dropna()

    # optional sampling
    if samples is not None and samples > 0:
        step = max(1, len(clean_df) // samples)
        clean_df = clean_df.iloc[::step].head(samples)

    # transform all coordinates to UTM
    geo_coords = clean_df.apply(
        lambda row: [row['location-long'], row['location-lat']], axis=1
    )

    # transform all coordinates to UTM
    utm_coords = clean_df.apply(
        lambda row: transformer.transform(row['location-long'], row['location-lat']), axis=1
    )
    clean_df['x'] = [pt[0] for pt in utm_coords]
    clean_df['y'] = [pt[1] for pt in utm_coords]

    # bounding box: either provided (in UTM) or computed from animal
    if bbox_utm is not None:
        min_x, min_y, max_x, max_y = bbox_utm
    else:
        print("bbox_utm not provided")
        exit(1)

    # map to discrete grid
    mapped_coords = []
    time_stamps = []
    for _, row in clean_df.iterrows():
        # Ensure coordinates are within bounds
        x_utm = max(min(row['x'], max_x), min_x)
        y_utm = max(min(row['y'], max_y), min_y)

        x = int((x_utm - min_x) / (max_x - min_x) * width)
        y = int((max_y - y_utm) / (max_y - min_y) * height)

        # Ensure grid coordinates are within [0, width-1] and [0, height-1]
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))

        mapped_coords.append((x, y))
        time_stamps.append(row['timestamp'])

    if time_stamped:
        return mapped_coords, geo_coords, time_stamps
    else:
        return mapped_coords, geo_coords, None
