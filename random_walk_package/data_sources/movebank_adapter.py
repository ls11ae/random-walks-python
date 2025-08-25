import ctypes
import math

import pandas as pd

from random_walk_package import dll
from random_walk_package.bindings.data_processing.movebank_parser import Coordinate_array, Coordinate


def get_unique_animal_ids(df: pd.DataFrame) -> list:
    """Return list of unique animal IDs from the DataFrame."""
    return df['tag-local-identifier'].unique().tolist()


def get_bounding_boxes_per_animal(df: pd.DataFrame, padding: float = 0.05) -> dict[
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

        lon_range = max(max_lon_raw - min_lon_raw, 0.0)
        lat_range = max(max_lat_raw - min_lat_raw, 0.0)

        # Handle degenerate cases by inflating a tiny range (avoids division by zero later)
        if lon_range == 0.0:
            print(f"Zero longitude range found for animal {animal_id}, inflating")
            lon_range = 1e-6
            min_lon_raw -= lon_range / 2.0
            max_lon_raw += lon_range / 2.0
        if lat_range == 0.0:
            print(f"Zero latitude range found for animal {animal_id}, inflating")
            lat_range = 1e-6
            min_lat_raw -= lat_range / 2.0
            max_lat_raw += lat_range / 2.0

        lon_pad = lon_range * padding
        lat_pad = lat_range * padding

        min_lon = min_lon_raw - lon_pad
        max_lon = max_lon_raw + lon_pad
        min_lat = min_lat_raw - lat_pad
        max_lat = max_lat_raw + lat_pad

        # Clamp to valid geographic bounds
        min_lon = max(min_lon, -180.0)
        max_lon = min(max_lon, 180.0)
        min_lat = max(min_lat, -90.0)
        max_lat = min(max_lat, 90.0)

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


def get_start_end_dates(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Get the minimum (start) and maximum (end) dates
    start_date = df['timestamp'].min().strftime('%Y-%m-%d')
    end_date = df['timestamp'].max().strftime('%Y-%m-%d')

    return start_date, end_date


def map_lon_to_x(lon, min_lon, max_lon, x_res):
    return int((lon - min_lon) / (max_lon - min_lon) * x_res)


def map_lat_to_y(lat, min_lat, max_lat, y_res):
    return int((max_lat - lat) / (max_lat - min_lat) * y_res)


def get_animal_coordinates(df: pd.DataFrame, animal_id: str, samples: int = None,
                           width=100, height=100, bbox: tuple[float, float, float, float] | None = None):
    """Return coordinates and timestamps for a specific animal.
    If samples is provided, returns equidistant samples.
    If bbox is provided, map using that bbox (must match terrain bbox) for consistent alignment.
    """
    # Filter and clean data
    animal_df = df[df['tag-local-identifier'] == animal_id]
    clean_df = animal_df[['timestamp', 'location-long', 'location-lat']].dropna()

    # Handle sampling
    if samples is not None and samples > 0:
        step = max(1, len(clean_df) // samples)
        clean_df = clean_df.iloc[::step].head(samples)

    # Decide mapping extents: use provided bbox if given, else fallback to per-animal extents
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
    else:
        min_lon, max_lon = clean_df['location-long'].min(), clean_df['location-long'].max()
        min_lat, max_lat = clean_df['location-lat'].min(), clean_df['location-lat'].max()

    mapped_coords = []
    for _, row in clean_df.iterrows():
        x = map_lon_to_x(row['location-long'], min_lon, max_lon, width)
        y = map_lat_to_y(row['location-lat'], min_lat, max_lat, height)
        mapped_coords.append((x, y))

    return mapped_coords