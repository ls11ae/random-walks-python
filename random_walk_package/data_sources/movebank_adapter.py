import ctypes

import pandas as pd

from random_walk_package import dll
from random_walk_package.bindings.data_processing.movebank_parser import Coordinate_array, Coordinate


def get_unique_animal_ids(df: pd.DataFrame) -> list:
    """Return list of unique animal IDs from the DataFrame."""
    return df['tag-local-identifier'].unique().tolist()


def get_bounding_box(df: pd.DataFrame, padding: float = 0.05) -> tuple[float, float, float, float]:
    """Return padded bounding box as (min_lon, min_lat, max_lon, max_lat).
    `padding` is the fraction to add on each side (e.g., 0.05 = 5%).
    """
    coords = df[['location-long', 'location-lat']].dropna()
    min_lon_raw = coords['location-long'].min()
    max_lon_raw = coords['location-long'].max()
    min_lat_raw = coords['location-lat'].min()
    max_lat_raw = coords['location-lat'].max()

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

    # Optionally clamp to valid geographic bounds
    min_lon = max(min_lon, -180.0)
    max_lon = min(max_lon, 180.0)
    min_lat = max(min_lat, -90.0)
    max_lat = min(max_lat, 90.0)

    return min_lon, min_lat, max_lon, max_lat


def bbox_to_discrete_space(bbox: tuple[float, float, float, float], samples: int) -> tuple[int, int, int, int]:
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    aspect_ratio = lon_range / lat_range

    if lon_range >= lat_range:
        x_res = samples
        y_res = round(samples / aspect_ratio)
    else:
        y_res = samples
        x_res = round(samples * aspect_ratio)

    return 0, 0, x_res, y_res


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



def get_animal_coordinates_safe(df: pd.DataFrame, animal_id: str):
    """Version that returns pointer for C interop with proper memory management.
       The caller is responsible for freeing the memory using coordinate_array_free."""

    # First get the coordinates as a Python-managed array
    arr = get_animal_coordinates(df, animal_id)

    # Allocate memory that can be freed by the C code
    c_array = (Coordinate * len(arr))()

    # Copy the data
    for i in range(len(arr)):
        c_array[i].x = arr[i].x
        c_array[i].y = arr[i].y

    # Create a new Coordinate_array that owns its memory
    result = dll.coordinate_array_new(ctypes.pointer(Coordinate_array(c_array)), len(arr))

    return result
