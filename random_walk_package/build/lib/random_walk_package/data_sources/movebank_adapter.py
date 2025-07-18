import ctypes

import numpy as np
import pandas as pd

from random_walk_package import dll
from random_walk_package.bindings.data_processing.movebank_parser import Coordinate_array, Coordinate


def get_unique_animal_ids(df: pd.DataFrame) -> list:
    """Return list of unique animal IDs from the DataFrame."""
    return df['tag-local-identifier'].unique().tolist()


def get_bounding_box(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """Return bounding box as (min_lon, max_lon, min_lat, max_lat)."""
    coords = df[['location-long', 'location-lat']].dropna()
    return (
        coords['location-long'].min(),
        coords['location-lat'].min(),
        coords['location-long'].max(),
        coords['location-lat'].max()
    )


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


def get_animal_coordinates(df: pd.DataFrame, animal_id: str, samples: int = None, width=100, height=100):
    """Return coordinates and timestamps for a specific animal.
    If samples is provided, returns equidistant samples."""
    # Filter and clean data
    animal_df = df[df['tag-local-identifier'] == animal_id]
    clean_df = animal_df[['timestamp', 'location-long', 'location-lat']].dropna()

    # Handle sampling
    if samples is not None and samples > 0:
        step = max(1, len(clean_df) // samples)
        clean_df = clean_df.iloc[::step].head(samples)

    # Create Coordinate array
    num_points = len(clean_df)
    min_lon, max_lon = clean_df['location-long'].min(), clean_df['location-long'].max()
    min_lat, max_lat = clean_df['location-lat'].min(), clean_df['location-lat'].max()

    mapped_coords = []
    for i, (_, row) in enumerate(clean_df.iterrows()):
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
    c_array = (Coordinate * arr.length)()

    # Copy the data
    for i in range(arr.length):
        print(str(arr.points[i].x) + ":" + str(arr.points[i].y))
        c_array[i].x = arr.points[i].x
        c_array[i].y = arr.points[i].y

    # Create a new Coordinate_array that owns its memory
    result = dll.coordinate_array_new(ctypes.pointer(Coordinate_array(c_array)), arr.length)

    return result
