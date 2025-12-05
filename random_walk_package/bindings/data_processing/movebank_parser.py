import datetime

import pandas as pd
from pandas import DataFrame
from pyproj import Transformer

from random_walk_package.bindings.data_structures.types import *
from random_walk_package.wrapper import dll

dll.kernel_parameters_create.argtypes = [c_bool,  # is brownian?
                                         c_ssize_t,  # step size
                                         c_ssize_t,  # directions
                                         c_float,  # diffusity
                                         c_ssize_t,  # max bias x
                                         c_ssize_t]  # max bias y
dll.kernel_parameters_create.restype = KernelParametersPtr


def create_kernel_parameters(is_brownian: bool, step_size: int, directions: int, diffusity: float, max_bias_x: int,
                             max_bias_y: int) -> KernelParametersPtr:
    return dll.kernel_parameters_create(is_brownian, step_size, directions, diffusity, max_bias_y, max_bias_x)


def compute_kernel(landmark, row):
    x, y, t, is_brownian, S, D, diffusity, bias_x, bias_y = 0, 1, 2, 3, 4, 5, 6, 7, 8
    return x, y, t, is_brownian, S, D, diffusity, bias_x, bias_y


def df_add_properties(df: DataFrame,
                      kernel_resolver,  # function (landmark, row) -> KernelParametersPtr
                      terrain: TerrainMapPtr, bbox_geo, grid_width,
                      grid_height, utm_code, start_date: datetime, end_date: datetime,
                      time_stamp='timestamp',
                      grid_points_per_edge=5,
                      lon='location-long',
                      lat='location-lat'):
    """
    Add random walk relevant properties to a DataFrame within a specific geographic bounding box.

    This function takes a DataFrame with geospatial + temporal data, and additional parameters that influence animal motion
    and processes it to include the required properties for random walks. Properties include
    grid coordinates, terrain information, and kernel parameters, among others. The input
    data is filtered and transformed to adopt a uniform scale and grid system, making it
    suitable for further geospatial analysis or modeling.

    Parameters:
        df (DataFrame): Input DataFrame containing geospatial and temporal data.
        kernel_resolver: Function that takes a terrain type and a row of the DataFrame as input and returns a KernelParametersPtr.
        terrain (TerrainMapPtr): Reference to the terrain object for retrieving terrain
            information.
        bbox_geo: Geographic bounding box coordinates as (min_lon, min_lat, max_lon, max_lat).
        grid_width: Number of horizontal cells in the grid.
        grid_height: Number of vertical cells in the grid.
        utm_code: UTM coordinate system code as a string.
        start_date (datetime): Starting date for temporal range filtering.
        end_date (datetime): Ending date for temporal range filtering.
        time_stamp: Column name in the DataFrame for the timestamp values. Defaults to 'timestamp'.
        grid_points_per_edge: Number environmental data samples along each dimension. Defaults to 5 (25 env entries).
        lon: Column name in the DataFrame for longitude values. Defaults to 'location-long'.
        lat: Column name in the DataFrame for latitude values. Defaults to 'location-lat'.
        T: Optional. Total number of sampled time steps to retain. Defaults to None.

    Returns:
        DataFrame: A new DataFrame with added properties including grid coordinates, terrain
        information, and kernel parameters.

    Raises:
        KeyError: Raised if required columns are missing from the input DataFrame.
        ValueError: Raised if malformed inputs or mismatched dimensions are provided.

    Notes:
        The input DataFrame is expected to have columns for spatial and temporal information
        such as longitude, latitude, and timestamp. Missing or null values in these columns
        are dropped. The function also performs filtering based on the provided geographic
        bounding box to exclude data outside the defined bounds.
    """
    clean_df = df.dropna()

    # filter by time
    clean_df[time_stamp] = pd.to_datetime(clean_df[time_stamp], errors='coerce')
    clean_df = clean_df[(clean_df[time_stamp] >= start_date) & (clean_df[time_stamp] <= end_date)]

    min_lon, min_lat, max_lon, max_lat = bbox_geo

    # filter by bounding box
    clean_df = clean_df[
        (clean_df[lon] >= min_lon) & (clean_df[lon] <= max_lon) &
        (clean_df[lat] >= min_lat) & (clean_df[lat] <= max_lat)]

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_code}", always_xy=True)

    # transform all coordinates to UTM
    utm_xy = clean_df.apply(lambda row: transformer.transform(row[lon], row[lat]), axis=1)
    min_x, min_y = transformer.transform(min_lon, min_lat)
    max_x, max_y = transformer.transform(max_lon, max_lat)

    def utm_to_grid(pt):
        x_utm, y_utm = pt

        if pd.isna(x_utm) or pd.isna(y_utm):
            return 0, 0

        if max_x == min_x or max_y == min_y:
            return 0, 0

        gx = int((x_utm - min_x) / (max_x - min_x) * grid_width)
        gy = int((max_y - y_utm) / (max_y - min_y) * grid_height)

        return (
            max(0, min(grid_width - 1, gx)),
            max(0, min(grid_height - 1, gy)),
        )

    # utm -> grid
    grid_coords = utm_xy.apply(utm_to_grid)

    s_x = grid_width // grid_points_per_edge
    s_y = grid_height // grid_points_per_edge

    # add landcover info
    clean_df['terrain'] = [terrain_at(terrain, x, y) for x, y in grid_coords]

    # add grid coordinates to df
    clean_df['x'] = [pt[0] // s_x for pt in grid_coords]
    clean_df['y'] = [pt[1] // s_y for pt in grid_coords]

    # your custom geo data to kernel parameters conversion
    clean_df[['is_brownian', 'S', 'D', 'diffusity', 'bias_x', 'bias_y']] = clean_df.apply(
        lambda row: kernel_resolver(row['terrain'], row),
        axis=1, result_type='expand')
    clean_df = clean_df[[time_stamp, 'x', 'y', 'terrain', 'is_brownian', 'S', 'D', 'diffusity', 'bias_x', 'bias_y']]

    return clean_df
