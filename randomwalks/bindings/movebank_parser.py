from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from pyproj import Transformer

from randomwalks.bindings.data_structures.Terrain import TerrainMapHandle


def df_add_properties2(df: DataFrame,
                       kernel_resolver,  # function (row, start, end) -> KernelParametersPtr
                       terrain: TerrainMapHandle, bbox_geo, grid_width,
                       grid_height, utm_code,
                       time_stamp='timestamp',
                       grid_points_per_edge=5,
                       lon='location-long',
                       lat='location-lat',
                       start=None, end=None,
                       S=None) -> Tuple[pd.DataFrame, int]:
    min_lon, min_lat, max_lon, max_lat = bbox_geo

    # filter by bounding box
    clean_df = df[
        (df[lon] >= min_lon) & (df[lon] <= max_lon) &
        (df[lat] >= min_lat) & (df[lat] <= max_lat)].copy()
    if clean_df.empty:
        return None, 0

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_code}", always_xy=True)

    # transform all coordinates to UTM
    xs, ys = transformer.transform(clean_df[lon].values, clean_df[lat].values)
    clean_df["_x_utm"] = xs
    clean_df["_y_utm"] = ys
    min_x, min_y = transformer.transform(min_lon, min_lat)
    max_x, max_y = transformer.transform(max_lon, max_lat)

    # map lon/lat -> env grid
    gx = ((clean_df["_x_utm"] - min_x) / (max_x - min_x) * grid_width).astype(int)
    gy = ((max_y - clean_df["_y_utm"]) / (max_y - min_y) * grid_height).astype(int)

    gx = gx.clip(0, grid_width - 1)
    gy = gy.clip(0, grid_height - 1)

    # step size in grid
    s_x = grid_width // grid_points_per_edge
    s_y = grid_height // grid_points_per_edge

    # only take env grid points (filter)
    clean_df["x"] = (gx // s_x).clip(0, grid_points_per_edge - 1)
    clean_df["y"] = (gy // s_y).clip(0, grid_points_per_edge - 1)

    # collapse duplicates (same t,x,y)
    clean_df = (
        clean_df
        .groupby([time_stamp, "y", "x"], as_index=False)
        .first()
    )

    # enforce FULL (t,y,x) GRID
    xs = np.arange(grid_points_per_edge)
    ys = np.arange(grid_points_per_edge)
    all_times = clean_df[time_stamp].drop_duplicates().sort_values()

    full_index = pd.MultiIndex.from_product(
        [all_times, ys, xs],
        names=[time_stamp, "y", "x"]
    )

    clean_df = (
        clean_df
        .set_index([time_stamp, "y", "x"])
        .reindex(full_index)
        .reset_index()
    )

    # add terrain info
    clean_df["terrain"] = [
        terrain.at(x, y)
        for x, y in zip(clean_df["x"], clean_df["y"])
    ]

    # compute kernel parameters
    kp = clean_df.apply(
        lambda row: kernel_resolver(row, start, end, S),
        axis=1,
        result_type="expand"
    )
    # apply kernel params
    clean_df[["is_brownian", "S", "D", "len_diffusity", "angle_diffusity", "bias_x", "bias_y"]] = kp

    clean_df.sort_values(
        by=["y", "x", time_stamp],
        inplace=True
    )

    # fill missing via NN (forward/backward fill)
    cols = ["is_brownian", "S", "D", "len_diffusity", "angle_diffusity", "bias_x", "bias_y"]
    clean_df[cols] = (
        clean_df
        .groupby(["y", "x"], sort=False)[cols]
        .transform(lambda s: s.ffill().bfill())
    )

    # final order for C backend
    clean_df = clean_df[
        [time_stamp, "y", "x", "terrain",
         "is_brownian", "S", "D", "len_diffusity", "angle_diffusity", "bias_x", "bias_y"]
    ]
    # sort to match C grid y,x,t order
    clean_df.sort_values(
        by=["y", "x", time_stamp],
        inplace=True
    )

    # sanity check
    T = clean_df[time_stamp].nunique()
    expected = grid_points_per_edge ** 2 * T
    assert len(clean_df) == expected, f"{len(clean_df)} != {expected}"

    return clean_df, T
