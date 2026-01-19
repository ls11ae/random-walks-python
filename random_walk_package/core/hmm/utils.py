import json
from collections import Counter

import numpy as np
import pandas as pd
import geopandas as gpd


def to_json(density, num_directions=8, name="test"):
    print(len(density))
    data = {"name": name,
            "d": num_directions,
            "s": int((len(density[0]) - 1) / 2),
            "tm": [list(densit.flatten()) for densit in density]
            }
    with open("noncor.json", 'w') as json_file:
        json.dump(data, json_file, indent=4)


def rotate_vector(a, b, c):
    # Define the vectors
    d = np.array(a) - np.array(b)
    e = np.array(c) - np.array(b)

    # Compute the angle to rotate
    # Angle between d and the horizontal axis
    angle_d = np.arctan2(d[1], d[0])

    theta = -angle_d

    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Rotate vector e
    e_rotated = np.dot(rotation_matrix, e)
    return e_rotated


def angle_diff(a, b):
    d = b - a
    return (d + np.pi) % (2 * np.pi) - np.pi


def generate_angles(num_directions: int) -> list[float]:
    step = 360.0 / num_directions
    return [i * step for i in range(num_directions)]


def detect_typical_interval(entries):
    diffs = []
    for i in range(1, len(entries)):
        delta = ((entries[i][2] - entries[i - 1][2]).total_seconds() + 0.5) // 60
        diffs.append(delta)
    if len(diffs) == 0:
        return None
    # Häufigster Zeitabstand
    mode = Counter(diffs).most_common(1)[0][0]
    return mode


def calculate_durations(animal_trajectories):
    durations = []
    for bettong_id, entries in animal_trajectories.items():
        # Calculate time differences between consecutive entries
        for i in range(1, len(entries)):
            time_diff = entries[i][2] - entries[i - 1][2]
            durations.append(time_diff.total_seconds() // 60)  # Convert to minutes
    return durations


def merge_states_to_gdf(gdf, seq_dfs, columns):
    # extract states
    state_rows = []
    for seq in seq_dfs:
        if {'timestamp', 'state'}.issubset(seq.columns):
            state_rows.append(seq[['timestamp', 'state']])

    if not state_rows:
        gdf['state'] = -1
        return gdf

    states_df = (
        pd.concat(state_rows, ignore_index=True)
        .drop_duplicates(subset='timestamp')
    )

    gdf_tmp = gdf.reset_index()
    gdf_tmp = gdf_tmp.merge(
        states_df,
        on='timestamp',
        how='left'
    )
    assigned = gdf_tmp['state'].notna().sum()
    print(f"States zugewiesen: {assigned} von {len(gdf_tmp)} Punkten")

    gdf_tmp = gdf_tmp.sort_values(
        by=[columns.id_col, columns.time_col]
    )

    gdf_tmp['state'] = (
        gdf_tmp
        .groupby(columns.id_col)['state']
        .ffill()
        .bfill()
        .fillna(-1)
        .astype(int)
    )
    gdf_out = gpd.GeoDataFrame(
        gdf_tmp,
        geometry=columns.geom_col,
        crs=gdf.crs
    ).set_index(columns.time_col)

    print("\nFinal State-Distribution:")
    print(gdf_out['state'].value_counts().sort_index())

    return gdf_out
