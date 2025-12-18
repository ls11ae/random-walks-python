import json
from collections import Counter

import numpy as np


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
    state_dict = {}
    for seq in seq_dfs:
        if 'state' in seq.columns and 'timestamp' in seq.columns:
            for _, row in seq.iterrows():
                timestamp = row['timestamp']
                state = row['state']
                state_dict[timestamp] = state

    print(f"Anzahl States gefunden: {len(state_dict)}")

    if not state_dict:
        gdf['state'] = -1
        return gdf

    gdf_with_states = gdf.copy()

    # assign states
    state_count = 0
    for idx, row in gdf_with_states.iterrows():
        if idx in state_dict:
            gdf_with_states.at[idx, 'state'] = state_dict[idx]
            state_count += 1

    print(f"States zugewiesen: {state_count} von {len(gdf_with_states)} Punkten")

    # temporary gdf
    temp_df = gdf_with_states.reset_index()
    temp_df['state'] = temp_df['state'].astype('float')  # für NaN-Handling

    # sort by animal and time
    temp_df = temp_df.sort_values(by=[columns.id_col, columns.time_col])

    # Interpolation per animal
    temp_df['state'] = temp_df.groupby(columns.id_col)['state'].ffill()
    temp_df['state'] = temp_df.groupby(columns.id_col)['state'].bfill()

    temp_df['state'] = temp_df['state'].fillna(-1).astype(int)
    import geopandas as gpd
    gdf_with_states = gpd.GeoDataFrame(
        temp_df.drop(columns='geometry'),
        geometry=temp_df['geometry'],
        crs=gdf.crs
    )
    gdf_with_states = gdf_with_states.set_index(columns.time_col)

    print(f"\nFinal State-Distribution:")
    print(gdf_with_states['state'].value_counts().sort_index())

    return gdf_with_states
