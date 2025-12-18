import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from random_walk_package.core.hmm.utils import angle_diff, detect_typical_interval


class ColumnConfig:
    def __init__(self, id_cols='individual-local-identifier',
                 time_col='timestamp',
                 geom_col='geometry',
                 provided_dir_col='direction',  # degrees
                 feature_cols=('distance', 'angular_difference', 'speed')):
        self.time_col = time_col
        self.geom_col = geom_col
        self.id_col = id_cols
        self.provided_dir_col = provided_dir_col
        self.feature_cols = feature_cols


def preprocess_hmm(gdf, columns: ColumnConfig, scale=True):
    time_col = columns.time_col
    geom_col = columns.geom_col
    id_cols = columns.id_col
    provided_dir_col = columns.provided_dir_col
    feature_cols = columns.feature_cols

    df = gdf.copy()
    df = df.reset_index()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.dropna(subset=[time_col, geom_col])

    grouped = df.groupby(id_cols)
    print(f'Found {len(grouped)} animals')
    seq_dfs = []
    for group_id, group in grouped:
        group = group.sort_values(time_col)
        # delta t (s)
        group[time_col] = pd.to_datetime(group[time_col])
        group['dt'] = group[time_col].diff().dt.total_seconds().fillna(0.0)

        # step size from geometry (m)
        xs = group.geometry.x.values
        ys = group.geometry.y.values
        dx = np.concatenate([[0.0], np.diff(xs)])
        dy = np.concatenate([[0.0], np.diff(ys)])
        group['step_length'] = np.sqrt(dx ** 2 + dy ** 2)

        group['speed'] = group['speed'].fillna(0.0).astype(float)
        # direction of movement (degrees)
        headings = np.deg2rad(group[provided_dir_col].fillna(method='pad').astype(float).values)
        if len(headings) > 1 and np.isnan(headings[0]):
            headings[0] = headings[1]
        turn_angles = np.concatenate(
            [[0.0], [angle_diff(headings[i - 1], headings[i]) for i in range(1, len(headings))]])
        group['turn_angle'] = turn_angles
        group['heading_rad'] = headings

        # split into subsequences where dt > threshold
        local_dts = group['dt'].values[1:]
        dt_mode = np.median(local_dts)
        max_gap_seconds = 5 * dt_mode

        gaps = group['dt'].values
        split_idx = np.where(gaps > max_gap_seconds)[0]
        start = 0
        for idx in split_idx:
            seq = group.iloc[start:idx].copy().reset_index(drop=True)
            if len(seq) >= 2:
                seq_dfs.append(seq)
            start = idx
        # final chunk
        final = group.iloc[start:].copy().reset_index(drop=True)
        if len(final) >= 2:
            seq_dfs.append(final)

    # Build numeric arrays
    arrays = []
    for seq in seq_dfs:
        X = seq[list(feature_cols)].fillna(0.0).to_numpy(dtype=float)
        if X.shape[0] >= 2:
            arrays.append(X)
    scaler = None
    if scale and len(arrays) > 0:
        stacked = np.vstack(arrays)
        scaler = StandardScaler().fit(stacked)
        arrays = [scaler.transform(X) for X in arrays]
    return arrays, scaler, seq_dfs


def process_trajectories(data_list: list[pd.DataFrame]):
    animal_trajectories = {}
    # Iterate over each row in the DataFrame
    print("Length" + str(len(data_list)))

    for data in data_list:
        for _, row in data.iterrows():
            bettong_id = row["individual-local-identifier"]
            if bettong_id not in animal_trajectories:
                animal_trajectories[bettong_id] = []
            animal_trajectories[bettong_id].append(
                (int(row["geometry"].x), int(row["geometry"].y), row["timestamp"], row["state"] + 1))

    # Sort each bettong's list of tuples by the datetime entry
    for bettong_id in animal_trajectories:
        animal_trajectories[bettong_id].sort(key=lambda entry: entry[2])

    from collections import Counter

    all_states = []
    for id, entries in animal_trajectories.items():
        for row in entries:
            all_states.append(row[3])

    print(Counter(all_states))
    intervals = []
    for id, entries in animal_trajectories.items():
        val = detect_typical_interval(entries)
        if val:
            intervals.append(val)
    print("alle intervalle", intervals)
    global_interval = np.median(intervals)
    print("Erkannter Zeitabstand:", global_interval)
    dt_threshold = global_interval

    return animal_trajectories, dt_threshold


def sub(self, animal_trajectories):
    bettong = []

    max_len = -1
    for bettong_id, entries in animal_trajectories.items():
        cur_len = 0
        cur_bettong = []
        # Calculate time differences between consecutive entries
        for i in range(2, len(entries)):
            time_diff_0 = (entries[i][2] - entries[i - 1][2]).total_seconds() / 60
            time_diff_1 = (entries[i - 1][2] - entries[i - 2][2]).total_seconds() / 60
            if abs(time_diff_0 - self.dt_threshold) <= self.dt_threshold * self.dt_tolerance and abs(
                    time_diff_1 - self.dt_threshold) <= self.dt_tolerance * self.dt_threshold:
                # steps.append((entries[i][0]-entries[i-1][0],entries[i][1]-entries[i-1][1]))
                x, y = (entries[i][0], entries[i][1])
                cur_bettong.append((x, y, cur_len))
                cur_len += 1
            else:
                if cur_len > max_len:
                    max_len = max(max_len, cur_len)
                    bettong = cur_bettong
                cur_len = 0
                cur_bettong = []
        if cur_len > max_len:
            max_len = max(max_len, cur_len)
            bettong = cur_bettong

    # print(f"{min([x for x, _, _ in bettong]), max([x for x, _, _ in bettong]), min([y for _, y, _ in bettong]), max([y for _, y, _ in bettong])}")

    print("max_len: ", max_len)
    return {"": bettong}
