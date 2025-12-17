import geopandas as gpd
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from random_walk_package.core.KernelFactory import process_bettongs, pure_cor_grouped, show_fancy


def angle_diff(a, b):
    d = b - a
    return (d + np.pi) % (2 * np.pi) - np.pi


def preprocess_for_hmm(
        gdf: gpd.GeoDataFrame,
        id_cols='individual-local-identifier',
        time_col='timestamp',
        geom_col='geometry',
        provided_dir_col='direction',  # degrees
        feature_cols=('distance', 'angular_difference', 'speed'),  # additional data from workflow
        scale=True
):
    df = gdf.copy()
    df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.dropna(subset=[time_col, geom_col])

    grouped = df.groupby(id_cols)
    print(f'Found {len(grouped)} animals')
    seq_dfs = []
    for group_id, group in grouped:
        group = group.sort_values(time_col).reset_index(drop=True)
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


def apply_hmm(arrays, seq_dfs, n_components=3, plot=False):
    model = GaussianHMM(
        n_components=n_components,
        covariance_type='full',
        n_iter=200,
        random_state=42
    )

    lengths = [arr.shape[0] for arr in arrays]
    stacked = np.vstack(arrays)

    # Fit HMM
    model.fit(stacked, lengths)

    # Predict state sequence (0: sleeping, 2: foraging, 3: traveling)
    stacked_state_seq = model.predict(stacked)
    speed_column_index = 0
    # Sort states by speed
    state_speeds = []
    for k in range(n_components):
        idx = stacked_state_seq == k
        state_speeds.append(np.mean(stacked[idx, speed_column_index]))

    order = np.argsort(state_speeds)
    model_state_mapping = {old: new for new, old in enumerate(order)}

    # add state column to sequence dataframes
    state_seqs = []
    for arr in arrays:
        raw_states = model.predict(arr)
        mapped_states = np.array([model_state_mapping[s] for s in raw_states])
        state_seqs.append(mapped_states)

    for df, states in zip(seq_dfs, state_seqs):
        df["state"] = states

    # sum of gaussians for transition matrix creation
    animal_trajectories = process_bettongs(seq_dfs)
    pure_cor_grouped(animal_trajectories)
    show_fancy(animal_trajectories)
