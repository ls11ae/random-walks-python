import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import json



from random_walk_package.bindings.data_structures import kernels
from random_walk_package.core.BiasedWalker import BiasedWalker
from random_walk_package.core.CorrelatedWalker import *
from random_walk_package.bindings.mixed_walk import *
from random_walk_package.bindings.plotter import *
from random_walk_package import matrix_generator_gaussian_pdf
from random_walk_package.bindings.brownian_walk import *




from sklearn.mixture import GaussianMixture

#!/usr/bin/env python3
"""
Parallelized version of your script: the segment loop
    for i in range(0, len(bettong) - 1)
is executed in parallel using ProcessPoolExecutor on Linux.

Key design points:
- Avoid passing un-pickleable objects (e.g., kernels) to workers.
- Use Linux 'fork' start method and module-level globals inherited by workers.
- Worker takes only an integer index i (cheap to pickle).
"""

import os
import json
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from random_walk_package.bindings.data_structures import kernels as _kernels_mod  # kept if you need it elsewhere
from random_walk_package.core.BiasedWalker import BiasedWalker  # kept if you need it elsewhere
from random_walk_package.core.CorrelatedWalker import CorrelatedWalker
from random_walk_package.bindings.mixed_walk import *
from random_walk_package.bindings.plotter import *
from random_walk_package import matrix_generator_gaussian_pdf
from random_walk_package.bindings.brownian_walk import *

# -------------------------------------------------------------------
# IDs and date formats
# -------------------------------------------------------------------
md = {
    "lagartha", "bjorn", "baldur", "sifa", "floki", "freya", "andive",
    "beetroot", "durian", "parsnip", "potato", "pumpkin", "raddish", "sprout",
    "swede", "turnip", "tomato"
}
dm = {"dot", "edwina", "egbert", "maud", "olga", "othello", "percy", "renet"}

# -------------------------------------------------------------------
# CSV processing
# -------------------------------------------------------------------
def process_bettongs(file_path: str):
    data = pd.read_csv(file_path)
    bettongs = {}

    for _, row in data.iterrows():
        bettong_id = row["ID"]

        if bettong_id in md:
            date_time_obj = datetime.strptime(row["datetime"], "%m/%d/%y %H:%M")
        elif bettong_id in dm:
            date_time_obj = datetime.strptime(row["datetime"], "%d/%m/%y %H:%M")
        else:
            raise ValueError(f"UNKNOWN ID: {bettong_id}")

        state = row["states"]

        if bettong_id not in bettongs:
            bettongs[bettong_id] = []
        bettongs[bettong_id].append((int(row["x"]), int(row["y"]), date_time_obj, state))

    for bettong_id in bettongs:
        bettongs[bettong_id].sort(key=lambda entry: entry[2])

    return bettongs


def calculate_durations(bettongs):
    durations = []
    for _, entries in bettongs.items():
        for i in range(1, len(entries)):
            time_diff = entries[i][2] - entries[i - 1][2]
            durations.append(time_diff.total_seconds() / 60)
    return durations


def delta_vector(a, b, step_size=1):
    d = np.array(a) - np.array(b)
    return d / step_size


def calculate_steps_grouped(bettongs, step_size=1):
    steps = [[], [], []]
    count_total = 0
    count_discarded = 0

    for _, entries in bettongs.items():
        for i in range(1, len(entries)):
            count_total += 1
            time_diff_0 = (entries[i][2] - entries[i - 1][2]).total_seconds() / 60
            if time_diff_0 == 15:
                a = (entries[i - 2][0], entries[i - 2][1])
                b = (entries[i - 1][0], entries[i - 1][1])
                steps[entries[i - 1][3] - 1].append(delta_vector(b, a, step_size))
            else:
                count_discarded += 1

    if count_total > 0:
        print(f"  total of {count_total} steps, {count_discarded / count_total}% discarded")

    return steps


# -------------------------------------------------------------------
# Heatmaps / GMM fitting
# -------------------------------------------------------------------
rnge = 40
reso = rnge * 2 + 1

def fit_data(axs, steps):
    data = np.array(steps)
    n_components = 3

    gmm = GaussianMixture(n_components=n_components, covariance_type="full")
    gmm.fit(data)

    x = np.linspace(-rnge, rnge, reso)
    y = np.linspace(-rnge, rnge, reso)
    X, Y = np.meshgrid(x, y)
    grid = np.column_stack([X.ravel(), Y.ravel()])

    log_density = gmm.score_samples(grid)
    density = np.exp(log_density)
    Z = density.reshape(X.shape)

    axs.imshow(
        Z,
        extent=(-rnge, rnge, -rnge, rnge),
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    return Z


def generate_heatmap(axs, coords):
    coords = np.array(coords)

    x_edges = np.linspace(-rnge, rnge, reso)
    y_edges = np.linspace(-rnge, rnge, reso)

    heatmap, xedges, yedges = np.histogram2d(coords[:, 0], coords[:, 1], bins=[x_edges, y_edges])

    axs.imshow(
        heatmap.T,
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        origin="lower",
        cmap="viridis",
    )


def pure_grouped(bettongs, step_size=1):
    a, b, c = calculate_steps_grouped(bettongs, step_size)

    fig, axs = plt.subplots(2, 3, figsize=(12, 6))

    generate_heatmap(axs[0, 0], a)
    generate_heatmap(axs[0, 1], b)
    generate_heatmap(axs[0, 2], c)

    m_a = fit_data(axs[1, 0], a)
    m_b = fit_data(axs[1, 1], b)
    m_c = fit_data(axs[1, 2], c)

    linked_pairs = [(axs[0, 0], axs[1, 0]), (axs[0, 1], axs[1, 1]), (axs[0, 2], axs[1, 2])]

    def on_xlim_changed(event_ax):
        global updating
        if updating:
            return

        updating = True
        for ax1, ax2 in linked_pairs:
            if event_ax == ax1:
                ax2.set_xlim(ax1.get_xlim())
                ax2.set_ylim(ax1.get_ylim())
            elif event_ax == ax2:
                ax1.set_xlim(ax2.get_xlim())
                ax1.set_ylim(ax2.get_ylim())

        fig.canvas.draw_idle()
        updating = False

    for ax1, ax2 in linked_pairs:
        ax1.callbacks.connect("xlim_changed", on_xlim_changed)
        ax1.callbacks.connect("ylim_changed", on_xlim_changed)
        ax2.callbacks.connect("xlim_changed", on_xlim_changed)
        ax2.callbacks.connect("ylim_changed", on_xlim_changed)

    plt.tight_layout()
    return m_a, m_b, m_c


def points_to_np_float64(points, n: int) -> np.ndarray:
    """
    Convert `points` to a float64 ndarray WITHOUT using the buffer protocol.
    This avoids PEP 3118 pointer-format errors like '&<d'.
    """
    # Fast path if it is iterable (or sequence); Python will fall back to __getitem__ iteration if needed.
    try:
        arr = np.fromiter(points, dtype=np.float64, count=n)
        if arr.size == n:
            return arr
    except TypeError:
        pass

    # Robust path: force float conversion elementwise (handles wrapped C++ scalar types)
    return np.fromiter((float(points[i]) for i in range(n)), dtype=np.float64, count=n)


# -------------------------------------------------------------------
# Parallel segment computation (module-level globals + fork inheritance)
# -------------------------------------------------------------------
G_BETTONG = None
G_KERNELS = None
G_W = None
G_H = None
G_T = None
G_MIN_X = None
G_MIN_Y = None
G_PADDING = None

def compute_segment_util(i: int):
    
    x0, y0, t0, s0 = G_BETTONG[i]
    x1, y1, t1, s1 = G_BETTONG[i + 1]

    if (t1 - t0).total_seconds() / 60 != 15:
        print(f"Skipping segment {i+1} due to non 15 minute interval between {t0} and {t1}.")
        return None
    
    sx, sy = x0 - G_MIN_X + G_PADDING, y0 - G_MIN_Y + G_PADDING
    tx, ty = x1 - G_MIN_X + G_PADDING, y1 - G_MIN_Y + G_PADDING
    
    print(f"Computing segment {i+1}: from ({sx}, {sy}, {t0}) to ({tx}, {ty}, {t1}) with state {s0}...")

    walker = CorrelatedWalker(S=25, kernel=G_KERNELS[s0 - 1], D=1, W=G_W, H=G_H, T=G_T)
    walker.generate(start_x=sx, start_y=sy, use_serialization=False)
    utilization_distribution = walker.utilize(end_x=tx, end_y=ty)

    n = G_W * G_H
    acc = np.zeros(n, dtype=np.float64)

    for t in range(G_T):
        pts = utilization_distribution[t][0].data[0][0].data.points
        acc += points_to_np_float64(pts, n)

    acc /= G_T
    
    print(f"  Finished computing segment {i+1}: from ({sx}, {sy}, {t0}) to ({tx}, {ty}, {t1}) with state {s0}")
    return acc.reshape((G_H, G_W))


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure fork on Linux; also ensures safe multiprocessing entry point.
    mp.set_start_method("fork", force=True)

    updating = False  # used by pure_grouped zoom sync

    file_path = "/home/mart/Code/eco/Gardiner- Habitat.Perception.Bettongs.csv"
    bettongs = process_bettongs(file_path)

    # Fit the step models and create kernels
    m_1, m_2, m_3 = pure_grouped(bettongs, 15)

    kernel_1 = correlated_kernels_from_matrix(m_1, reso, reso, 1)
    kernel_2 = correlated_kernels_from_matrix(m_2, reso, reso, 1)
    kernel_3 = correlated_kernels_from_matrix(m_3, reso, reso, 1)
    kernels = [kernel_1, kernel_2, kernel_3]

    # Select a track
    bettong = bettongs["lagartha"][0:3]

    xs, ys = zip(*[(x, y) for x, y, *_ in bettong])
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    padding = 100
    W = (max_x - min_x) + 2 * padding
    H = (max_y - min_y) + 2 * padding
    print(f"W: {W}, H: {H}")

    # Parallel parameters
    T = 15
    total = len(bettong) - 1

    # Set worker globals (inherited by forked processes; avoids pickling kernels)
    G_BETTONG = bettong
    G_KERNELS = kernels
    G_W, G_H, G_T = W, H, T
    G_MIN_X, G_MIN_Y, G_PADDING = min_x, min_y, padding

    total_utilization = np.zeros((H, W), dtype=np.float64)

    # Use explicit fork context
    ctx = mp.get_context("fork")
    max_workers = min(os.cpu_count() or 1, total)

    used = 0
    done = 0

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = [ex.submit(compute_segment_util, i) for i in range(total)]

        for fut in as_completed(futures):
            done += 1
            seg = fut.result()
            if seg is None:
                # print(f"  Skipped segment {done} due to invalid interval.")
                continue
            used += 1
            total_utilization += seg
            # print(f"  Finished {done}/{total} (accepted {used})")

    # Match your previous behavior: divide by total segments (even if some were skipped)
    # If you want to average only valid segments, replace `total` with `max(used, 1)`.
    total_utilization /= max(used, 1)
    

    plot_single_utilisation_matrix(total_utilization, 1, W, H)
