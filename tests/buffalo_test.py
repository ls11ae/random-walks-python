import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from random_walk_package.bindings.data_structures import kernel_context
from random_walk_package.bindings.data_structures.kernel_context import kernel_context_pool
from random_walk_package.bindings.data_structures.kernel_terrain_mapping import create_brownian_kernel_parameters, \
    set_landmark_mapping, create_correlated_kernel_parameters, set_forbidden_landmark
from random_walk_package.bindings.data_structures.terrain import *
from random_walk_package.bindings.plotter import plot_walk_terrain, ud_isopleth_mask
from random_walk_package.core.MixedWalker import *

HAB_CSV = "tests/buf_data/buffalo_habitat.csv"
TRAJ_CSV = "tests/buf_data/buffalo_trajectory.csv"

# Your column names:
HAB_CLASS_COL = "habi.asc"
HAB_X_COL = "s1"
HAB_Y_COL = "s2"


def add_with_offset(total_arr, seg_arr, dx, dy):
    Ht, Wt = total_arr.shape
    Hs, Ws = seg_arr.shape

    x0 = max(dx, 0)
    y0 = max(dy, 0)
    x1 = min(dx + Ws, Wt)
    y1 = min(dy + Hs, Ht)
    if x0 >= x1 or y0 >= y1:
        return

    sx0 = x0 - dx
    sy0 = y0 - dy
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)

    total_arr[y0:y1, x0:x1] += seg_arr[sy0:sy1, sx0:sx1]


def points_to_np_float64(points, n: int) -> np.ndarray:
    try:
        arr = np.fromiter(points, dtype=np.float64, count=n)

        if arr.size == n:
            return arr
    except TypeError:

        pass
    return np.fromiter((float(points[i]) for i in range(n)), dtype=np.float64, count=n)


def main():
    terrain_map = parse_terrain(
        str("/home/omar/PycharmProjects/RW-Python-gitlab/tests/landcover_1F5B2F1 (4118)_16.80_48.12_16.86_48.15_600.txt"),
        " ")
    W = width = terrain_map.width
    H = height = terrain_map.height

    kernel_mapping = create_mixed_kernel_parameters(
        animal_type=Animal.TERRESTRIAL,
        base_step_size=7,
    )
    set_landmark_mapping(
        kernel_mapping,
        GRASSLAND,
        is_brownian=False,
        step_size=5,
        directions=8,
        len_diffusity=0.66
    )
    set_landmark_mapping(
        kernel_mapping,
        TREE_COVER,
        is_brownian=True,
        step_size=6,
        len_diffusity=0.8,
        directions=1,
    )
    set_landmark_mapping(
        kernel_mapping,
        CROPLAND,
        is_brownian=False,
        step_size=6,
        directions=8,
        angle_diffusity=0.1
    )
    # set_forbidden_landmark(kernel_mapping, BUILT_UP)

    rel_points = [
        (0.907, 0.944),
        (0.587, 0.621),
        (0.800, 0.187),
        (0.244, 0.236),
        (0.455, 0.404),
    ]

    """points = [
        (200, 200),
        (230, 180),
        (170, 220),
        (190, 233),
        (240, 210),
        (160, 190),
        (131, 220),
        (210, 165)
    ]"""
    points = [
        # ── bestehende 10 ──────────────────────────────────────────
        (570, 560),
        (530, 510),
        (470, 480),
        (390, 450),
        (310, 410),
        (260, 340),
        (300, 260),
        (230, 200),
        (150, 160),
        (80, 100),

        # ── neue 10 ────────────────────────────────────────────────
        (150, 78),  # 11  dist ≈  73  – offenes Land (40)
        (230, 52),  # 12  dist ≈  84  – Cropland obere Kante (40/30)
        (295, 35),  # 13  dist ≈  67  – Übergang grau/urban (50)
        (350, 29),  # 14  dist ≈  55  – Waypoint oben-mitte (50/40)
        (337, 117),  # 15  dist ≈  89  – zurück in Wald (10)
        (317, 204),  # 16  dist ≈  89  – helles Grün (40)
        (300, 292),  # 17  dist ≈  90  – Mitte, rundes Gewässer-Rand (90)
        (283, 380),  # 18  dist ≈  90  – Grünland Mitte (30)
        (270, 468),  # 19  dist ≈  89  – Fluss-Nähe (80)
        (267, 558),  # 20  dist ≈  90  – Endpunkt unten-mitte (30/10)
    ]

    test = 2
    T = 50
    resolution = 400

    total_utilization = np.zeros((H, W), dtype=np.float64)
    home_range_mask = np.zeros((H, W), dtype=bool)
    kernels_context = kernel_context_pool(terrain_map, kernel_mapping, Reachability.SOFT)
    plot_walk_terrain(terrain_map, points, W, H)
    print(points)
    full_path = np.empty((0, 2))

    for i in range(len(points) - 1):
        print(f"step {i} / {len(points)}")
        start_x = points[i][0]
        start_y = points[i][1]
        end_x = points[i + 1][0]
        end_y = points[i + 1][1]
        print(f"{start_x}, {start_y} -> {end_x}, {end_y}")
        util, segment = MixedWalker.generate_utilization_distribution(start_x=points[i][0],
                                                                      start_y=points[i][1],
                                                                      end_x=points[i + 1][0],
                                                                      end_y=points[i + 1][1],
                                                                      T=T,
                                                                      kernel_context=kernels_context)
        n = W * H

        acc = np.zeros(n, dtype=np.float64)

        full_path = np.vstack((full_path, segment[:-1]))

        # normalization
        for t in range(T):
            # print(util[t][0].data[0][0].len)
            pts = util[t][0].data[0][0].data.points
            acc += points_to_np_float64(pts, n)

        acc /= T

        # pts = util[2][0].data[0][0].data.points
        # acc += points_to_np_float64(pts, n)

        acc = acc.reshape((H, W))

        add_with_offset(total_utilization, acc, 0, 0)

    # plot_terrain_and_traj(seg_terrain, meta, [(sx, sy), (ex, ey)], ud=acc)

    # walk = MixedWalker.generate_custom_walks(
    #     terrain=seg_terrain_map,
    #     steps=[(sx, sy), (ex, ey)],
    #     T=30,
    #     kernel_mapping=kernel_mapping,
    #     plot=True,
    #     plot_title="Custom Terrain Walk"
    # )

    # print(type(terrain_map))
    plot_walk_terrain(terrain_map, points, ud=total_utilization, show_home_range=True)
    plot_walk_terrain(terrain_map, points, ud=total_utilization, show_home_range=True, ud_p=0.99)
    plot_walk_terrain(terrain_map, points, ud=total_utilization, show_home_range=True, ud_p=0.8)
    plot_walk_terrain(terrain_map, points, ud=total_utilization, show_home_range=True, ud_p=0.7)
    plot_walk_terrain(terrain_map, points, ud=total_utilization, show_home_range=True, ud_p=0.6)


if __name__ == "__main__":
    main()
