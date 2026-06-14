# debugging: gdb --args python -m tests.test
import os.path
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from random_walk_package import StateDependentWalker, FixedStepsPolicy, TimeStepPolicy, tensor4D_free, \
    set_forbidden_landmark
from random_walk_package.bindings.data_structures.kernel_terrain_mapping import marine_kernels_baseline_crw, \
    update_kernels_mapping, set_landmark_mapping
from random_walk_package.bindings.plotter import plot_walk_from_json
from random_walk_package.bindings.mixed_walk import mix_backtrace, mix_backtrace2, mix_walk, mix_walk2
from random_walk_package.core.MixedWalker import *
from random_walk_package.data_sources.walk_visualization import save_trajectory_collection_timed
from tests.mixed_walk_test import test_marine_walker
import movingpandas as mpd

def weather_terrain_params(row):
    S = min(15, max(1, np.round(float(row["wind_speed_10m_max"]) / 2.0).astype(int)))
    D = min(16, max(1, np.round(int(row["wind_direction_10m_dominant"] // 45)).astype(int)))
    is_brownian = D == 1
    diffusity = float(row["cloud_cover_mean"]) / 100.0
    bias_x = int(row["precipitation_sum"] > 0.1)
    bias_y = int(row["terrain"] in (50, 60))
    return [is_brownian, S, D, diffusity, bias_x, bias_y]

def filter_bbox(traj_collection, bbox):
    gdf = traj_collection.to_point_gdf().copy()
    lon_min, lat_min, lon_max, lat_max = bbox

    keep = (
        (gdf.geometry.x >= lon_min) &
        (gdf.geometry.x <= lon_max) &
        (gdf.geometry.y >= lat_min) &
        (gdf.geometry.y <= lat_max)
    )

    gdf = gdf[keep]

    return mpd.TrajectoryCollection(
        gdf,
        traj_id_col="traj_id",
        t="time"
    )


def examine_traj_coll_pickle(file, out):
    with open(file, "rb") as ff:
        traj_col: mpd.TrajectoryCollection = pickle.load(ff)

    print("\n=== OUTPUT PICKLE ===")
    print(type(traj_col))
    print(traj_col.to_point_gdf().head(20))
    print(traj_col.to_point_gdf().columns)
    print(traj_col.to_point_gdf().dtypes)

    out_dir = "random_walk_package/resources/move_apps"

    walk_dir = os.path.join(out_dir, out)
    save_trajectory_collection_timed(traj_col, str(walk_dir))


def crop_terrain_txt(file, points, padding=10, max_padding=50):
    padding = min(padding, max_padding)

    with open(file, "r") as f:
        rows = [
            [int(value) for value in line.split()]
            for line in f
            if line.strip()
        ]

    height = len(rows)
    width = len(rows[0])
    min_x = max(0, min(x for x, _ in points) - padding)
    max_x = min(width - 1, max(x for x, _ in points) + padding)
    min_y = max(0, min(y for _, y in points) - padding)
    max_y = min(height - 1, max(y for _, y in points) + padding)

    crop_file = Path("random_walk_test_terrain_crop.txt")
    crop_file.parent.mkdir(parents=True, exist_ok=True)

    with open(crop_file, "w") as f:
        for row in rows[min_y:max_y + 1]:
            f.write(" ".join(str(value) for value in row[min_x:max_x + 1]))
            f.write("\n")

    cropped_points = [
        (x - min_x, y - min_y)
        for x, y in points
    ]

    print(
        f"Cropped terrain saved to {crop_file} with bounds "
        f"x={min_x}:{max_x}, y={min_y}:{max_y}"
    )
    print(f"Adjusted points for cropped grid: {cropped_points}")

    return str(crop_file), cropped_points


if __name__ == "__main__":
    walk = "/home/omar/CLionProjects/random-walks/mixed_walk_main.json"
    plot_walk_from_json(walk)
    exit()
    source_file = "/home/omar/PycharmProjects/random-walks-python/random_walk_package/resources/grid_upper_left_400.txt"
    points = [(100, 100), (100, 200), (200, 230), (204, 320), (70, 330)]

    """points = [
        (65, 110),
        (99, 220),
        (135, 110),
        (220, 90),
        (187, 213),
        (230, 350)
    ]"""
    file, points = crop_terrain_txt(source_file, points, padding=40, max_padding=70)
    terrain = parse_terrain(file, " ")

    W = width = terrain.contents.width
    H = height = terrain.contents.height
    mapping = create_mixed_kernel_parameters(Animal.TERRESTRIAL, 7)

    set_landmark_mapping(mapping, GRASSLAND, True, 5, 1, 0.9, 0.9, 0, 0)
    set_landmark_mapping(mapping, CROPLAND, False, 7, 12, len_diffusity=0.7, angle_diffusity=0.2)
    set_forbidden_landmark(mapping, BUILT_UP)

    kmap = get_tensor_map_terrain(terrain, mapping)
    T = 150
    walk_count = 3
    walk_set_count = 2
    max_backtrace_attempts = 25

    def generate_m_walk(start_x, start_y):
        return mix_walk(W, H, terrain, kmap, T, start_x, start_y, False, True, "", mapping)

    def backtrace_m_walk(dp, end_x, end_y):
        return mix_backtrace(dp, T, kmap, terrain, end_x, end_y, False, "", "", mapping)

    def generate_m_walk2(start_x, start_y):
        return mix_walk2(W, H, terrain, kmap, T, start_x, start_y)

    def backtrace_m_walk2(dp, end_x, end_y):
        return mix_backtrace2(dp, T, kmap, terrain, end_x, end_y)

    def generate_walks(version_name, walk_fn, backtrace_fn):
        walk_sets = [
            [[] for _ in range(walk_count)]
            for _ in range(walk_set_count)
        ]

        for i in range(len(points) - 1):
            print(f"Generating {version_name} DP from {points[i]} to {points[i + 1]}")
            start_x, start_y = points[i]
            end_x, end_y = points[i + 1]

            dp = walk_fn(start_x, start_y)
            try:
                for set_idx, full_walks in enumerate(walk_sets):
                    for walk_idx in range(walk_count):
                        print(
                            f"Backtracing {version_name} set {set_idx + 1}, "
                            f"walk {walk_idx + 1}, segment {i + 1}"
                        )
                        segment = None
                        for attempt in range(max_backtrace_attempts):
                            try:
                                segment = backtrace_fn(dp, end_x, end_y)
                                break
                            except ValueError:
                                if attempt == max_backtrace_attempts - 1:
                                    raise
                        if i > 0:
                            segment = segment[1:]
                        full_walks[walk_idx].extend(segment)
            finally:
                tensor4D_free(dp, T)

        return [
            [np.array(walk) for walk in full_walks]
            for full_walks in walk_sets
        ]

    m_walk_sets = generate_walks("m_walk", generate_m_walk, backtrace_m_walk)
    m_walk2_sets = generate_walks("m_walk2", generate_m_walk2, backtrace_m_walk2)

    m_walk_png = "m_walk.png"
    m_walk2_png = "m_walk2.png"
    comparison_png = "m_walk_comparison.png"
    comparison2_png = "m_walk_comparison_2.png"

    plot_combined_terrain(
        terrain,
        m_walk_sets[0],
        terrain_width=terrain.contents.width,
        terrain_height=terrain.contents.height,
        title="m_walk",
        steps=points,
        save_path=m_walk_png,
        show=False,
    )
    plt.close()

    plot_combined_terrain(
        terrain,
        m_walk2_sets[0],
        terrain_width=terrain.contents.width,
        terrain_height=terrain.contents.height,
        title="m_walk2",
        steps=points,
        save_path=m_walk2_png,
        show=False,
    )
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    plot_combined_terrain(
        terrain,
        m_walk_sets[0],
        terrain_width=terrain.contents.width,
        terrain_height=terrain.contents.height,
        title="m_walk",
        steps=points,
        ax=axes[0],
        show=False,
    )
    plot_combined_terrain(
        terrain,
        m_walk2_sets[0],
        terrain_width=terrain.contents.width,
        terrain_height=terrain.contents.height,
        title="m_walk2",
        steps=points,
        ax=axes[1],
        show=False,
    )
    fig.savefig(comparison_png, dpi=200, bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    plot_combined_terrain(
        terrain,
        m_walk_sets[1],
        terrain_width=terrain.contents.width,
        terrain_height=terrain.contents.height,
        title="m_walk - set 2",
        steps=points,
        ax=axes[0],
        show=False,
    )
    plot_combined_terrain(
        terrain,
        m_walk2_sets[1],
        terrain_width=terrain.contents.width,
        terrain_height=terrain.contents.height,
        title="m_walk2 - set 2",
        steps=points,
        ax=axes[1],
        show=False,
    )
    fig.savefig(comparison2_png, dpi=200, bbox_inches="tight")
    plt.show()
