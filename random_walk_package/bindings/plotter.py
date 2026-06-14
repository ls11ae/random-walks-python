import json

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from random_walk_package.bindings import terrain_at
from environmentcma import padded_utm_bbox
from segmentationcma import bbox_of_segment


def plot_walk(walk_points, terrain_width, terrain_height, title="Walk"):
    if walk_points is not None:
        plt.ylim(-1, terrain_height)
        plt.xlim(-1, terrain_width)
        plt.plot(walk_points[:, 0], walk_points[:, 1], 'r-')  # Remove dots
        plt.scatter([walk_points[0, 0]], [walk_points[0, 1]], color='green', label='Start')  # First point
        plt.scatter([walk_points[-1, 0]], [walk_points[-1, 1]], color='blue', label='End')  # Last point
        plt.legend()
        plt.title(title)
        plt.show()
    else:
        print("No path generated.")


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def plot_combined_terrain(
    terrain,
    walk_points,
    terrain_width=None,
    terrain_height=None,
    steps=None,
    title=None,
    ax=None,
    show=True,
    save_path=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    if terrain_width is None or terrain_height is None:
        terrain_width = terrain.contents.width
        terrain_height = terrain.contents.height

    try:
        terrain_array = np.array([
            [terrain_at(terrain, x, y) for x in range(terrain_width)]
            for y in range(terrain_height)
        ])
    except Exception as e:
        print(f"Error generating terrain_array with terrain_at_func: {e}")
        print("Ensure terrain_at_func correctly accesses terrain with x, y coordinates and returns landmarkType values.")
        return

    landmark_colors_map = {
        10: (0.0, 0.4, 0.0, 0.99),
        #20: (0.5, 0.5, 0.0, 0.8),
        30: (0.0, 0.8, 0.0, 0.99),
        40: (0.6, 0.8, 0.2, 0.99),
        50: (0.5, 0.5, 0.5, 0.99),
        60: (0.82, 0.71, 0.55, 0.99),
        #70: (0.9, 0.95, 1.0, 0.8),
        80: (0.0, 0.0, 1.0, 0.7),
        90: (0.45, 0.62, 0.52, 1.0),
        #95: (0.0, 0.5, 0.5, 0.8),
        #100: (0.33, 0.42, 0.18, 0.8),
    }

    landmark_labels = {
        10: "Tree cover",
        #20: "Shrubland",
        30: "Grassland",
        40: "Cropland",
        50: "Built-up",
        60: "Sparse vegetation",
        #70: "Snow and ice",
        80: "Water",
        90: "Herbaceous wetland",
        #95: "Mangroves",
        #100: "Moss and lichen",
    }

    sorted_landmark_values = sorted(landmark_colors_map.keys())
    cmap_colors_list = [landmark_colors_map[val] for val in sorted_landmark_values]

    if not sorted_landmark_values:
        cmap = mcolors.ListedColormap([(0, 0, 0, 0)])
        norm = mcolors.BoundaryNorm([0, 1], cmap.N)
    elif len(sorted_landmark_values) == 1:
        bounds = [
            sorted_landmark_values[0] - 0.5,
            sorted_landmark_values[0] + 0.5,
        ]
        cmap = mcolors.ListedColormap([cmap_colors_list[0]])
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
    else:
        bounds = [sorted_landmark_values[0] - 0.5]

        for i in range(len(sorted_landmark_values) - 1):
            bounds.append(
                (sorted_landmark_values[i] + sorted_landmark_values[i + 1]) / 2.0
            )

        bounds.append(sorted_landmark_values[-1] + 0.5)

        cmap = mcolors.ListedColormap(cmap_colors_list)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(
        terrain_array,
        cmap=cmap,
        norm=norm,
        origin="lower",
        extent=(-0.5, terrain_width - 0.5, -0.5, terrain_height - 0.5),
        interpolation="nearest",
    )

    walk_colors = ["black", "red", "blue"]
    walks = []
    if walk_points is not None and len(walk_points) > 0:
        try:
            walk_array = np.asarray(walk_points)
        except ValueError:
            walks = [np.asarray(walk) for walk in walk_points]
        else:
            if walk_array.ndim == 2:
                walks = [walk_array]
            else:
                walks = [np.asarray(walk) for walk in walk_points]

    if len(walks) > 3:
        raise ValueError("plot_combined_terrain supports at most 3 walks.")

    for walk_idx, walk in enumerate(walks):
        if len(walk) == 0:
            continue

        ax.plot(
            walk[:, 0],
            walk[:, 1],
            color=walk_colors[walk_idx],
            linewidth=2,
            label=f"Walk {walk_idx + 1}",
            zorder=2,
        )

    non_empty_walks = [walk for walk in walks if len(walk) > 0]
    if non_empty_walks:
        first_walk = non_empty_walks[0]
        ax.scatter(
            first_walk[0, 0],
            first_walk[0, 1],
            color="lime",
            edgecolor="black",
            s=50,
            label="Start",
            zorder=3,
        )

        ax.scatter(
            first_walk[-1, 0],
            first_walk[-1, 1],
            color="white",
            edgecolor="black",
            s=50,
            label="End",
            zorder=3,
        )
    step_color = "#D08A00"
    if steps is not None:
        for i, (x, y) in enumerate(steps):
            if i == 0 or i == len(steps) - 1:
                continue

            ax.scatter(
                x,
                y,
                s=100,
                marker="s",
                color=step_color,
                edgecolor="black",
                zorder=2,
            )

            ax.text(
                x,
                y,
                str(i),
                color="black",
                ha="center",
                va="center",
                fontsize=9,
                zorder=3,
            )

    classes = sorted(landmark_colors_map.keys())

    legend_handles = [
        Patch(
            facecolor=landmark_colors_map[c],
            edgecolor="black",
            label=landmark_labels[c],
        )
        for c in classes
    ]

    path_handles = [
        Line2D(
            [0],
            [0],
            color=walk_colors[i],
            linewidth=2,
            label=f"Walk {i + 1}",
        )
        for i in range(len(walks))
    ]

    start_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="lime",
        markeredgecolor="black",
        markersize=8,
        label="Start",
    )

    end_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=8,
        label="End",
    )

    ax.legend(
        handles=legend_handles + path_handles + [start_handle, end_handle],
        loc="upper left",
        fontsize=9,
    )

    ax.set_xlim(-1, terrain_width)
    ax.set_ylim(terrain_height, -1)

    ax.set_xticks(np.arange(0, terrain_width + 1, 50))
    ax.set_yticks(np.arange(0, terrain_height + 1, 50))

    ax.tick_params(
        axis="both",
        which="both",
        labelsize=7,
        colors="gray",
        length=2,
        width=0.5,
    )

    for spine in ax.spines.values():
        spine.set_color("lightgray")
        spine.set_linewidth(0.5)

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.5,
        alpha=0.4,
    )

    ax.set_aspect("equal", adjustable="box")

    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def plot_walk_from_json(json_path, title=None):
    # Lade JSON-Daten
    with open(json_path, 'r') as f:
        data = json.load(f)

    terrain_array = np.array(data["Terrain"])
    terrain_height, terrain_width = terrain_array.shape

    # Extrahiere Walk- und Step-Punkte
    walk_points = np.array([[pt["x"], pt["y"]] for pt in data.get("Walk", [])])
    steps = [(step["x"], step["y"]) for step in data.get("Steps", [])]

    # Farbzuordnung wie gehabt
    landmark_colors_map = {
        10: (0.0, 0.4, 0.0, 0.5),
        20: (0.5, 0.5, 0.0, 0.5),
        30: (0.0, 0.8, 0.0, 0.5),
        40: (0.6, 0.8, 0.2, 0.5),
        50: (0.5, 0.5, 0.5, 0.5),
        60: (0.82, 0.71, 0.55, 0.5),
        70: (0.9, 0.95, 1.0, 0.5),
        80: (0.0, 0.0, 1.0, 0.5),
        90: (0.25, 0.88, 0.82, 0.5),
        95: (0.0, 0.5, 0.5, 0.5),
        100: (0.33, 0.42, 0.18, 0.5)
    }

    sorted_landmark_values = sorted(landmark_colors_map.keys())
    cmap_colors_list = [landmark_colors_map[val] for val in sorted_landmark_values]

    PADDING = 0.5

    # Erzeuge Boundaries und Farbzuordnung
    plot_bounds = []
    if not sorted_landmark_values:
        cmap = mcolors.ListedColormap([(0, 0, 0, 0)])
        norm = mcolors.BoundaryNorm([0, 1], cmap.N)
    elif len(sorted_landmark_values) == 1:
        plot_bounds = [sorted_landmark_values[0] - PADDING, sorted_landmark_values[0] + PADDING]
        cmap = mcolors.ListedColormap([cmap_colors_list[0]])
        norm = mcolors.BoundaryNorm(plot_bounds, cmap.N)
    else:
        plot_bounds.append(sorted_landmark_values[0] - PADDING)
        for i in range(len(sorted_landmark_values) - 1):
            plot_bounds.append((sorted_landmark_values[i] + sorted_landmark_values[i + 1]) / 2.0)
        plot_bounds.append(sorted_landmark_values[-1] + PADDING)
        cmap = mcolors.ListedColormap(cmap_colors_list)
        norm = mcolors.BoundaryNorm(plot_bounds, cmap.N)

    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(terrain_array, cmap=cmap, norm=norm, origin='lower',
               extent=(-PADDING, terrain_width - PADDING, -PADDING, terrain_height - PADDING),
               interpolation='nearest')

    if walk_points.size > 0:
        plt.plot(walk_points[:, 0], walk_points[:, 1], 'r-', label='Path', zorder=2)
        plt.scatter(walk_points[0, 0], walk_points[0, 1], color='black', s=50, label='Start', zorder=3)
        plt.scatter(walk_points[-1, 0], walk_points[-1, 1], color='blue', s=50, label='End', zorder=3)

    for i, (x, y) in enumerate(steps):
        plt.scatter(x, y, s=100, marker='s', color='orange', edgecolor='black', zorder=2)
        plt.text(x, y, str(i), color='black', ha='center', va='center', fontsize=9, zorder=3)

    plt.title(title or "Terrain Map with Path")
    plt.xlim(-1, terrain_width)
    plt.ylim(terrain_height, -1)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def plot_walk_terrain(terrain, walk_points, terrain_width, terrain_height):
    plt.figure(figsize=(10, 10))

    # Convert terrain to NumPy array
    terrain_array = np.array([[terrain_at(terrain, x, y) for x in range(terrain_width)]
                              for y in range(terrain_height)])

    # Define custom colormap
    cmap = mcolors.ListedColormap([
        (0.0, 0.0, 1.0, 0.5),  # Water (blue, 50% opacity)
        (0.956, 0.643, 0.376, 0.5),  # Desert/Rock (sandybrown, 50% opacity)
        (0.0, 0.5, 0.0, 0.5)  # Forest (green, 50% opacity)
    ])

    bounds = [0, 1, 2, 3]  # Define boundaries for each terrain type
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Display terrain as an image with the custom colormap
    plt.imshow(terrain_array, cmap=cmap, norm=norm, origin='lower',
               extent=(-0.5, terrain_width - 0.5, -0.5, terrain_height - 0.5))

    if walk_points is not None:
        plt.plot(walk_points[:, 0], walk_points[:, 1], 'r-')  # Red line for the path
        plt.scatter([walk_points[0, 0]], [walk_points[0, 1]], color='black', label='Start')  # First point
        plt.scatter([walk_points[-1, 0]], [walk_points[-1, 1]], color='blue', label='End')  # Last point

    plt.title("Walk")
    plt.xlim(-1, terrain_width)
    plt.ylim(terrain_height, -1)
    plt.legend()
    plt.show()


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm

def plot_animal_segments_overview(
    steps,
    segments,
    animal_id=None,
    show_segment_lines=True,
    show_step_points=False,
    use_padded_bbox=False,
    max_cell_size=None,
    resolution=None,
):

    xs = steps["geo_x"].to_numpy()
    ys = steps["geo_y"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.plot(xs, ys, color="black", linewidth=1.8, alpha=0.8, label="trajectory")

    if show_step_points:
        ax.scatter(xs, ys, s=10, color="black", alpha=0.5)

    # Start/Ende der gesamten Trajektorie
    ax.scatter(xs[0], ys[0], s=80, color="green", edgecolor="black", zorder=5, label="start")
    ax.scatter(xs[-1], ys[-1], s=80, color="red", edgecolor="black", zorder=5, label="end")

    cmap = cm.get_cmap("tab20", max(1, len(segments)))

    for i, segment in enumerate(segments):
        seg_start, seg_end = segment
        color = cmap(i)

        if show_segment_lines:
            seg_x = xs[seg_start:seg_end + 1]
            seg_y = ys[seg_start:seg_end + 1]
            ax.plot(seg_x, seg_y, color=color, linewidth=3, alpha=0.95)

            ax.scatter(seg_x[0], seg_y[0], s=45, color=color, edgecolor="black", zorder=6)
            ax.scatter(seg_x[-1], seg_y[-1], s=45, color=color, marker="s", edgecolor="black", zorder=6)

        # Box bestimmen
        if not use_padded_bbox:
            min_lon, min_lat, max_lon, max_lat = bbox_of_segment(steps, segment)
        else:
            if max_cell_size is None:
                raise ValueError("use_padded_bbox=True benötigt max_cell_size")
            min_lon, min_lat, max_lon, max_lat = bbox_of_segment(steps, segment)
            utm_bbox, zone, hemi, epsg_code, fwd, inv = padded_utm_bbox(
                min_lon, min_lat, max_lon, max_lat,
                padding=0.2,
                max_cell_size=max_cell_size
            )
            min_utm_x, min_utm_y, max_utm_x, max_utm_y = utm_bbox
            min_lon, min_lat = inv.transform(min_utm_x, min_utm_y)
            max_lon, max_lat = inv.transform(max_utm_x, max_utm_y)

        rect = Rectangle(
            (min_lon, min_lat),
            max_lon - min_lon,
            max_lat - min_lat,
            fill=False,
            edgecolor=color,
            linewidth=2,
            linestyle="--",
            alpha=0.9
        )
        ax.add_patch(rect)

        ax.text(
            min_lon,
            max_lat,
            f"S{segment}",
            color=color,
            fontsize=10,
            weight="bold",
            verticalalignment="bottom"
        )

    title = f"Animal {animal_id} - {segments}" if animal_id is not None \
            else "Trajectory with segment boxes"
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()

def plot_walk_multistep(steps, walk_points, terrain_width, terrain_height):
    if walk_points is not None:
        # Create the plot
        plt.figure(figsize=(10, 10))
        plt.ylim(-1, terrain_height)
        plt.xlim(-1, terrain_width)

        # Plot the path without dots
        plt.plot(walk_points[:, 0], walk_points[:, 1], 'b-', label='Path')

        # Plot the steps as squares with step indices
        for i, (x, y) in enumerate(steps):
            plt.scatter(x, y, s=200, marker='s', color='red', edgecolor='black')  # Square marker
            plt.text(x, y, str(i), color='white', ha='center', va='center', fontsize=12)  # Step index

        # Add labels and legend
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Correlated Walks with Steps')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No path generated.")
