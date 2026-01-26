import json

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from random_walk_package.bindings import terrain_at


def plot_walk(walk_points, terrain_width, terrain_height, title="Walk"):
    if walk_points is not None:
        plt.ylim(-1, terrain_height)
        plt.xlim(-1, terrain_width)
        plt.plot(walk_points[:, 0], walk_points[:, 1], 'r-o', markersize=4, linewidth=1, label='Path')  # Line with points
        plt.scatter([walk_points[0, 0]], [walk_points[0, 1]], color='green', label='Start')  # First point
        plt.scatter([walk_points[-1, 0]], [walk_points[-1, 1]], color='blue', label='End')  # Last point
        plt.legend()
        plt.title(title)
        plt.show()
    else:
        print("No path generated.")


def plot_combined_terrain(terrain, walk_points, terrain_width=None, terrain_height=None, steps=None, title=None):
    plt.figure(figsize=(10, 10))

    # Convert terrain data (using the provided terrain_at_func) to a NumPy array
    # The terrain_at_func is expected to return integer values from the landmarkType enum
    if terrain_width is None or terrain_height is None:
        terrain_width = terrain.contents.width
        terrain_height = terrain.contents.height
    try:
        terrain_array = np.array([[terrain_at(terrain, x, y) for x in range(terrain_width)]
                                  for y in range(terrain_height)])
    except Exception as e:
        print(f"Error generating terrain_array with terrain_at_func: {e}")
        print(
            "Ensure terrain_at_func correctly accesses terrain with x, y coordinates and returns landmarkType values.")
        return

    # Define landmark types and their corresponding colors (RGBA format)
    # Opacity (alpha) is set to 0.5 for all, adjust as needed.
    landmark_colors_map = {
        10: (0.0, 0.4, 0.0, 0.5),  # TREE_COVER: Dark Green
        20: (0.5, 0.5, 0.0, 0.5),  # SHRUBLAND: Olive
        30: (0.0, 0.8, 0.0, 0.5),  # GRASSLAND: Light Green
        40: (0.6, 0.8, 0.2, 0.5),  # CROPLAND: Yellow-Green
        50: (0.5, 0.5, 0.5, 0.5),  # BUILT_UP: Grey
        60: (0.82, 0.71, 0.55, 0.5),  # SPARSE_VEGETATION: Tan/Light Brown
        70: (0.9, 0.95, 1.0, 0.5),  # SNOW_AND_ICE: Very Light Blue / White
        80: (0.0, 0.0, 1.0, 0.5),  # WATER: Blue
        90: (0.25, 0.88, 0.82, 0.5),  # HERBACEOUS_WETLAND: Aquamarine/Turquoise
        95: (0.0, 0.5, 0.5, 0.5),  # MANGROVES: Teal
        100: (0.33, 0.42, 0.18, 0.5)  # MOSS_AND_LICHEN: Dark Olive Green / Brownish Green
    }

    # Get sorted list of landmark values and corresponding colors
    # These are the exact values expected in the terrain_array
    sorted_landmark_values = sorted(landmark_colors_map.keys())
    cmap_colors_list = [landmark_colors_map[val] for val in sorted_landmark_values]

    # Create bounds for BoundaryNorm
    # The bounds ensure that each specific landmark value gets its designated color
    plot_bounds = []
    if not sorted_landmark_values:  # Handle empty landmark list
        cmap = mcolors.ListedColormap([(0, 0, 0, 0)])  # transparent
        norm = mcolors.BoundaryNorm([0, 1], cmap.N)
    elif len(sorted_landmark_values) == 1:  # Handle single landmark type
        plot_bounds = [sorted_landmark_values[0] - 0.5, sorted_landmark_values[0] + 0.5]
        cmap = mcolors.ListedColormap([cmap_colors_list[0]])
        norm = mcolors.BoundaryNorm(plot_bounds, cmap.N)
    else:
        plot_bounds.append(sorted_landmark_values[0] - 0.5)  # Lower bound for the first color
        for i in range(len(sorted_landmark_values) - 1):
            # Midpoints between consecutive landmark values
            plot_bounds.append((sorted_landmark_values[i] + sorted_landmark_values[i + 1]) / 2.0)
        plot_bounds.append(sorted_landmark_values[-1] + 0.5)  # Upper bound for the last color

        cmap = mcolors.ListedColormap(cmap_colors_list)
        norm = mcolors.BoundaryNorm(plot_bounds, cmap.N)

    # Display terrain with coordinate system origin at lower-left
    plt.imshow(terrain_array, cmap=cmap, norm=norm, origin='lower',
               extent=(-0.5, terrain_width - 0.5, -0.5, terrain_height - 0.5),
               interpolation='nearest')  # 'nearest' is good for discrete categories

    # Plot walk path if provided
    if walk_points is not None and len(walk_points) > 0:
        plt.plot(walk_points[:, 0], walk_points[:, 1], 'r-', label='Path', zorder=2)  # Red path
        plt.scatter(walk_points[0, 0], walk_points[0, 1],
                    color='black', s=50, label='Start', zorder=3)  # s for size
        plt.scatter(walk_points[-1, 0], walk_points[-1, 1],
                    color='blue', s=50, label='End', zorder=3)  # s for size

    # Plot steps if provided
    if steps is not None:
        for i, (x, y) in enumerate(steps):
            plt.scatter(x, y, s=100, marker='s', color='orange',  # Changed color for visibility
                        edgecolor='black', zorder=2)
            plt.text(x, y, str(i), color='black', ha='center',  # Changed text color for visibility
                     va='center', fontsize=9, zorder=3)

    # Configure axes and labels
    if title:
        plt.title(title)
    else:
        plt.title("Terrain Map with Path")

    plt.xlim(-1, terrain_width)
    plt.ylim(terrain_height, -1)  # Y-axis inverted to match common array indexing (optional)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)  # Optional grid
    plt.gca().set_aspect('equal', adjustable='box')  # Keep aspect ratio

    plt.show()


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

# def plot_dp_utilisation_matrix(utilization_distribution, T, W, H):
#     utilization_array = np.zeros((H, W))
    
#     D = utilization_distribution[0][0].len
    
    
#     for y in range(H):
#         for x in range(W):
#             utilization_array[y, x] = sum(utilization_distribution[T][0].data[d][0].data.points[y*H + x] for d in range(D))

    plt.figure(figsize=(10, 8))
    plt.imshow(utilization_array, cmap='viridis', origin='lower')
    plt.colorbar(label='Utilization Value')
    plt.title(f'Utilization Distribution at T={T}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()
    
from matplotlib.colors import LogNorm


def plot_single_utilisation_matrix(utilization_distribution, T, W, H, level=None):


    
    # im = ax.imshow(utilization_array, cmap='viridis', origin='lower')
    plt.figure(figsize=(10, 8))
    
    # levels = [0.01, 0.5, 0.75]
    vmin = utilization_distribution[utilization_distribution>0].min()
    vmax = utilization_distribution.max()
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 100)
    CS = plt.contour(utilization_distribution, levels=levels, colors='white', linewidths=1)
    
    plt.imshow(utilization_distribution, cmap='viridis', origin='lower')
    plt.colorbar(label='Utilization Value', norm=LogNorm())
    plt.xlabel('X')
    plt.ylabel('Y')
        
    plt.tight_layout()
    plt.show()
    
    

    
def plot_dp_utilisation_matrix(utilization_distribution, T, W, H, squish=False, level=None):
    D = utilization_distribution[0][0].len
    
    n_cols = 1
    n_rows = 1
    
    if not squish:
        # Decide subplot grid size (square-ish)
        n_cols = int(np.ceil(np.sqrt(T)))
        n_rows = int(np.ceil(T / n_cols))
    
    
    if squish:
        utilization_array = np.zeros((H, W))
        for y in range(H):
            for x in range(W):
                utilization_array[y, x] = sum(
                    utilization_distribution[t][0].data[d][0].data.points[y*W + x]
                    for d in range(D) for t in range(T)
                ) / T
        
        # im = ax.imshow(utilization_array, cmap='viridis', origin='lower')
        plt.figure(figsize=(10, 8))
        
        # levels = [0.01, 0.5, 0.75]
        vmin = utilization_array[utilization_array>0].min()
        vmax = utilization_array.max()
        levels = np.logspace(np.log10(vmin), np.log10(vmax), 10)
        CS = plt.contour(utilization_array, levels=levels, colors='white', linewidths=1)
        
        plt.imshow(utilization_array, cmap='viridis', origin='lower', norm=LogNorm())
        plt.colorbar(label='Utilization Value')
        plt.xlabel('X')
        plt.ylabel('Y')
    
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten()  # Flatten in case of single row/column
        
        for t in range(T):
            utilization_array = np.zeros((H, W))
            for y in range(H):
                for x in range(W):
                    utilization_array[y, x] = sum(
                        utilization_distribution[t][0].data[d][0].data.points[y*W + x]
                        for d in range(D)
                    )
            
            ax = axes[t]
            im = ax.imshow(utilization_array, cmap='viridis', origin='lower')
            ax.set_title(f'T={t}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Leave extra axes empty (no data, no axis labels)
        for i in range(T, len(axes)):
            axes[i].axis('off')  # Completely remove axis, grid, and ticks
            axes[i].set_facecolor('white')  # Optional: ensure background is white
        
    plt.tight_layout()
    plt.show()
    # plt.show(block=False)
    # plt.pause(0.001)
    

def plot_visit_matrix(visit, T, W, H, start_x, start_y, target_area):
    D = visit[0][0].len
    
    n_cols = 1
    n_rows = 1
    
    
    n_cols = int(np.ceil(np.sqrt(T)))
    n_rows = int(np.ceil(T / n_cols))
    
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()  # Flatten in case of single row/column
    # x = np.arange(W)# - 0.5
    # y = np.arange(H)# - 0.5
    # X, Y = np.meshgrid(x, y)
    # plt.contour(
    #     X, Y,
    #     target_area.astype(int),      # convert bool → int
    #     levels=[0.5],                 # draw boundary between 0/1
    #     colors='red',                 # contour color
    #     linewidths=2,
    #     origin='lower'
    # )
    for t in range(T):
    
        utilization_array = np.zeros((H, W))
        for y in range(H):
            for x in range(W):
                utilization_array[y, x] = sum(
                    visit[t][0].data[d][0].data.points[y*W + x]
                    for d in range(D)
                ) / D
                
        ax = axes[t]
        
        
        x = np.arange(W)# - 0.5
        y = np.arange(H)# - 0.5
        X, Y = np.meshgrid(x, y)
        mask = np.zeros((H, W), dtype=int)
        mask[start_y, start_x] = 1 
        
        ax.contour(
            X, Y,
            mask.astype(int),      # convert bool → int
            levels=[0.5],                 # draw boundary between 0/1
            colors='Blue',                 # contour color
            linewidths=4,
            origin='lower'
        )
        ax.contour(
            X, Y,
            target_area.astype(int),      # convert bool → int
            levels=[0.5],                 # draw boundary between 0/1
            colors='red',                 # contour color
            linewidths=2,
            origin='lower'
        )
        
        
        im = ax.imshow(utilization_array, cmap='viridis', origin='lower')
        ax.set_title(f'T={t}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        
    # Leave extra axes empty (no data, no axis labels)
    for i in range(T, len(axes)):
        axes[i].axis('off')  # Completely remove axis, grid, and ticks
        axes[i].set_facecolor('white')  # Optional: ensure background is white
    
    # im = ax.imshow(utilization_array, cmap='viridis', origin='lower')
    # print(utilization_array)
    
    plt.figure(figsize=(10, 8))
    
    
    x = np.arange(W)# - 0.5
    y = np.arange(H)# - 0.5
    X, Y = np.meshgrid(x, y)
    mask = np.zeros((H, W), dtype=int)
    mask[start_y, start_x] = 1 
    plt.contour(
        X, Y,
        target_area.astype(int),      # convert bool → int
        levels=[0.5],                 # draw boundary between 0/1
        colors='red',                 # contour color
        linewidths=2,
        origin='lower'
    )
    plt.contour(
        X, Y,
        mask.astype(int),      # convert bool → int
        levels=[0.5],                 # draw boundary between 0/1
        colors='orange',                 # contour color
        linewidths=4,
        origin='lower'
    )
    
    # levels = [0.01, 0.5, 0.75]
    vmin = utilization_array[utilization_array>0].min()
    vmax = utilization_array.max()
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 10)
    # CS = plt.contour(utilization_array, levels=levels, colors='white', linewidths=1)
    
    plt.imshow(utilization_array, cmap='viridis', origin='lower')
    plt.colorbar(label='Visit Probability')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.tight_layout()
    plt.show()