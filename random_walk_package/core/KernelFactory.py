import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


class th:
    THRESHOLD = 300


# np.set_printoptions(formatter={'float_kind':"{:.2f}".format})


def detect_typical_interval(entries):
    diffs = []
    for i in range(1, len(entries)):
        delta = (entries[i][2] - entries[i - 1][2]).total_seconds() // 60
        diffs.append(delta)
    if len(diffs) == 0:
        return None
    # Häufigster Zeitabstand
    mode = Counter(diffs).most_common(1)[0][0]
    return mode


# Function to read and process CSV data
def process_bettongs(data_list: list[pd.DataFrame]):
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
    th.THRESHOLD = global_interval
    # exit(0)

    return animal_trajectories


def calculate_durations(animal_trajectories):
    durations = []
    for bettong_id, entries in animal_trajectories.items():
        # Calculate time differences between consecutive entries
        for i in range(1, len(entries)):
            time_diff = entries[i][2] - entries[i - 1][2]
            durations.append(time_diff.total_seconds() // 60)  # Convert to minutes
    return durations


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


def calculate_steps(animal_trajectories):
    steps = []
    count_total = 0
    count_discarded = 0
    for bettong_id, entries in animal_trajectories.items():
        # Calculate time differences between consecutive entries
        for i in range(1, len(entries)):
            count_total += 1
            time_diff = entries[i][2] - entries[i - 1][2]
            if abs(time_diff.total_seconds() // 60 - th.THRESHOLD) <= th.THRESHOLD * 0.1:
                a = (entries[i][0] - entries[i - 1][0], entries[i][1] - entries[i - 1][1])
                steps.append(rotate_vector((1, 0), (0, 0), a))
            else:
                print(time_diff.total_seconds() // 60)
                print(th.THRESHOLD)
                count_discarded += 1
            # durations.append(time_diff.total_seconds()/60)  # Convert to
    print(
        f"{min([x for x, _ in steps]), max([x for x, _ in steps]), min([y for _, y in steps]), max([y for _, y in steps])}")
    print(f"  total of {count_total} steps, {count_discarded / count_total}% discarded")

    return steps


def calculate_steps_cor_grouped(animal_trajectories):
    steps = [[], [], []]
    count_total = 0
    count_discarded = 0

    for bettong_id, entries in animal_trajectories.items():
        cur_len = 0
        # Calculate time differences between consecutive entries
        for i in range(2, len(entries)):
            count_total += 1
            time_diff_0 = (entries[i][2] - entries[i - 1][2]).total_seconds() / 60
            time_diff_1 = (entries[i - 1][2] - entries[i - 2][2]).total_seconds() / 60
            if abs(time_diff_0 - th.THRESHOLD) <= th.THRESHOLD * 0.1 and abs(
                    time_diff_1 - th.THRESHOLD) <= 0.1 * th.THRESHOLD:
                # steps.append((entries[i][0]-entries[i-1][0],entries[i][1]-entries[i-1][1]))
                a = (entries[i - 2][0], entries[i - 2][1])
                b = (entries[i - 1][0], entries[i - 1][1])
                c = (entries[i][0], entries[i][1])
                steps[entries[i - 1][3] - 1].append(rotate_vector(a, b, c))
                cur_len += 1
            else:
                count_discarded += 1

            # durations.append(time_diff.total_seconds()/60)  # Convert to
    # print(steps[0])
    # print(f"{min([x for x, _ in steps]), max([x for x, _ in steps]), min([y for _, y in steps]), max([y for _, y in steps])}")
    print(f"  total of {count_total} steps, {count_discarded / count_total}% discarded")
    print(f" State 1 {len(steps[0])}")
    print(f" State 2 {len(steps[1])}")
    print(f" State 3 {len(steps[2])}")
    return steps


def calculate_steps_cor(animal_trajectories):
    steps = []
    count_total = 0
    count_discarded = 0

    max_len = 0

    for amimal_id, entries in animal_trajectories.items():
        cur_len = 0
        # Calculate time differences between consecutive entries
        for i in range(2, len(entries)):
            count_total += 1
            time_diff_0 = (entries[i][2] - entries[i - 1][2]).total_seconds() / 60
            time_diff_1 = (entries[i - 1][2] - entries[i - 2][2]).total_seconds() / 60
            if abs(time_diff_0 - th.THRESHOLD) <= th.THRESHOLD * 0.1 and abs(
                    time_diff_1 - th.THRESHOLD) <= 0.1 * th.THRESHOLD:
                # steps.append((entries[i][0]-entries[i-1][0],entries[i][1]-entries[i-1][1]))
                a = (entries[i - 2][0], entries[i - 2][1])
                b = (entries[i - 1][0], entries[i - 1][1])
                c = (entries[i][0], entries[i][1])
                steps.append(rotate_vector(a, b, c))
                cur_len += 1
            else:
                count_discarded += 1
                max_len = max(max_len, cur_len)
        max_len = max(max_len, cur_len)

    print("max_len: ", max_len)
    # durations.append(time_diff.total_seconds()/60)  # Convert to
    print(
        f"{min([x for x, _ in steps]), max([x for x, _ in steps]), min([y for _, y in steps]), max([y for _, y in steps])}")
    print(f"  total of {count_total} steps, {count_discarded / count_total}% discarded")

    return steps


def plot_durations(durations):
    counts = Counter(durations)
    # Prepare data for plotting
    unique_numbers = list(counts.keys())
    frequencies = list(counts.values())

    # Create the histogram
    plt.figure(figsize=(8, 6))
    plt.bar(unique_numbers, frequencies, edgecolor='black')

    plt.xticks(unique_numbers)  # Set x-ticks to be each unique number

    # Show the plot
    plt.show()


rnge = 400
reso = rnge * 2 + 1


def fit_data(axs, steps):
    data = np.array(steps)
    # Define the number of Gaussian components
    n_components = 3

    # Fit a Gaussian Mixture Model to the data
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=22)
    gmm.fit(data)

    x = np.linspace(-rnge, rnge, reso)
    y = np.linspace(-rnge, rnge, reso)
    X, Y = np.meshgrid(x, y)
    grid = np.column_stack([X.ravel(), Y.ravel()])

    # Compute the density
    log_density = gmm.score_samples(grid)
    density = np.exp(log_density)
    Z = density.reshape(X.shape)

    # plt.figure(figsize=(8, 6))
    axs.imshow(Z, extent=(-rnge, rnge, -rnge, rnge), origin='lower', cmap='viridis', interpolation='nearest')
    return Z
    # axs.colorbar(label='Intensity')
    # axs.title('2D Heatmap Centered Around (0,0) with Squares')
    # axs.xlabel('X')
    # axs.ylabel('Y')
    # plt.show()


def generate_heatmap(axs, coords):
    # Define the grid

    # Convert to numpy array for easier manipulation
    coords = np.array(coords)
    # Define the Gaussian function# Define the grid boundaries
    x_edges = np.linspace(-rnge, rnge, reso)  # Adjust as needed
    y_edges = np.linspace(-rnge, rnge, reso)  # Adjust as needed

    c = 0
    for coord in coords:
        if coord[0] == 0 or coord[1] == 0:
            c += 1

    # Compute 2D histogram
    heatmap, xedges, yedges = np.histogram2d(coords[:, 0], coords[:, 1], bins=[x_edges, y_edges])

    # axs.scatter(coords[:, 0], coords[:, 1], alpha=0.01)

    # plt.figure(figsize=(8, 6))
    axs.imshow(heatmap.T, extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]), origin='lower', cmap='viridis')
    # axs.set_colorbar(label='Counts')
    # axs.set_xlabel('X')
    # axs.ylabel('Y')
    # plt.show()


def plot_trajectories(animal_trajectories):
    # print(bettongs)
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot each trajectory

    for i, trajectory in enumerate(animal_trajectories.values()):
        x, y, d = zip(*trajectory)  # Unzip the list of tuples into x and y coordinates
        ax.plot(x, y, label=f'Trajectory {i + 1}')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Trajectories')
    ax.legend()
    plt.grid(True)
    plt.show()


def sub(animal_trajectories):
    bettong = []

    max_len = -1
    for bettong_id, entries in animal_trajectories.items():
        cur_len = 0
        cur_bettong = []
        # Calculate time differences between consecutive entries
        for i in range(2, len(entries)):
            time_diff_0 = (entries[i][2] - entries[i - 1][2]).total_seconds() / 60
            time_diff_1 = (entries[i - 1][2] - entries[i - 2][2]).total_seconds() / 60
            if abs(time_diff_0 - th.THRESHOLD) <= th.THRESHOLD * 0.1 and abs(
                    time_diff_1 - th.THRESHOLD) <= 0.1 * th.THRESHOLD:
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


def to_json(density, num_directions=8, name="test"):
    print(len(density))
    data = {"name": name,
            "d": num_directions,
            "s": int((len(density[0]) - 1) / 2),
            "tm": [list(densit.flatten()) for densit in density]
            }
    with open("noncor.json", 'w') as json_file:
        json.dump(data, json_file, indent=4)


def show_fancy(animal_trajectories):
    steps = calculate_steps(animal_trajectories)
    steps_cor = calculate_steps_cor(animal_trajectories)

    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    generate_heatmap(axs[0, 0], steps)
    generate_heatmap(axs[0, 1], steps_cor)

    a = fit_data(axs[1, 0], steps)
    to_json(a)
    fit_data(axs[1, 1], steps_cor)

    plt.tight_layout()

    # Show plot
    plt.show()


def generate_angles(num_directions: int) -> list[float]:
    step = 360.0 / num_directions
    return [i * step for i in range(num_directions)]


def pure_cor(animal_trajectories, num_directions):
    angles = generate_angles(num_directions)

    steps_cor = calculate_steps_cor(animal_trajectories)

    plots = []
    fig, axs = plt.subplots(1, min(num_directions, 8), figsize=(12, 6))
    for i, theta in enumerate(angles):
        theta = -theta / 180 * np.pi
        e = (1, 0)
        # Rotation matrix
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        # Rotate vector e
        e_rotated = np.dot(rotation_matrix, e)
        print(theta, e_rotated, rotate_vector(e_rotated, (0, 0), (1, 0)))

        r_steps = [rotate_vector(e_rotated, (0, 0), step) for step in steps_cor]
        if i < min(num_directions, 8):
            a = fit_data(axs[i], r_steps)
            plots.append(a)
        # to_json(a)

    to_json(plots, num_directions)
    plt.tight_layout()
    plt.show()


updating = False


def pure_cor_grouped(animal_trajectories, num_directions=1):
    a, b, c = calculate_steps_cor_grouped(animal_trajectories)
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))

    if len(a) != 0:
        generate_heatmap(axs[0, 0], a)
        fit_data(axs[1, 0], a)
    if len(b) != 0:
        generate_heatmap(axs[0, 1], b)
        fit_data(axs[1, 1], b)
    if len(c) != 0:
        generate_heatmap(axs[0, 2], c)
        fit_data(axs[1, 2], c)

    linked_pairs = [(axs[0, 0], axs[1, 0]), (axs[0, 1], axs[1, 1]), (axs[0, 2], axs[1, 2])]

    # Function to synchronize zoom/pan between pairs
    def on_xlim_changed(event_ax):
        global updating
        if updating:
            return  # Prevent recursive calls

        # Temporarily disable updating to avoid recursion
        updating = True
        # Check each pair of linked plots
        for ax1, ax2 in linked_pairs:
            if event_ax == ax1:
                # Sync ax2 with ax1
                ax2.set_xlim(ax1.get_xlim())
                ax2.set_ylim(ax1.get_ylim())
            elif event_ax == ax2:
                # Sync ax1 with ax2
                ax1.set_xlim(ax2.get_xlim())
                ax1.set_ylim(ax2.get_ylim())

        # Redraw the figure to apply changes
        fig.canvas.draw_idle()
        updating = False

    # Connect zoom/pan event listeners for all axes in the linked pairs
    for ax1, ax2 in linked_pairs:
        ax1.callbacks.connect('xlim_changed', on_xlim_changed)
        ax1.callbacks.connect('ylim_changed', on_xlim_changed)
        ax2.callbacks.connect('xlim_changed', on_xlim_changed)
        ax2.callbacks.connect('ylim_changed', on_xlim_changed)

    """plt.tight_layout()
    plt.show()"""

# pure_cor(bettongs, 32)


# plot_trajectories(sub(bettongs))

# plot_trajectories(bettongs)

# Rotate vector b-c to make a-b point left

# durations = calculate_durations(sorted_bettongs)
# print(durations)
# plot_durations(durations)

# To print the sorted bettongs for verification
# for bettong_id, entries in sorted_bettongs.items():
#     print(f"Bettong {bettong_id}:")
#     for entry in entries:
#         print(entry)
