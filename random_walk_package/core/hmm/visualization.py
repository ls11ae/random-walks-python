import matplotlib.pyplot as plt
import numpy as np

from random_walk_package.core.hmm.kernels import calculate_steps, calculate_steps_cor
from random_walk_package.core.hmm.models import fit_data
from random_walk_package.core.hmm.utils import to_json


def generate_heatmap(axs, coords, rnge, reso):
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


def show_fancy(animal_trajectories, dt_threshold, dt_tolerance, rnge, reso):
    steps = calculate_steps(dt_threshold, dt_tolerance, animal_trajectories)
    steps_cor = calculate_steps_cor(dt_threshold, dt_tolerance, animal_trajectories)

    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    generate_heatmap(axs[0, 0], steps, rnge, reso)
    generate_heatmap(axs[0, 1], steps_cor, rnge, reso)

    a = fit_data(axs[1, 0], steps, rnge, reso)
    to_json(a)
    fit_data(axs[1, 1], steps_cor, rnge, reso)

    plt.tight_layout()

    # Show plot
