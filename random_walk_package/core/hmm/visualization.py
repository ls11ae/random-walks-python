import matplotlib.pyplot as plt
import numpy as np

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
    axs.imshow(heatmap.T, extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]), origin='lower', cmap='viridis')