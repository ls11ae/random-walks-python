from tests.brownian_test import *

from random_walk_package.core.BiasedWalker import BiasedWalker
from random_walk_package.core.CorrelatedWalker import *
from random_walk_package.bindings.mixed_walk import *
from random_walk_package.bindings.plotter import *
from random_walk_package import matrix_generator_gaussian_pdf
from random_walk_package.bindings.brownian_walk import *
import numpy as np
import matplotlib.pyplot as plt


def plot_control(usage_count, T, W, H):
    
    # Decide subplot grid size (square-ish)
    n_cols = int(np.ceil(np.sqrt(kT)))
    n_rows = int(np.ceil(T / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()  # Flatten in case of single row/column
    
    for t in range(T):
        utilization_array = np.zeros((H, W))
        for y in range(H):
            for x in range(W):
                utilization_array[y, x] = usage_count[t,y,x]
        
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


if __name__ == "__main__":
    
    T = 9
    size = 100
    
    S = 12
    KS = 2 * S + 1
    
    walker = CorrelatedWalker(S=S, kernel=None, D=32, W=size, H=size, T=T)
    
    target_area = np.zeros((size, size), dtype=bool)
    
    target_area[0:50, 20:33] = True
    
    start_x = 10
    start_y = 50
    
    visit = walker.generate(start_x, start_y, target_area=target_area, use_serialization=False)
    
    
    
    # utilization_distribution = walker.utilize(end_x=30, end_y=30)
    
    # target_area = np.zeros((size, size), dtype=bool)
    
    # utilization_distribution = walker.visit_probability(start_x=10, start_y=10, end_x=80, end_y=80, target_area=target_area)
    
    plot_visit_matrix(visit, T, size, size, start_x, start_y, target_area)
    


    