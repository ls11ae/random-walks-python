
# from tests.cuda_test import *
from tests.brownian_test import *

from random_walk_package.core.BiasedWalker import BiasedWalker
from random_walk_package.core.CorrelatedWalker import *
from random_walk_package.bindings.mixed_walk import *
from random_walk_package.bindings.plotter import *
from random_walk_package import matrix_generator_gaussian_pdf
from random_walk_package.bindings.brownian_walk import *
import numpy as np
import matplotlib.pyplot as plt



# def test_brownian_walk_gpu():
#     T = 400
#     W = 2 * T + 1
#     H = 2 * T + 1
#     S = 15

#     start_x = 8 * T // 5
#     start_y = 8 * T // 5
#     end_x = T // 4
#     end_y = T // 3

#     kernel = matrix_generator_gaussian_pdf(2 * S + 1, 2 * S + 1, sigma=10.0)

#     result_ptr = brownian_walk_gpu(kernel, S=S, T=T, W=W, H=H, start_x=start_x, start_y=start_y, end_x=end_x,
#                                    end_y=end_y)

#     plot_walk(result_ptr, W, H)

def plot_control(usage_count, T, W, H):
    
    # Decide subplot grid size (square-ish)
    n_cols = int(np.ceil(np.sqrt(T)))
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
    
    T = 20
    size = 100
    
    S = 20
    KS = 2 * S + 1
    matrix = matrix_generator_gaussian_pdf(KS, KS, sigma=10.0)
    

    
    kernels = correlated_kernels_from_matrix(matrix, KS, KS, 1)
    
    walker = CorrelatedWalker(S=S, kernel=None, D=16, W=size, H=size, T=T)
    
    walker.generate(start_x=10, start_y=40, use_serialization=False)
    
    # # dp = walker.dp_matrix
    
    utilization_distribution = walker.utilize(end_x=90, end_y=60)
    
    # plot_dp_utilisation_matrix(walker.dp_matrix, T, size, size)
    plot_dp_utilisation_matrix(utilization_distribution, T, size, size, squish=True)
    
    # # usage_counts = np.zeros((T, size, size), dtype=np.float64)

    # # N = 1000
    
    # # for i in range(N):
    # #     if i % (N/1000) == 0:
    # #         print(f"Simulation progress: {i/N*100:.1f}%",end='\r')
    # #     walk = walker.backtrace(end_x=40, end_y=40, plot=False)
    # #     for t, (x, y) in enumerate(walk):
    # #         usage_counts[t, y, x] += 1/N 
    
    # plot_control(usage_counts, T, size, size)
    
    # print(utilisation_correlated_init(walker, T=T, kernels=walker.kernels, end_x=200, end_y=280, output_folder="output_utilisation"))
    # walker.generate(bias_offsets=biases, start_x=200, start_y=50)
    
    
    # W = 201
    # H = 201
    # T = 50

    # start_x = 10
    # start_y = 10
    # end_x = 50
    # end_y = 50 
    # steps = [(start_x, start_y), (end_x, end_y)]

    # print(1)
    # terrain = create_terrain_map("landcover_142.txt", ' ')


    # mapping = create_mixed_kernel_parameters(MEDIUM, 23)


    # kernel_map = get_tensor_map_terrain(terrain, mapping)
    # print(2)

    # dp = mix_walk(W, H, terrain, kernel_map, T, start_x, start_y, False, False, "", mapping)

    # plot_dp_matrix(dp, 0, W, H)

    # print(dp)
    # print(3)
    # # walkptr = mix_backtrace(dp, T, kernel_map, terrain, end_x, end_y, 0, False, "", "", mapping)
    # # walkptr = mix_backtrace(dp, T, kernel_map, terrain, end_x, end_y, 0, False, "", "", mapping)
    # print(4)
    # # walk = get_walk_points(walkptr)
    # print(5)
    # plot_combined_terrain(terrain, walk, W, H, steps)



    