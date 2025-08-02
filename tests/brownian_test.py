import time

from random_walk_package.core.BiasedWalker import BiasedWalker
from random_walk_package.core.BrownianWalker import *


def test_terrain():
    file = "landcover_142.txt"
    terrain = get_terrain_map(file, " ")

    num_steps = 200
    step_size = 7
    brownian_walker = BrownianWalker.from_terrain_map(terrain=terrain, T=num_steps, S=step_size, sigma=0.5, start_x=200,
                                                      start_y=200)
    walk = brownian_walker.generate_walk(375, 375)
    plot_walk_terrain(terrain=terrain, walk_points=walk, terrain_height=brownian_walker.H,
                      terrain_width=brownian_walker.W)


def benchmark_brownian(num_steps=100):
    brownian_walker = BrownianWalker.from_parameters(T=num_steps, S=7, W=2 * num_steps + 1, H=2 * num_steps + 1)
    start = time.perf_counter()
    brownian_walker.generate_walk(175, 175)
    end = time.perf_counter()
    print(f"Generated walk in {end - start:.4f} seconds")


def test_brownian_wrapper():
    # Test 1: Initialize with parameters
    num_steps = 100
    step_size = 8
    brownian_walker = BrownianWalker.from_parameters(T=num_steps, S=step_size, sigma=0.5, scale=2)

    kernel_np = matrix_to_numpy(brownian_walker.kernel)
    print(kernel_np)

    # Test 2: Initialize with existing kernel/tensor
    brownian_walker2 = BrownianWalker.from_kernel_and_tensor(
        kernel=brownian_walker.kernel,
        dp_tensor=brownian_walker.dp_tensor,
        T=num_steps,
        W=brownian_walker.W,
        H=brownian_walker.H
    )

    # Test 3: Save/load functionality
    brownian_walker.save_tensor("test_tensor.bin")
    brownian_walker3 = BrownianWalker.load_tensor(
        filename="test_tensor.bin",
        kernel=brownian_walker.kernel,
        T=num_steps,
        W=brownian_walker.W,
        H=brownian_walker.H
    )

    # Test 4: Single walk generation
    walk = brownian_walker.generate_walk(end_x=num_steps // 3, end_y=num_steps // 4)
    plot_walk(walk_points=walk, terrain_height=brownian_walker.H, terrain_width=brownian_walker.W)

    walk2 = brownian_walker.generate_walk(end_x=num_steps // 2, end_y=num_steps)
    plot_walk(walk_points=walk2, terrain_height=brownian_walker2.H, terrain_width=brownian_walker2.W)

    walk3 = brownian_walker.generate_walk(end_x=num_steps // 5, end_y=num_steps // 2)
    plot_walk(walk_points=walk3, terrain_height=brownian_walker3.H, terrain_width=brownian_walker3.W)

    # Test 5: Multistep walk generation
    points = [(num_steps, num_steps), (num_steps // 2, num_steps // 2), (num_steps + 10, num_steps - 10)]
    multi_walk = brownian_walker.generate_multistep_walk(points)

    plot_walk_multistep(steps=points, walk_points=multi_walk, terrain_width=brownian_walker.W,
                        terrain_height=brownian_walker.H)


def movebank_test():
    csv_file = "1000 Cranes. Southern Kazakhstan.csv"
    [walk_csv, steps_csv] = BrownianWalker.generate_movebank_walk(csv_file, 10, 7, 100, 200, 200)
    walk_plot_array_csv = get_walk_points(walk_csv)
    plot_walk_multistep(steps_csv, walk_plot_array_csv, 200, 200)


def test_biased_walk():
    terrain = parse_terrain("terrain2.txt", " ")
    west_bias = [(-4, 0)] * 20
    no_bias = [(0, 0)] * 20
    east_bias = [(4, 0)] * 20
    biases = west_bias + no_bias + east_bias
    biased_walk = BiasedWalker(terrain=terrain, bias_array=biases)
    biased_walk.generate(start_x=100, start_y=150)
    walk = biased_walk.backtrace(end_x=100, end_y=45)
    plot_combined_terrain(
        terrain,
        walk,
        terrain.width,
        terrain.height,
        title="Time-Aware Mixed Walk"
    )
