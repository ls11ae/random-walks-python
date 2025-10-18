from random_walk_package.core.AnimalMovement import *
from random_walk_package.core.CorrelatedWalker import *
from random_walk_package.core.MixedWalker import *


# @profile
def test_terrain_correlated():
    file = "landcover_142.txt"
    terrain = parse_terrain(file, " ")

    T = 150
    S = 7
    D = 8
    walker = CorrelatedWalker(D=D, S=S, W=401, H=401, T=T)
    walker.generate_kernel()
    walker.generate_from_terrain(terrain=terrain, start_x=200, start_y=200)
    walk = walker.backtrace(end_x=355, end_y=385, initial_direction=0)
    plot_walk_terrain(terrain=terrain, walk_points=walk, terrain_height=walker.H, terrain_width=walker.W)


def corr(S, D, T):
    W = 2 * T + 1
    H = 2 * T + 1

    walker = CorrelatedWalker(D, S, W, H, T)
    walker.generate_kernel()

    # Time the execution

    walker.generate(start_x=50, start_y=50)
    start = time.perf_counter()
    walk = walker.backtrace(180, 190, 0)
    end = time.perf_counter()
    print(f"Generated walk in {end - start:.4f} seconds")
    plot_walk(walk, W, H)
    return None


def corr_simple(D):
    S = 15
    T = 120
    W = 401
    H = 401

    walker = CorrelatedWalker(D, S, W, H, T)
    walker.generate(start_x=200, start_y=100)

    # Time the execution
    start = time.perf_counter()
    walk = walker.backtrace(330, 280, 0)
    end = time.perf_counter()

    print(f"Generated walk in {end - start:.4f} seconds")
    plot_walk(walk, 401, 401)
