from memory_profiler import profile

from random_walk_package.bindings.data_structures.kernel_terrain_mapping import set_landmark_mapping
from random_walk_package.core.MixedTimeWalker import MixedTimeWalker
from random_walk_package.core.MixedWalker import *


def mixed_walk_test():
    T = 10

    study = "catharus_bicknelly/"
    kernel_mapping = create_mixed_kernel_parameters(animal_type=AIRBORNE, base_step_size=3)
    set_landmark_mapping(kernel_mapping, GRASSLAND, is_brownian=False, step_size=3, directions=4, diffusity=1, max_bias_x=0,
                         max_bias_y=0)
    set_landmark_mapping(kernel_mapping, TREE_COVER, is_brownian=True,
                         step_size=3,
                         directions=1,
                         diffusity=1.6,
                         max_bias_x=0, max_bias_y=0)
    walker = MixedWalker(T=T, resolution=600, animal_type=AIRBORNE, kernel_mapping=kernel_mapping, S=3, study_folder=study)
    #steps = [(166, 166), (422, 300)]
    walker.generate_walk(serialized=False)

# @profile
def test_time_walk():
    terrain = parse_terrain("time_walk_data/terrain_movebank.txt", " ")
    T = 100

    start = (100, 120)
    mid = (150, 120)
    end = (60, 20)

    steps = [start, mid, end]
    kernels_mapping = create_mixed_kernel_parameters(animal_type=MEDIUM, base_step_size=7)

    walk_points = time_walk_geo_multi(
        T=T,
        csv_path="time_walk_data/my_gridded_weather_grid_csvs",
        terrain_path="time_walk_data/terrain_movebank.txt",
        walk_path="time_walk_data/time_walk.json",
        grid_x=5,
        grid_y=5,
        steps=steps,
        mapping=kernels_mapping
    )
    walknp = get_walk_points(walk_points)
    plot_combined_terrain(terrain, walk_points=walknp, terrain_height=terrain.height, terrain_width=terrain.width,
                          title="Time-Aware Mixed Walk")


@profile
def test_time_walker():
    start = (50, 70)
    end = (10, 10)

    walker = MixedTimeWalker(
        T=50,
        resolution=100,
        duration_in_days=3,
        study_folder="elephant_study/"
    )
    walker.preprocess()
    walker.generate_walk(start=start, end=end, output_file="time_walk3.json")


def test_time_walker_multi():
    start = (100, 120)
    mid = (60, 20)
    end = (150, 120)

    steps = [start, mid, end]

    walker = MixedTimeWalker(
        T=50,
        resolution=200,
        duration_in_days=5,
        study_folder="baboon_SA_study/"
    )
    walker.preprocess()
    walker.generate_walk_multi(steps=steps, output_file="time_walk3.json", serialized=True)
