from random_walk_package import plot_combined_terrain
from random_walk_package.core.MixedTimeWalker import MixedTimeWalker
from random_walk_package.core.MixedWalker import *

studies = ["elephant_study/", "baboon_SA_study/", "leap_of_the_cat/", "Boars_Austria/", "Cranes Kazakhstan/"]


def mixed_walk_test():
    T = 10
    # todo: dynamic resolution based on bounding box size
    study = studies[2]
    kernel_mapping = create_mixed_kernel_parameters(animal_type=MEDIUM, base_step_size=4)
    """set_landmark_mapping(kernel_mapping, GRASSLAND, is_brownian=False, step_size=5, directions=6, diffusity=1)
    set_landmark_mapping(kernel_mapping, TREE_COVER, is_brownian=True,
                         step_size=5,
                         directions=1,
                         diffusity=2.6)"""

    walker = MixedWalker(T=T, resolution=300, kernel_mapping=kernel_mapping, study_folder=study)
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


def test_time_walker():
    walker = MixedTimeWalker(
        T=50,
        resolution=100,
        duration_in_days=3,
        study_folder="elephant_study/"
    )
    walker.generate_walk_from_movebank(serialized=False)


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
