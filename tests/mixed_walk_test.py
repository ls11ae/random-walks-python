from random_walk_package.bindings.data_structures.kernel_terrain_mapping import set_forbidden_landmark, \
    set_landmark_mapping
from random_walk_package.core.MixedTimeWalker import MixedTimeWalker
from random_walk_package.core.MixedWalker import *

studies = ["elephant_study/", "baboon_SA_study/", "leap_of_the_cat/", "Boars_Austria/", "Cranes Kazakhstan/"]


def mixed_walk_test():
    T = 10
    # todo: dynamic resolution based on bounding box size
    study = studies[3]
    kernel_mapping = create_mixed_kernel_parameters(animal_type=MEDIUM, base_step_size=4)
    set_landmark_mapping(kernel_mapping, GRASSLAND, is_brownian=False, step_size=5, directions=6, diffusity=1)
    set_landmark_mapping(kernel_mapping, TREE_COVER, is_brownian=True,
                         step_size=5,
                         directions=1,
                         diffusity=2.6)
    set_forbidden_landmark(kernel_mapping, WATER)

    walker = MixedWalker(T=T, resolution=300, kernel_mapping=kernel_mapping, study_folder=study)
    walker.generate_walk(serialized=False)


def test_time_walker():
    walker = MixedTimeWalker(
        T=50,
        resolution=100,
        duration_in_days=3,
        study_folder="elephant_study/"
    )
    walker.generate_walk_from_movebank(serialized=False)
