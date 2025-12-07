from random_walk_package.bindings.data_structures.kernel_terrain_mapping import set_forbidden_landmark, \
    set_landmark_mapping
from random_walk_package.core.MixedWalker import *

studies = ["elephant_study/", "baboon_SA_study/", "movebank_test/", "Boars_Austria/", "Cranes Kazakhstan/"]


def test_mixed_walk():
    T = 10
    # todo: dynamic resolution based on bounding box size
    study = studies[2]
    kernel_mapping = create_mixed_kernel_parameters(animal_type=MEDIUM, base_step_size=5)
    set_landmark_mapping(kernel_mapping, GRASSLAND, is_brownian=False, step_size=5, directions=6, diffusity=1)
    set_landmark_mapping(kernel_mapping, TREE_COVER, is_brownian=True,
                         step_size=5,
                         directions=1,
                         diffusity=2.6)
    set_forbidden_landmark(kernel_mapping, WATER)

    walker = MixedWalker(T=T, resolution=300, kernel_mapping=kernel_mapping, study_folder=study)
    walk_path = walker.generate_movebank_walks(serialized=False)
    assert len(walk_path) > 0


"""def test_time_walker():
    walker = MixedTimeWalker(
        T=50,
        resolution=100,
        duration_in_days=3,
        study_folder="leap_of_the_cat/"
    )
    walker.generate_walk_from_movebank()
"""
