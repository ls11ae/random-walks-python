from random_walk_package.bindings.data_structures.kernel_terrain_mapping import set_forbidden_landmark, \
    set_landmark_mapping
from random_walk_package.core.MixedTimeWalker import MixedTimeWalker
from random_walk_package.core.MixedWalker import *

studies = ["elephant_study/", "baboon_SA_study/", "leap_of_the_cat/", "Boars_Austria/", "Cranes Kazakhstan/"]

from IPython.display import *


def mixed_walk_test():
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
    print(walk_path)
    # Read the HTML file and display
    with open(walk_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    display(HTML(html_content))


def test_time_walker():
    walker = MixedTimeWalker(
        T=50,
        resolution=100,
        duration_in_days=3,
        study_folder="elephant_study/"
    )
    walker.generate_walk_from_movebank(serialized=False)
