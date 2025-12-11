import pandas as pd

from random_walk_package.bindings.data_structures.kernel_terrain_mapping import set_forbidden_landmark, \
    set_landmark_mapping
from random_walk_package.core.MixedWalker import *

studies = ["elephant_study/", "baboon_SA_study/", "random_walk_package/resources/movebank_test/The Leap of the Cat.csv", "Boars_Austria/", "Cranes Kazakhstan/"]


def test_mixed_walk():
    # todo: dynamic resolution based on bounding box size
    study = studies[2]
    df = pd.read_csv(study)
    kernel_mapping = create_mixed_kernel_parameters(animal_type=MEDIUM, base_step_size=4)
    set_landmark_mapping(kernel_mapping, GRASSLAND, is_brownian=False, step_size=5, directions=6, diffusity=1)
    set_landmark_mapping(kernel_mapping, TREE_COVER, is_brownian=True,
                         step_size=5,
                         directions=1,
                         diffusity=2.6)
    set_forbidden_landmark(kernel_mapping, WATER)
    out_dir = "random_walk_package/resources/movebank_test/"
    walker = MixedWalker(data=df,
                         kernel_mapping=kernel_mapping,
                         resolution=100,
                         out_directory=out_dir,
                         time_col="timestamp",
                         lon_col="location-long",
                         lat_col="location-lat",
                         id_col="tag-local-identifier",
                         crs="EPSG:4326")
    walks_dir = out_dir
    trajectory_collection = walker.generate_movebank_walks(walks_dir)




"""def test_time_walker():
    walker = MixedTimeWalker(
        T=50,
        resolution=100,
        duration_in_days=3,
        study_folder="leap_of_the_cat/"
    )
    walker.generate_walk_from_movebank()
"""
