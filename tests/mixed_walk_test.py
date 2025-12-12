from random_walk_package import create_correlated_kernel_parameters
from random_walk_package.bindings.data_structures.kernel_terrain_mapping import set_forbidden_landmark
from random_walk_package.core.MixedWalker import *

studies = ["turtles_study/Striped Mud Turtles (Kinosternon baurii) Lakeland, FL.csv",
           "movebank_test/The Leap of the Cat.csv",
           "Boars_Austria/",
           "Cranes Kazakhstan/"]


def test_mixed_walk():
    resources_dir = os.path.dirname("random_walk_package/resources/")
    study = os.path.join(resources_dir, studies[0])
    df = pd.read_csv(study)
    kernel_mapping = create_correlated_kernel_parameters(animal_type=MEDIUM, base_step_size=3)
    """set_landmark_mapping(kernel_mapping, GRASSLAND, is_brownian=False, step_size=4, directions=8, diffusity=1)
    set_landmark_mapping(kernel_mapping, TREE_COVER, is_brownian=True,
                         step_size=4,
                         directions=1,
                         diffusity=2.6)"""
    set_forbidden_landmark(kernel_mapping, WATER)

    out_dir = os.path.dirname(study)
    walker = MixedWalker(data=df,
                         kernel_mapping=kernel_mapping,
                         resolution=400,
                         out_directory=out_dir,
                         time_col="timestamp",
                         lon_col="location-long",
                         lat_col="location-lat",
                         id_col="individual-local-identifier",
                         crs="EPSG:4326")
    walks_dir = out_dir
    trajectory_collection = walker.generate_movebank_walks()


"""def test_time_walker():
    walker = MixedTimeWalker(
        T=50,
        resolution=100,
        duration_in_days=3,
        study_folder="leap_of_the_cat/"
    )
    walker.generate_walk_from_movebank()
"""
