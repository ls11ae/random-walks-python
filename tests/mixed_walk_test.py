from random_walk_package import create_correlated_kernel_parameters, MixedTimeWalker
from random_walk_package.bindings.data_structures.kernel_terrain_mapping import set_forbidden_landmark
from random_walk_package.core.MixedWalker import *
from random_walk_package.data_sources.walk_visualization import plot_trajectory_collection_timed

studies = ["turtles_study/Striped Mud Turtles (Kinosternon baurii) Lakeland, FL.csv",
           "movebank_test/The Leap of the Cat.csv",
           "Boars_Austria/",
           "Cranes Kazakhstan/"]


def test_mixed_walk():
    resources_dir = os.path.dirname("random_walk_package/resources/")
    study = os.path.join(resources_dir, studies[1])
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
                         resolution=200,
                         out_directory=out_dir,
                         time_col="timestamp",
                         lon_col="location-long",
                         lat_col="location-lat",
                         id_col="individual-local-identifier",
                         crs="EPSG:4326")
    walks_dir = out_dir
    trajectory_collection = walker.generate_walks()
    leaflet_path = plot_trajectory_collection_timed(trajectory_collection, save_path=str(walks_dir))


def weather_terrain_params(landmark, row):
    is_brownian = True
    S = float(row["wind_speed_10m_max"]) / 2.0
    D = int(row["wind_direction_10m_dominant"] // 45)
    diffusity = float(row["cloud_cover_mean"]) / 100.0
    bias_x = int(row["precipitation_sum"] > 0.1)
    bias_y = int(landmark in (50, 80))
    return [is_brownian, S, D, diffusity, bias_x, bias_y]


def test_time_walker():
    study = 'random_walk_package/resources/leap_of_the_cat/The Leap of the Cat.csv'
    df = pd.read_csv(study)

    environment_csv = 'random_walk_package/resources/movebank_test/weather/weather_data_full.csv'
    df_env = pd.read_csv(environment_csv)

    out_dir = os.path.dirname(study)

    mapping = create_mixed_kernel_parameters(animal_type=HEAVY, base_step_size=7)
    walker = MixedTimeWalker(data=df,
                             env_data=df_env,
                             kernel_mapping=mapping,
                             resolution=400,
                             out_directory=out_dir,
                             env_samples=5,
                             kernel_resolver=weather_terrain_params,
                             time_col="timestamp",
                             lon_col="location-long",
                             lat_col="location-lat",
                             id_col="tag-local-identifier",
                             crs="EPSG:4326"
                             )
    walker.generate_walks()
