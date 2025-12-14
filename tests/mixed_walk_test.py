import math

from random_walk_package import create_correlated_kernel_parameters
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


def weather_terrain_params(row):
    # "daily": "weather_code,temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,snowfall_sum,
    # wind_speed_10m_max,wind_direction_10m_dominant,cloud_cover_mean",

    # --- Step Size (S) based on wind speed ---
    wind_speed = float(row["wind_speed_10m_max"])
    base_step = 3.0
    S = base_step * (1 + math.log1p(wind_speed / 5.0))
    S = min(S, 8.0)
    wind_dir_deg = float(row["wind_direction_10m_dominant"])
    temp = float(row.get("temperature_2m_mean", 20))
    humidity = float(row.get("relative_humidity_2m_mean", 50))
    env_stochasticity = (temp - 10) / 30 * 0.5 + (humidity - 30) / 70 * 0.5
    env_stochasticity = max(0, min(1, env_stochasticity))
    if env_stochasticity > 0.7 or wind_speed < 1.0:
        is_brownian = True
        D = 1
    else:
        is_brownian = False
        wind_strength_factor = min(1.0, wind_speed / 10.0)
        if wind_strength_factor > 0.7:
            D = 8
        else:
            D = 4

    cloud_cover = float(row.get("cloud_cover_mean", 50))
    precipitation = float(row.get("precipitation_sum", 0))
    cloud_diffusion = cloud_cover / 100.0
    precip_diffusion = min(1.0, precipitation * 2.0)
    diffusity = 0.3 + 0.5 * max(cloud_diffusion, precip_diffusion) + 0.2 * env_stochasticity
    diffusity = min(0.95, diffusity)

    wind_rad = math.radians(wind_dir_deg)
    wind_x_bias = math.sin(wind_rad)

    precip_bias = -0.3 if precipitation > 0.5 else 0.0
    bias_x = 0.7 * wind_x_bias + 0.3 * precip_bias
    wind_y_bias = -math.cos(wind_rad)

    # Combined y-bias (-1 = strong north, +1 = strong south)
    bias_y = 0.5 * wind_y_bias
    bias_x = max(-1.0, min(1.0, bias_x))
    bias_y = max(-1.0, min(1.0, bias_y))

    snowfall = float(row.get("snowfall_sum", 0))
    if snowfall > 5.0:
        D = max(4, D // 2)

    weather_code = int(row.get("weather_code", 0))
    if weather_code in [95, 96, 99]:
        is_brownian = True
        D = 1
        diffusity = 0.9

    return [bool(is_brownian), float(S), int(D), float(diffusity),
            float(bias_x), float(bias_y)]


"""def test_time_walker():
    study = 'random_walk_package/resources/leap_of_the_cat/The Leap of the Cat.csv'
    df = pd.read_csv(study)

    environment_csv = 'random_walk_package/resources/movebank_test/weather/weather_data_full.csv'
    df_env = pd.read_csv(environment_csv)

    out_dir = os.path.dirname(study)

    mapping = create_mixed_kernel_parameters(animal_type=MEDIUM, base_step_size=5)
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
    trajectory_collection = walker.generate_walks()

    walks_dir = os.path.dirname(study)
    walks_dir = os.path.join(walks_dir, "walks")
    os.makedirs(walks_dir, exist_ok=True)
    # serialize trajectory collection
    pickle_path = os.path.join(walks_dir, "walks.pickle")
    with gzip.open(pickle_path, 'wb') as f:
        pickle.dump(trajectory_collection, f, protocol=pickle.HIGHEST_PROTOCOL)

    leaflet_path = plot_trajectory_collection_timed(trajectory_collection, save_path=str(walks_dir))"""
