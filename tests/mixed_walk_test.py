import gzip
import math
import os
import pickle
import random

import numpy as np
import pytest

import pandas as pd

from random_walk_package import MixedWalker, GRASSLAND, WATER, TREE_COVER, MixedTimeWalker, MEDIUM
from random_walk_package import create_correlated_kernel_parameters, set_forbidden_landmark, set_landmark_mapping
from random_walk_package.bindings import create_mixed_kernel_parameters
from random_walk_package.bindings.data_structures.EnvWeights import EnvWeights
from random_walk_package.bindings.data_structures.kernel_terrain_mapping import marine_kernels_baseline
from random_walk_package.core.MovementPolicy import TimeStepPolicy
from random_walk_package.data_sources.walk_visualization import save_trajectory_collection_timed, \
    save_trajectory_coll_leaflet

studies = ["turtles_study/Striped Mud Turtles (Kinosternon baurii) Lakeland, FL.csv",
           "movebank_test/The Leap of the Cat.csv",
           "leap_of_the_cat/The Leap of the Cat.csv",
           "ruby_throat_china/Siberian rubythroat tracking from Qinghai, China.csv"
           ]


def test_mixed_walk():
    resources_dir = os.path.dirname("random_walk_package/resources/")
    study = os.path.join(resources_dir, studies[2])
    df = pd.read_csv(study)
    kernel_mapping = create_correlated_kernel_parameters(animal_type=MEDIUM, base_step_size=3)
    set_landmark_mapping(kernel_mapping, GRASSLAND, is_brownian=False, step_size=3, directions=8)
    set_landmark_mapping(kernel_mapping, TREE_COVER, is_brownian=True,
                         step_size=2,
                         directions=4)
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
    walks_dir = os.path.join(out_dir, "walks")
    os.makedirs(walks_dir, exist_ok=True)
    trajectory_collection = walker.generate_walks()
    save_trajectory_coll_leaflet(trajectory_collection, save_path=walks_dir)
    save_trajectory_collection_timed(trajectory_collection, save_path=str(walks_dir))
    return trajectory_collection


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
    angle_diffusivity = 0.5 * max(cloud_diffusion, precip_diffusion) + 0.1 * env_stochasticity
    angle_diffusivity = min(0.95, angle_diffusivity)
    len_diffusivity = 0.9

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
        angle_diffusivity = 0.9

    return [bool(is_brownian), float(S), int(D), float(len_diffusivity), float(angle_diffusivity),
            float(bias_x), float(bias_y)]


# map row of your csv to kernel params, terrain is always part of a row, so is x,y,t if needed
# keep in mind that NaN values can (and almost always) appear so must be handled here (unless you filled them earlier)
def marine_params(row):
    uo = row.get("uo")
    vo = row.get("vo")

    if pd.isna(uo) or pd.isna(vo):
        bias_x = 0
        bias_y = 0
        is_brownian = False
        diffusity = 1.0
    else:
        bias_x = int(np.round(float(uo) * 10))
        bias_y = int(np.round(float(vo) * 10))
        is_brownian = row.get("depth", 0) < 0.2
        diffusity = 0.9

    S = random.randint(3, 7)
    D = 8

    return [
        bool(is_brownian),
        float(S),
        int(D),
        float(diffusity),
        float(0.4),
        int(bias_x),
        int(bias_y),
    ]

def marine_resolver(row, start, end, S):
    
    """
    ASSUMES TRAVELING STATE i.e current-positive, not avoiding or actively resisting 
    Compute the ocean-current displacement (meters) for a timestep dt. A systematic displacement caused by
    the ocean currents that affects the shark’s trajectory, added on top of the shark’s swimming movement.
    """
    #start end 
    ref_speed = 1.5
    is_brownian = True
    D = 1
    length_diffusity= 1.0
    angle_diffusity = 0.5

    BIAS_STRENGTH = 0.7
    
    if pd.isna(row.get("uo")):
         uo = 0.0
    else: uo= row.get("uo", 0.0)
    if pd.isna(row.get("vo")):
        vo = 0.0
    else: vo =row.get("vo", 0.0)
    
    current_vec = np.array([uo, vo])
    creature_vector =  np.array([end[0] - start[0], end[1] - start[1]])
    creature_norm = np.linalg.norm(creature_vector)
    if creature_norm == 0 or np.isnan(creature_norm):
        creature_norm =0.001
    creature_vector= creature_vector / creature_norm #check norm func
    creature_vector = ref_speed * creature_vector
    mixed_vel = creature_vector + current_vec
    mixed_dir = mixed_vel
    if np.linalg.norm(mixed_vel) == 0 or np.isnan(np.linalg.norm(mixed_vel)):
        bias_x = 0
        bias_y = 0
    else:
        mixed_dir = mixed_dir / np.linalg.norm(mixed_dir)
        bias_x = float(mixed_dir[0]) * S * BIAS_STRENGTH
        bias_y = float(mixed_dir[1]) * S * BIAS_STRENGTH


    
    if not np.isfinite(bias_x):
        bias_x = 0.0
    if not np.isfinite(bias_y):
        bias_y = 0.0
 

    #correct dir and reff speed 
    # we need to normalize by speed but by the constant, norm the creature vector to have unit length and mult by the lit.speed and 
    return [bool(is_brownian), int(S), int(D), float(length_diffusity), float(angle_diffusity),
        int(np.round(bias_x)), int(np.round(bias_y))]


@pytest.mark.skip(reason="takes too long")
def test_marine_walker():
    study = 'random_walk_package/resources/tiger_sharks/shark_13_filtered_full.csv'
    df = pd.read_csv(study)

    environment_csv = '/home/omar/Downloads/current_filename.csv'
    out_dir = os.path.dirname(study)

    mapping = marine_kernels_baseline(step_size=5, directions=1, angle_diffusity=0.3, len_diffusivity=1)
    movement_policy = TimeStepPolicy(timestep_s=3600 * 6)  # 8 hours per step
    walker = MixedTimeWalker(data=df,
                             env_data=environment_csv,
                             kernel_mapping=mapping,
                             resolution=800,
                             out_directory=out_dir,
                             env_samples=5,
                             kernel_resolver=marine_resolver,
                             time_col="timestamp",
                             lon_col="location-long",
                             lat_col="location-lat",
                             id_col="tag-local-identifier",
                             crs="EPSG:4326",
                             is_marine=True,
                             movement_policy=movement_policy,
                             reference_speed=1.5)

    bias_only = EnvWeights.bias_only()

    trajectory_collection = walker.generate_walks(movement_policy=movement_policy, env_weights=bias_only)

    walks_dir = os.path.dirname(study)
    walks_dir = os.path.join(walks_dir, "walks")
    os.makedirs(walks_dir, exist_ok=True)
    # serialize trajectory collection
    pickle_path = os.path.join(walks_dir, "walks.pickle")
    with gzip.open(pickle_path, 'wb') as f:
        pickle.dump(trajectory_collection, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_trajectory_collection_timed(trajectory_collection, walks_dir)
    print(f"walks saved at {walks_dir}")


@pytest.mark.skip(reason="takes too long")
def test_time_walker():
    study = 'random_walk_package/resources/movebank_test/The Leap of the Cat.csv'
    df = pd.read_csv(study)

    environment_csv = 'random_walk_package/resources/movebank_test/weather/weather_data_full.csv'
    out_dir = os.path.dirname(study)

    mapping = create_mixed_kernel_parameters(MEDIUM, 5)
    set_landmark_mapping(mapping, GRASSLAND, is_brownian=False, step_size=3, directions=12, len_diffusity=0.7,
                         angle_diffusity=0.2)
    walker = MixedTimeWalker(data=df,
                             env_data=environment_csv,
                             kernel_mapping=mapping,
                             resolution=400,
                             out_directory=out_dir,
                             env_samples=5,
                             kernel_resolver=weather_terrain_params,
                             time_col="timestamp",
                             lon_col="location-long",
                             lat_col="location-lat",
                             id_col="tag-local-identifier",
                             crs="EPSG:4326",
                             is_marine=False)
    movement_policy = TimeStepPolicy(timestep_s=3600 * 4)  # 4 hours per step
    bias_only = EnvWeights.bias_only()
    trajectory_collection = walker.generate_walks(movement_policy=movement_policy, env_weights=bias_only)

    walks_dir = os.path.dirname(study)
    walks_dir = os.path.join(walks_dir, "walks")
    os.makedirs(walks_dir, exist_ok=True)
    # serialize trajectory collection
    pickle_path = os.path.join(walks_dir, "walks.pickle")
    with gzip.open(pickle_path, 'wb') as f:
        pickle.dump(trajectory_collection, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_trajectory_collection_timed(trajectory_collection, walks_dir)
    print(f"walks saved at {walks_dir}")
