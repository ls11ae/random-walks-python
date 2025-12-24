# debugging: gdb --args python -m tests.test
import gzip
import os.path
import pickle
import random
from datetime import datetime

import pandas as pd

from random_walk_package import MixedTimeWalker
from random_walk_package.core.MixedWalker import *
from random_walk_package.core.StateDependentWalker import StateDependentWalker
from random_walk_package.data_sources.walk_visualization import save_trajectory_collection_timed, \
    save_trajectory_coll_leaflet
from random_walk_package.data_sources.walks_serialization import serialize_env_grid
from tests.mixed_walk_test import test_mixed_walk


def weather_terrain_params(landmark, row):
    is_brownian = True
    S = float(row["wind_speed_10m_max"]) / 2.0
    D = int(row["wind_direction_10m_dominant"] // 45)
    diffusity = float(row["cloud_cover_mean"]) / 100.0
    bias_x = int(row["precipitation_sum"] > 0.1)
    bias_y = int(landmark in (50, 80))
    return [is_brownian, S, D, diffusity, bias_x, bias_y]


def save_walks_pickle(traj_coll, pickle_path="walks.pickle"):
    with gzip.open(pickle_path, 'wb') as f:
        pickle.dump(traj_coll, f, protocol=pickle.HIGHEST_PROTOCOL)
        return pickle_path


def load_walks_pickle(filepath):
    with gzip.open(filepath, 'rb') as f:
        return pickle.load(f)


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
        int(bias_x),
        int(bias_y),
    ]


if __name__ == "__main__":
    study_path = 'random_walk_package/resources/movebank_test/The Leap of the Cat.csv'
    study_df = pd.read_csv(study_path)
    env_samples = 5
    # i took the original csv but this also works for your processed csv with additional data, just adapt the kernel resolver
    # env_path = '/home/omar/Downloads/current_filename.csv'
    env_path = 'random_walk_package/resources/movebank_test/weather/weather_data_full.csv'
    processor = AnimalMovementProcessor(study_df, env_samples=env_samples)
    processor.create_landcover_data_txt(is_marine=True, resolution=600, out_directory=os.path.dirname(study_path))
    processor.kernel_params_per_animal_binary(env_path=env_path,
                                              kernel_resolver=marine_params,
                                              time_stamp="timestamp", lon="longitude", lat="latitude",
                                              out_directory=os.path.dirname(study_path))
