# debugging: gdb --args python -m tests.test
import gzip
import pickle
import random
import pandas as pd
import numpy as np
from random_walk_package import create_correlated_kernel_parameters
from random_walk_package.bindings.data_structures.kernel_terrain_mapping import marine_kernels_baseline, \
    update_kernels_mapping
from random_walk_package.bindings.plotter import plot_walk_from_json
from random_walk_package.core.MixedWalker import *
from random_walk_package.core.MovementPolicy import TimeStepPolicy
from random_walk_package.core.StateDependentWalker import StateDependentWalker
from random_walk_package.data_sources.walk_visualization import save_trajectory_collection_timed
from tests.mixed_walk_test import test_marine_walker, test_time_walker, test_mixed_walk
from random_walk_package.core.MarineMovement import shark_data_filter


def weather_terrain_params(row):
    S = min(15, max(1, np.round(float(row["wind_speed_10m_max"]) / 2.0).astype(int)))
    D = min(16, max(1, np.round(int(row["wind_direction_10m_dominant"] // 45)).astype(int)))
    is_brownian = D == 1
    diffusity = float(row["cloud_cover_mean"]) / 100.0
    bias_x = int(row["precipitation_sum"] > 0.1)
    bias_y = int(row["terrain"] in (50, 60))
    return [is_brownian, S, D, diffusity, bias_x, bias_y]

if __name__ == "__main__":
    plot_walk_from_json("random_walk_package/resources/tiger_sharks/kernels/204413/.json")
    test_marine_walker()
    exit()


    
    study = "random_walk_package/resources/elephants/Elephant Research - Lobeke National Park (Cameroon) - Collar 46179.csv"
    df = pd.read_csv(study)  # or a traj collection

    out_dir = os.path.dirname(study)
    walker = StateDependentWalker(data=df, animal_type=MEDIUM, resolution=1000,
                                  out_directory=out_dir)  # data can also be a traj collection (MoveApp's input)
    mvm_pol = TimeStepPolicy(60 * 15)
    traj_coll = walker.generate_walks(dt_tolerance=100.0, rnge=1000, movement_policy=mvm_pol)  # gets output from MoveApp0
    walk_dir = os.path.join(out_dir, "walks")
    os.makedirs(walk_dir, exist_ok=True)
    save_trajectory_collection_timed(traj_coll, str(walk_dir))
    pickle_path = os.path.join(walk_dir, "state_walks.pickle")
    with gzip.open(pickle_path, 'wb') as f:
        pickle.dump(traj_coll, f, protocol=pickle.HIGHEST_PROTOCOL)

    # test_marine_walker()
    plot_walk_from_json(
        "/home/omar/PycharmProjects/random-walks-python/random_walk_package/resources/tiger_sharks/kernels/204413/.json")

    test_marine_walker()
    exit()

    study_path = 'random_walk_package/resources/tiger_sharks/shark_13_filtered.csv'
    study_df = pd.read_csv(study_path)
    env_samples = 5
    # i took the original csv but this also works for your processed csv with additional data, just adapt the kernel resolver
    env_path = '/home/omar/Downloads/current_filename.csv'
    # env_path = 'random_walk_package/resources/movebank_test/weather/weather_data_full.csv'
    processor = AnimalMovementProcessor(study_df, env_samples=env_samples)
    processor.create_landcover_data_txt(is_marine=True, resolution=1000, out_directory=os.path.dirname(study_path))
    processor.kernel_params_per_animal_binary(env_path=env_path,
                                              kernel_resolver=marine_params,
                                              time_stamp="time", lon="longitude", lat="latitude",
                                              out_directory=os.path.dirname(study_path))
    # example mapping for marine animals. Water is the only allowed landmark, motion is always correlated
    kernels_mapping = marine_kernels_baseline(step_size=5, directions=8, angle_diffusity=0.3, len_diffusivity=1)
    # update the mapping parameters
    update_kernels_mapping(kernels_mapping, landmark=WATER, stepsize=7, directions=6, diffusity=1.5)

    exit()
    test_marine_walker()
