# debugging: gdb --args python -m tests.test
import random

from random_walk_package.bindings.data_structures.kernel_terrain_mapping import marine_kernels_baseline, \
    update_kernels_mapping
from random_walk_package.core.MixedWalker import *
from tests.mixed_walk_test import test_time_walker


def weather_terrain_params(row):
    S = min(15, max(1, np.round(float(row["wind_speed_10m_max"]) / 2.0).astype(int)))
    D = min(16, max(1, np.round(int(row["wind_direction_10m_dominant"] // 45)).astype(int)))
    is_brownian = D == 1
    diffusity = float(row["cloud_cover_mean"]) / 100.0
    bias_x = int(row["precipitation_sum"] > 0.1)
    bias_y = int(row["terrain"] in (50, 60))
    return [is_brownian, S, D, diffusity, bias_x, bias_y]


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
    kernels_mapping = marine_kernels_baseline(step_size=5, directions=8, diffusity=2.1)
    # update the mapping parameters
    update_kernels_mapping(kernels_mapping, landmark=WATER, stepsize=7, directions=6, diffusity=1.5)

    exit()
    test_time_walker()
