# debugging: gdb --args python -m tests.test
import os
from datetime import datetime

import pandas as pd

from random_walk_package import create_terrain_map, set_forbidden_landmark, WATER, KernelParamsYXT, \
    EnvironmentInfluenceGrid, dll, point2d_arr_free, MixedTimeWalker
from random_walk_package.bindings import create_mixed_kernel_parameters, HEAVY, get_walk_points, terrain_map_free
from random_walk_package.bindings.data_processing.environment_handling import parse_kernel_parameters, \
    get_kernels_environment_grid, free_environment_influence_grid, free_kernel_parameters_yxt
from random_walk_package.bindings.data_structures.kernel_terrain_mapping import kernel_mapping_free
from random_walk_package.bindings.mixed_walk import environment_mixed_walk
from random_walk_package.bindings.plotter import plot_combined_terrain
from random_walk_package.core.AnimalMovementNew import AnimalMovementProcessor
from tests.mixed_walk_test import test_mixed_walk


def weather_terrain_params(landmark, row):
    is_brownian = True
    S = float(row["wind_speed_10m_max"]) / 2.0
    D = int(row["wind_direction_10m_dominant"] // 45)
    diffusity = float(row["cloud_cover_mean"]) / 100.0
    bias_x = int(row["precipitation_sum"] > 0.1)
    bias_y = int(landmark in (50, 80))
    return [is_brownian, S, D, diffusity, bias_x, bias_y]


def read_weather_csv(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.loc[:, df.columns != "index"]

    return df


def environment_pipeline_test():
    # merging weather data to one dataset
    out_dir = "weather_data_full.csv"
    data_dir = "/home/omar/PycharmProjects/random-walks-python/random_walk_package/resources/leap_of_the_cat/weather_data/CAMILA"
    csv_files = [
        os.path.join(data_dir, file)
        for file in os.listdir(data_dir)
        if file.endswith(".csv") and not file.endswith(out_dir)
    ]
    print(len(csv_files))
    dfs = (read_weather_csv(f) for f in csv_files)

    # Dataframe for your environmental data, for weather i need to merge the CSVs, normally you wouldnt need this
    df_full = pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
    df_full.to_csv(os.path.join(data_dir, "weather_data_full.csv"))

    # path to the study containing the animal movement data
    study = 'leap_of_the_cat/The Leap of the Cat.csv'
    processor = AnimalMovementProcessor(study)
    # creates landcover grid txt files
    processor.create_landcover_data_txt(resolution=200, out_directory='leap_of_the_cat')

    # although filtering of dates also happens in C, it makes sense to set the dates of the interval of the study here
    start_date = datetime(1999, 8, 23)
    end_date = datetime(2030, 12, 6)

    # create kernel params csv files
    paths, times = processor.kernel_params_per_animal_csv(df=df_full,
                                                          kernel_resolver=weather_terrain_params,
                                                          start_date=start_date,
                                                          end_date=end_date,
                                                          time_stamp='timestamp',
                                                          lon='longitude',
                                                          lat='latitude')

    dimensions = processor.env_samples, processor.env_samples, times

    # C allocated, must be freed manually
    environment_parameters: EnvironmentInfluenceGrid = parse_kernel_parameters(paths['CAMILA'], start_date, end_date,
                                                                               dimensions)
    # some terrain txt, path starts from resources/ folder
    terrain = create_terrain_map("baboon_SA_study/landcover_baboons123_200.txt", " ")

    # mapping which defines kernel parameters based on landmark
    mapping = create_mixed_kernel_parameters(animal_type=HEAVY, base_step_size=7)
    set_forbidden_landmark(mapping, landmark=WATER)

    # C structure holding kernel params for each x,y in terrain grid and t in interval start_date, end_date
    kernel_environment: KernelParamsYXT = get_kernels_environment_grid(terrain, environment_parameters, mapping,
                                                                       environment_weight=0.5)
    # the actual walk: for a better idea, check out the MixedTimeWalk implementation (and how these details are abstracted/bundled for the user) in core/
    # for each 2 consecutive points, we set the number of steps we want in the RW, and the correct kernels are computed and used
    T = 80
    # the projected animal coords you iterate
    start_point = (50, 50)
    end_point = (150, 150)

    walk = environment_mixed_walk(T, mapping, terrain, kernel_environment, start_date, end_date, start_point, end_point)
    dll.point2d_array_print(walk)
    walknp = get_walk_points(walk)
    # print(terrain.contents.width, terrain.contents.height)
    plot_combined_terrain(terrain, walknp, steps=[(50, 50), (150, 150)], title="⋆༺︎⋆ my fancy walk ⋆༻⋆")

    # free C allocated memory (i will probably do that on the C side instead, unless we need these multiple times)
    point2d_arr_free(walk)
    free_kernel_parameters_yxt(kernel_environment)
    kernel_mapping_free(mapping)
    terrain_map_free(terrain)
    free_environment_influence_grid(environment_parameters)


if __name__ == "__main__":
    test_mixed_walk()
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

    processor = AnimalMovementProcessor(data=df,
                                        lat_col="location-lat",
                                        lon_col="location-long",
                                        time_col="timestamp",
                                        id_col="tag-local-identifier",
                                        crs="EPSG:4326")
    # creates landcover grid txt files
    processor.create_landcover_data_txt(resolution=500,
                                        out_directory='random_walk_package/resources/leap_of_the_cat/terrain')
    # processor.fetch_open_meteo_weather('random_walk_package/resources/leap_of_the_cat/weather', samples_per_dimension=2)
    movement_data = processor.create_movement_data_dict()

    start_date = datetime(2000, 8, 24)
    end_date = datetime(2001, 1, 15)

    # create kernel params csv files
    paths, _ = processor.kernel_params_per_animal_csv(df=df_env,
                                                      kernel_resolver=weather_terrain_params,
                                                      start_date=start_date,
                                                      end_date=end_date,
                                                      time_stamp='timestamp',
                                                      lon='longitude',
                                                      lat='latitude',
                                                      out_directory='random_walk_package/resources/leap_of_the_cat/kernel_data')
    print(paths)
