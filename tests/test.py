# debugging: gdb --args python -m tests.test
import os
from datetime import datetime

import pandas as pd

from random_walk_package.core.AnimalMovement import AnimalMovementProcessor


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


if __name__ == "__main__":
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

    # Dataframe for your environmental data
    df_full = pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
    df_full.to_csv(os.path.join(data_dir, "weather_data_full.csv"))
    # path to the study containing the animal movement data
    study = 'leap_of_the_cat/The Leap of the Cat.csv'
    processor = AnimalMovementProcessor(study)
    # creates landcover grid txt files
    processor.create_landcover_data_txt(resolution=200, out_directory='leap_of_the_cat')
    # create kernel params csv files
    paths = processor.kernel_params_per_animal_csv(df=df_full,
                                                   kernel_resolver=weather_terrain_params,
                                                   start_date=datetime(2000, 8, 23),
                                                   end_date=datetime(2030, 12, 6),
                                                   time_stamp='timestamp',
                                                   lon='longitude',
                                                   lat='latitude')
    print(paths)
