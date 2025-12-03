# debugging: gdb --args python -m tests.test
from datetime import datetime

import pandas

from random_walk_package.core.AnimalMovement import AnimalMovementProcessor
from tests.brownian_test import test_brownian_complex_terrain


def weather_terrain_params(landmark, row):
    is_brownian = True
    S = float(row["wind_speed_10m_max"]) / 2.0
    D = int(row["wind_direction_10m_dominant"] // 45)
    diffusity = float(row["cloud_cover_mean"]) / 100.0
    bias_x = int(row["precipitation_sum"] > 0.1)
    bias_y = int(landmark in (50, 80))

    return [is_brownian, S, D, diffusity, bias_x, bias_y]


if __name__ == "__main__":
    test_brownian_complex_terrain()
    # path to the study containing the animal movement data
    csv_path = 'leap_of_the_cat/The Leap of the Cat.csv'
    processor = AnimalMovementProcessor(csv_path)
    # creates landcover grid txt files
    processor.create_landcover_data_txt(resolution=200, out_directory='leap_of_the_cat')
    # your dataset containing weather data or ocean data or whatever
    df = pandas.read_csv('random_walk_package/resources/leap_of_the_cat/weather_data/CAMILA/weather_grid_y2_x0.csv')

    # must contain literally these 3 columns: latitude,longitude,timestamp (adjust to match your data)
    paths = processor.create_kernel_parameter_data_per_animal(df=df,
                                                              kernel_resolver=weather_terrain_params,
                                                              start_date=datetime(2000, 8, 24),
                                                              end_date=datetime(2000, 12, 6),
                                                              T=100,
                                                              time_stamp='timestamp',
                                                              lon='longitude',
                                                              lat='latitude')
    print(paths)
