# debugging: gdb --args python -m tests.test
from datetime import datetime

import pandas

from random_walk_package.core.AnimalMovement import AnimalMovementProcessor
from random_walk_package.data_sources.land_cover_adapter import load_landcover_raster, create_regular_grid, \
    resample_landcover_to_grid
from random_walk_package.data_sources.movebank_adapter import get_bounding_boxes_per_animal, bbox_to_discrete_space
from random_walk_package.data_sources.weather_api import fetch_era5


def weather_terrain_params(landmark, row):
    is_brownian = True
    S = float(row["wind_speed_10m_max"]) / 2.0
    D = int(row["wind_direction_10m_dominant"] // 45)
    diffusity = float(row["cloud_cover_mean"]) / 100.0
    bias_x = int(row["precipitation_sum"] > 0.1)
    bias_y = int(landmark in (50, 80))

    return [is_brownian, S, D, diffusity, bias_x, bias_y]


def kernel_params_combined():
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


import xarray as xr

if __name__ == "__main__":
    file = '/home/omar/PycharmProjects/random-walks-python/random_walk_package/resources/leap_of_the_cat/landcover_BEGONA_-52.4_-22.6_-52.3_-22.5.tif'
    landcover_data, meta, crs = load_landcover_raster(file)
    df = pandas.read_csv(
        '/home/omar/PycharmProjects/random-walks-python/random_walk_package/resources/leap_of_the_cat/The Leap of the Cat.csv')
    bboxes = get_bounding_boxes_per_animal(df)
    bbox = bboxes['BEGONA']
    print(bbox)
    _, _, x_res, y_res = bbox_to_discrete_space(bbox, 400)

    grid_lon, grid_lat = create_regular_grid(bbox, x_res, y_res)
    landcover_grid = resample_landcover_to_grid(landcover_data, meta, crs, grid_lon, grid_lat)
    # 1999-12-08 03:00:00.000
    start_date = datetime(1999, 12, 8)
    end_date = datetime(1999, 12, 12)
    # end_date = datetime(2001, 10, 15)

    # grib_file = '1540415201d1566730302321f5c07a33.grib'
    grib_file = fetch_era5(bbox, start_date, end_date, variables=None, filename="begona.nc")

    temp = xr.open_dataset(
        grib_file,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "2t", "step": 0}}
    )

    wind_v = xr.open_dataset(
        grib_file,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "10v", "step": 0}}
    )

    snow = xr.open_dataset(
        grib_file,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "sf", "step": 0}}
    )

    ds = xr.merge([temp, wind_v, snow], compat="override")
    df2 = ds.to_dataframe().reset_index()
    df2.to_csv("era5_export.csv", index=False)

    """
    filename = fetch_era5(bbox, start_date, end_date, variables=None, filename="begona.nc")
    print(filename)
    interp = interpolate_weather_to_grid(filename, grid_lat, grid_lon)
    export_to_csv(interp, landcover_grid, grid_lat, grid_lon, "merged.csv")
"""
