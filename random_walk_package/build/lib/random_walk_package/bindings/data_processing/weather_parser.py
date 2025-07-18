import ctypes

import pandas as pd

import requests

from random_walk_package.bindings.data_processing.movebank_parser import Coordinate_array
from random_walk_package.bindings.data_structures.point2D import *
from random_walk_package.bindings.data_structures.types import *


dll.weather_entry_new.argtypes = [ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                                  ctypes.c_float, ctypes.c_int, ctypes.c_int]
dll.weather_entry_new.restype = ctypes.POINTER(WeatherEntry)

dll.weather_timeline_new.argtypes = [ctypes.c_size_t]
dll.weather_timeline_new.restype = WeatherTimelinePtr

dll.weather_grid_new.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
dll.weather_grid_new.restype = WeatherGridPtr

dll.weather_entry_print.argtypes = [WeatherEntryPtr]
dll.weather_entry_print.restype = None

dll.weather_timeline_print.argtypes = [WeatherTimelinePtr]
dll.weather_timeline_print.restype = None

dll.weather_grid_print.argtypes = [WeatherGridPtr]
dll.weather_grid_print.restype = None

dll.weather_entry_free.argtypes = [WeatherEntryPtr]
dll.weather_entry_free.restype = None


dll.weather_entry_to_bias.argtypes = [WeatherEntryPtr]
dll.weather_entry_to_bias.restype = Point2DPtr

def weather_entry_to_bias(entry: WeatherEntryPtr) -> Point2DPtr: # type: ignore
    return dll.weather_entry_to_bias(entry)


def weather_timeline(length):
    return dll.weather_timeline_new(length)


def weather_entry_free(entry: WeatherEntryPtr): # type: ignore
    return dll.weather_entry_free(entry)


def weather_grid(height, width):
    return dll.weather_grid_new(height, width)


def weather_entry_new(temperature, humidity, precipitation, wind_speed, wind_direction, snow_fall, weather_code,
                      cloud_cover):
    return dll.weather_entry_new(temperature, humidity, precipitation, wind_speed, wind_direction, snow_fall,
                                 weather_code, cloud_cover)


def weather_grid_print(grid):
    return dll.weather_grid_print(grid)
