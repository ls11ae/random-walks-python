import ctypes

from random_walk_package.bindings.data_structures.point2D import *
from random_walk_package.bindings.data_structures.types import *

from datetime import datetime


def timed_location_of(x, y, time_stamp):
    dt_s = datetime.strptime(time_stamp, "%Y-%m-%d %H:%M:%S.%f")
    t_loc = TimedLocation(
        time=DateTime(dt_s.year, dt_s.month, dt_s.day, dt_s.hour),
        location=Point2D(x, y)
    )
    return t_loc
