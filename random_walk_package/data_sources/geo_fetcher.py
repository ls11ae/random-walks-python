"""Compatibility imports for environmental data helpers.

The implementations live in :mod:`environmentcma` so they can be used
without importing the random-walk compiled extension.
"""

from environmentcma.crs import (
    geodetic_to_utm,
    lonlat_bbox_to_utm,
    reproject_to_utm,
    utm_bbox_to_lonlat,
    utm_to_lonlat,
)
from environmentcma.currents import (
    build_currents_dataframe,
    convert_nc_in_csv,
    currents_df_to_grid,
    fetch_ocean_data,
)
from environmentcma.landcover import fetch_landcover_data
from environmentcma.weather import fetch_weather_data

__all__ = [
    "build_currents_dataframe",
    "convert_nc_in_csv",
    "currents_df_to_grid",
    "fetch_landcover_data",
    "fetch_ocean_data",
    "fetch_weather_data",
    "geodetic_to_utm",
    "lonlat_bbox_to_utm",
    "reproject_to_utm",
    "utm_bbox_to_lonlat",
    "utm_to_lonlat",
]
