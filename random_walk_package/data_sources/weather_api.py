from datetime import timedelta

import cdsapi


def fetch_era5(bbox, start_date, end_date, variables=None, filename="begona.grib"):
    if variables is None:
        variables = [
            "2m_temperature",
            "total_precipitation",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "snowfall",
        ]

    days = []
    current = start_date
    while current <= end_date:
        days.append(current)
        current += timedelta(days=1)

    years = sorted({d.year for d in days})
    months = sorted({f"{d.month:02d}" for d in days})
    dlist = sorted({f"{d.day:02d}" for d in days})

    print(years, months, dlist)

    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": years,
        "month": months,
        "day": dlist,
        "time": ["12:00"],
        "area": [bbox[3], bbox[0], bbox[1], bbox[2]],
        "data_format": "grib",
    }

    client = cdsapi.Client()
    return client.retrieve("reanalysis-era5-single-levels", request, filename)


import xarray as xr


def interpolate_weather_to_grid(nc_file, grid_lat, grid_lon):
    ds = xr.open_dataset(nc_file)
    if ds.latitude.values[0] > ds.latitude.values[-1]:
        ds = ds.sortby("latitude")

    interp = ds.interp(
        latitude=("y", grid_lat[:, 0]),
        longitude=("x", grid_lon[0, :]),
    )
    print(ds.values())
    return interp


import pandas as pd


def export_to_csv(weather_interp, landcover_grid, grid_lat, grid_lon, out="merged.csv"):
    records = []

    for t in weather_interp.valid_time.values:
        t_slice = weather_interp.sel(valid_time=t)
        for i in range(grid_lat.shape[0]):
            for j in range(grid_lat.shape[1]):
                rec = {
                    "timestamp": pd.to_datetime(str(t)),
                    "lat": float(grid_lat[i, j]),
                    "lon": float(grid_lon[i, j]),
                    "landcover": int(landcover_grid[i, j]),
                    "t2m": float(t_slice["number"][i, j]),
                    "u10": float(t_slice["latitude"][i, j]),
                    "v10": float(t_slice["longitude"][i, j]),
                    "precip": float(t_slice["expver"][i, j]),
                }
                records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(out, index=False)
    return df
