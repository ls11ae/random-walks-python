import pandas as pd
import planetary_computer
import requests
import rioxarray
from pyproj import Transformer
from pystac_client import Client


def reproject_to_utm(infile, outfile=None):
    da = rioxarray.open_rasterio(infile)

    # Get center coordinates
    bounds = da.rio.bounds()
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2

    utm_zone = int((center_lon + 180) / 6) + 1

    epsg_code = 32600 + utm_zone if center_lat >= 0 else 32700 + utm_zone

    print(f"Reprojecting to EPSG:{epsg_code} for center coordinates: {center_lon}, {center_lat}")

    # Reprojection
    da_utm = da.rio.reproject(f"EPSG:{epsg_code}")

    if outfile:
        da_utm.rio.to_raster(outfile, compress="LZW")

    return outfile if outfile else da_utm, epsg_code


def lonlat_bbox_to_utm(min_lon, min_lat, max_lon, max_lat, epsg_code):
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
    minx, miny = transformer.transform(min_lon, min_lat)
    maxx, maxy = transformer.transform(max_lon, max_lat)
    return minx, miny, maxx, maxy


def utm_bbox_to_lonlat(min_x, min_y, max_x, max_y, epsg_code):
    transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
    min_lon, min_lat = transformer.transform(min_x, min_y)
    max_lon, max_lat = transformer.transform(max_x, max_y)
    return min_lon, min_lat, max_lon, max_lat


def utm_to_lonlat(x, y, epsg_code):
    transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lon, lat


def geodetic_to_utm(lon, lat, epsg_code):
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y


# --- 1. Fetch Landcover Data (ESA WorldCover via Planetary Computer) ---
def fetch_landcover_data(bbox, output_filename="landcover_aoi.tif"):
    """
    Fetches ESA WorldCover landcover data for a given bounding box
    using Microsoft Planetary Computer's STAC API.
    Saves the result as a GeoTIFF.
    """
    print(f"\nFetching landcover data for BBOX: {bbox}...")
    try:
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        search = catalog.search(
            collections=["esa-worldcover"],
            bbox=bbox,
        )

        items = search.item_collection()
        if not items:
            print("No ESA WorldCover items found for the given AOI.")
            return None

        print(f"Found {len(items)} STAC items. Using the first one: {items[0].id}")

        # Get the href for the data asset (usually 'map')
        # Asset keys can vary; 'map' is common for ESA WorldCover
        asset_href = items[0].assets["map"].href

        # Open the raster data, clip to BBOX, and set CRS
        # ESA WorldCover is typically in EPSG:4326
        xds = rioxarray.open_rasterio(asset_href).rio.write_crs("EPSG:4326")

        # Clip to the exact bounding box
        # Ensure your bbox is in the same CRS as the raster (EPSG:4326 for WorldCover)
        # The `rio.clip_box` expects minx, miny, maxx, maxy
        clipped_xds = xds.rio.clip_box(
            minx=bbox[0],
            miny=bbox[1],
            maxx=bbox[2],
            maxy=bbox[3],
            crs="EPSG:4326"  # Specify CRS of the bbox if not already aligned
        )

        # Save the clipped raster
        clipped_xds.rio.to_raster(output_filename, compress='LZW', dtype='uint8')
        print(f"Landcover data saved to {output_filename}")

        # project to equiareal grid
        print("Reprojecting to UTM zone...")
        reproject_to_utm(output_filename, output_filename)                                    
        return output_filename

    except Exception as e:
        print(f"Error fetching landcover data: {e}")
        return None


# --- 2. Fetch Temporal Weather Data (Open-Meteo) ---
def fetch_weather_data(latitude, longitude, start_date, end_date, output_filename="weather_aoi.csv"):
    """
    Fetches hourly weather data from Open-Meteo for a given lat/lon and time period.
    Saves the result as a CSV file.
    """
    print(f"\nFetching weather data for Lat: {latitude}, Lon: {longitude}")
    print(f"Period: {start_date} to {end_date}")
    try:
        base_url = "https://archive-api.open-meteo.com/v1/archive"  # For historical data
        # For forecast, use: base_url = "https://api.open-meteo.com/v1/forecast"

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m",
            "timezone": "UTC"  # Important for consistency
        }
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)

        data = response.json()

        if 'hourly' not in data:
            print("No hourly weather data found in the response.")
            return None

        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        df.to_csv(output_filename)
        print(f"Weather data saved to {output_filename}")

        # print("\nWeather Data Snippet:")
        # print(df.head())
        return output_filename

    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with weather data: {e}")
    return None


def fetch_ocean_data(data, output_directory: str, dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"):
    
        "This function is fetching the data about northward and eastward ocean currents with the hardcoded dataset_id from copernicus. Credentials are to be loaded from env file "
        import copernicusmarine
        import os
        from dotenv import load_dotenv
        load_dotenv()
        max_lat = data["location-lat"].max()
        max_long = data["location-long"].max()
        min_lat = data["location-lat"].min()
        min_long = data["location-long"].min()
        start_date = data["timestamp"].min()
        end_data = data["timestamp"].max()
        
        get_result_annualmean = copernicusmarine.subset(
            dataset_id=dataset_id,
            output_directory=output_directory,
            username= os.getenv("COPERNICUS_USERNAME"),
            password = os.getenv("COPERNICUS_PASSWORD"),
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            minimum_longitude=min_long,
            maximum_longitude=max_long,
            start_datetime=start_date,
            end_datetime=end_data,
            variables=["time", "latitude", "longitude", "uo", "vo"]
        )
        return get_result_annualmean
    
def convert_nc_in_csv(file_path):
    import xarray as xr
    DS = xr.open_dataset(file_path)
    copernicus_marine_csv = DS.to_dataframe().to_csv()
    return copernicus_marine_csv

def build_currents_dataframe(raw_data):
    """
    Build a currents DataFrame suitable for per-step interpolation.
    
    Parameters
    ----------
    raw_data : pandas.DataFrame or list of dicts
        Must contain columns:
            - 'timestamp' (datetime or string)
            - 'longitude' (float)
            - 'latitude' (float)
            - 'uo' (float, eastward current m/s)
            - 'vo' (float, northward current m/s)
    
    Returns
    -------
    df_currents : pandas.DataFrame
        Cleaned DataFrame with sorted timestamps and numeric columns.
    """
    
    df = pd.DataFrame(raw_data)

    required_cols = ['time', 'longitude', 'latitude', 'uo', 'vo']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Convert timestamp to pandas datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Ensure numeric types
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['uo'] = pd.to_numeric(df['uo'], errors='coerce')
    df['vo'] = pd.to_numeric(df['vo'], errors='coerce')
    
    
    df = df.dropna(subset=required_cols)
    df = df.sort_values(by=['time', 'latitude', 'longitude']).reset_index(drop=True)
    
    return df  

def currents_df_to_grid(df_currents, lon_res=0.01, lat_res=0.01):
    """
    Convert currents DataFrame into a grid suitable for compute_current_offset.
    
    Parameters
    ----------
    df_currents : DataFrame
        Columns: timestamp, longitude, latitude, uo, vo
    lon_res : float
        Resolution of the longitude grid (degrees)
    lat_res : float
        Resolution of the latitude grid (degrees)
        
    Returns
    -------
    grid_x, grid_y : 1D arrays
        Longitude and latitude coordinates of the grid
    currents_u, currents_v : 2D arrays
        Eastward and northward current speeds on the grid (meters/sec)
    """
    import numpy as np
    # Create unique lon/lat arrays
    lon_min, lon_max = df_currents['longitude'].min(), df_currents['longitude'].max()
    lat_min, lat_max = df_currents['latitude'].min(), df_currents['latitude'].max()
    grid_x = np.arange(lon_min, lon_max + lon_res, lon_res)
    grid_y = np.arange(lat_min, lat_max + lat_res, lat_res)

    # Initialize grids
    currents_u = np.zeros((len(grid_y), len(grid_x)))
    currents_v = np.zeros((len(grid_y), len(grid_x)))

    # Fill grid with nearest neighbor (simplest)
    for i, y in enumerate(grid_y):
        for j, x in enumerate(grid_x):
            # Compute distance to all points
            dist2 = (df_currents['longitude'].values - x)**2 + (df_currents['latitude'].values - y)**2
            idx_min = np.argmin(dist2)
            currents_u[i, j] = df_currents['uo'].values[idx_min]
            currents_v[i, j] = df_currents['vo'].values[idx_min]

    return grid_x, grid_y, currents_u, currents_v
