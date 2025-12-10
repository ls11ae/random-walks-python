import pandas as pd
import planetary_computer
import requests
import rioxarray
from pyproj import Transformer
from pystac_client import Client

from random_walk_package.bindings.data_processing.movebank_parser import Coordinate_array


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


# --- 3. Fetch Movebank Animal Location Data ---
def fetch_movebank_data(study_id, sensor_type_id,
                        timestamp_start_str, timestamp_end_str,
                        bbox, output_filename="movebank_aoi.csv",
                        username="omarc98", password="Sgtb3942"):  # Add username/password for non-public
    """
    Fetches animal location data from Movebank for a given study, time, and AOI.
    Saves data as CSV in the desired format.
    Handles basic authentication if username/password are provided.
    """
    print(f"\nFetching Movebank data for Study ID: {study_id}")
    print(f"Period: {timestamp_start_str} to {timestamp_end_str}")
    print(f"BBOX: {bbox}")

    base_url = "https://www.movebank.org/movebank/service/direct-read"
    params = {
        "study_id": str(study_id),
        "sensor_type_id": str(sensor_type_id),
        "timestamp_start": timestamp_start_str,
        "timestamp_end": timestamp_end_str,
        "bbox": ",".join(map(str, bbox)),  # lon_min,lat_min,lon_max,lat_max
        "attributes": "event_id,visible,timestamp,location_long,location_lat,gps_satellite_count,ground_speed,heading,height_above_ellipsoid,tag_voltage,sensor_type,individual_taxon_canonical_name,tag_local_identifier,individual_local_identifier,study_name"
    }

    auth = None
    if username and password:
        auth = (username, password)
        print("Using Movebank authentication.")

    try:
        response = requests.get(base_url, params=params, auth=auth)

        if response.status_code == 403:  # Forbidden
            if "Login required to access study" in response.text or "User not authorized" in response.text:
                print("Error: Access to this Movebank study requires login or authorization.")
                print("Please provide your Movebank username and password if this is your study,")
                print("or ensure the study owner has granted you access.")
                return None
            else:
                response.raise_for_status()  # Raise for other 403 errors
        elif response.status_code == 400:  # Bad request
            print(f"Error: Bad request to Movebank API. Response: {response.text}")
            return None
        else:
            response.raise_for_status()

        # Movebank API returns CSV data directly
        if response.text.strip() == "" or response.text.count('\n') < 1:  # Check if empty or only header
            print("No Movebank data returned for the given criteria.")
            return None

        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Movebank data saved to {output_filename}")

        # Optionally load into pandas for quick inspection
        # try:
        #     df = pd.read_csv(output_filename)
        #     print("\nMovebank Data Snippet:")
        #     if not df.empty:
        #         print(df.head())
        #     else:
        #         print("Movebank CSV is empty after writing.")
        # except pd.errors.EmptyDataError:
        #     print("Movebank CSV is empty.")

        return output_filename

    except requests.exceptions.RequestException as e:
        print(f"Error fetching Movebank data: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
    except Exception as e:
        print(f"An unexpected error occurred with Movebank data: {e}")
    return None


# --- 3. Fetch Movebank Animal Location Data ---
def fetch_movebank_data_events(study_id, sensor_type_id,
                               timestamp_start_str, timestamp_end_str,
                               bbox=None, output_filename="movebank_locations.csv",
                               username=None, password=None):
    """
    Fetches animal location event data from Movebank for a given study, time, and AOI.
    Saves data as CSV in the desired format.
    Handles basic authentication if username/password are provided.
    """
    print(f"\nFetching Movebank EVENT data for Study ID: {study_id}")
    print(f"Period: {timestamp_start_str} to {timestamp_end_str}")
    print(f"BBOX: {bbox}")

    base_url = "https://www.movebank.org/movebank/service/direct-read"
    params = {
        "entity_type": "event",
        "study_id": str(study_id),
        "sensor_type_id": str(sensor_type_id),
        "timestamp_start": timestamp_start_str,
        "timestamp_end": timestamp_end_str,
        "attributes": "event_id,visible,timestamp,location_long,location_lat,gps_satellite_count,ground_speed,heading,height_above_ellipsoid,tag_voltage,sensor_type,individual_taxon_canonical_name,tag_local_identifier,individual_local_identifier,study_name"
    }
    if bbox:
        params["bbox"] = ",".join(map(str, bbox))

    auth = None
    if username and password:
        auth = (username, password)
        print("Using Movebank authentication.")

    try:
        response = requests.get(base_url, params=params, auth=auth)
        response.raise_for_status()

        if response.text.strip() == "" or response.text.count('\n') < 1:
            print("No Movebank data returned for the given criteria.")
            return None

        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Movebank event data saved to {output_filename}")

        try:
            df = pd.read_csv(output_filename)
            print("\nMovebank Event Data Snippet:")
            if not df.empty:
                print(df.head())
            else:
                print("Movebank Event CSV is empty after writing.")
        except pd.errors.EmptyDataError:
            print("Movebank Event CSV is empty.")

        return output_filename

    except requests.exceptions.RequestException as e:
        print(f"Error fetching Movebank event data: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
    except Exception as e:
        print(f"An unexpected error occurred with Movebank event data: {e}")
    return None


def fetch_weather_for_trajectory(coords: Coordinate_array, timestamps: list[str],
                                 output_filename: str = "weather_trajectory.csv") -> pd.DataFrame:
    """
    Fetches weather data for a trajectory of coordinates and timestamps.
    - `coords`: Your Coordinate_array (with x=longitude, y=latitude).
    - `timestamps`: List of timestamps matching the coordinates.
    """
    weather_data = []

    for i in range(coords.length):
        print("fetch t = " + str(i) + " / " + str(coords.length - 1))
        lat = coords.points[i].y
        lon = coords.points[i].x
        timestamp_str = timestamps[i]

        # Parse timestamp and format for API
        timestamp = pd.to_datetime(timestamp_str)
        date_str = timestamp.strftime("%Y-%m-%d")
        hour = timestamp.hour  # Open-Meteo uses hourly data

        # API call for this location/date
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,snowfall,weather_code,cloud_cover",
            "timezone": "UTC"
        }

        try:
            response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
            data = response.json()

            # Find matching hourly data
            for idx, time_str in enumerate(data["hourly"]["time"]):
                if pd.to_datetime(time_str).hour == hour:
                    weather_entry = {
                        "timestamp": timestamp_str,
                        "latitude": lat,
                        "longitude": lon,
                        "temperature_2m": data["hourly"]["temperature_2m"][idx],
                        "relative_humidity_2m": data["hourly"]["relative_humidity_2m"][idx],
                        "precipitation": data["hourly"]["precipitation"][idx],
                        "wind_speed_10m": data["hourly"]["wind_speed_10m"][idx],
                        "wind_direction_10m": data["hourly"]["wind_direction_10m"][idx],
                        "snowfall": data["hourly"]["snowfall"][idx],
                        "weather_code": data["hourly"]["weather_code"][idx],
                        "cloud_cover": data["hourly"]["cloud_cover"][idx]
                    }
                    weather_data.append(weather_entry)
                    break
            # time.sleep(0.1)

        except Exception as e:
            print(f"Failed for {timestamp_str} at ({lat}, {lon}): {e}")

    weather_df = pd.DataFrame(weather_data)
    weather_df.to_csv(output_filename)
    return weather_df
