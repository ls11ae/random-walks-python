import requests
import pandas as pd
from pystac_client import Client
import planetary_computer
import rioxarray

from random_walk_package.bindings.data_processing.movebank_parser import Coordinate_array

# --- Configuration ---

# 1. Area of Interest (AOI)
# Example Bounding Box around the study reference location (adjust as needed)
AOI_BBOX_KAZAKHSTAN: tuple[float, float, float, float] = (75.0, 43.0, 77.0, 44.0)  # Wider example
AOI_BBOX: tuple[float, float, float, float] = AOI_BBOX_KAZAKHSTAN
min_lon, min_lat, max_lon, max_lat = AOI_BBOX
AOI_CENTER_LAT: float = (min_lat + max_lat) / 2.0
AOI_CENTER_LON: float = (min_lon + max_lon) / 2.0

# 2. Time Period
# For Movebank, use the entire study duration based on the provided info
MB_TIMESTAMP_START_STR = "2018-07-08 02:40:08.000"
MB_TIMESTAMP_END_STR = "2021-11-03 12:59:35.000"

OM_START_DATE = "2018-07-08"
OM_END_DATE = "2021-11-03"

# 3. Movebank Configuration
MOVEBANK_STUDY_ID = 497138661
MOVEBANK_SENSOR_TYPE_ID = 653  # GPS
MOVEBANK_USERNAME = "omarc98"  # Keep your username
MOVEBANK_PASSWORD = "Sgtb3942"  # Keep your password

# 4. Output files
LANDCOVER_OUTPUT_FILE = "../landcover_aoi.tif"
WEATHER_OUTPUT_FILE = "../weather_aoi.csv"
MOVEBANK_OUTPUT_FILE = "../movebank_aoi.csv"


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

        # For further processing in Python, you can return the xarray.DataArray
        # print("Landcover data snippet (shape):", clipped_xds.shape)
        # print(clipped_xds)
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


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Data Fetching Script ---")

    # 1. Landcover Data
    # Note: This step requires internet and can take a moment.
    # Ensure the AOI is not excessively large for a quick test.
    landcover_file = fetch_landcover_data(AOI_BBOX, LANDCOVER_OUTPUT_FILE)
    if landcover_file:
        print(f"Successfully fetched landcover. Check: {landcover_file}")
        # You can add code here to open and inspect with rasterio if needed
        # e.g., src = rasterio.open(landcover_file)
        # print("Landcover CRS:", src.crs)
        # print("Landcover bounds:", src.bounds)
        # data_sample = src.read(1, window=((0,10),(0,10))) # read a small window
        # print("Landcover data sample (top-left 10x10 pixels):\n", data_sample)
    else:
        print("Landcover fetching failed.")

    # 2. Weather Data
    weather_file = fetch_weather_data(AOI_CENTER_LAT, AOI_CENTER_LON,
                                      OM_START_DATE, OM_END_DATE, WEATHER_OUTPUT_FILE)
    if weather_file:
        print(f"Successfully fetched weather data. Check: {weather_file}")
        # df_weather = pd.read_csv(weather_file)
        # print("Weather data sample:\n", df_weather.head())
    else:
        print("Weather fetching failed.")

    # 3. Movebank Data
    # Using public data, so no username/password needed for this specific study
    movebank_file = fetch_movebank_data(MOVEBANK_STUDY_ID, MOVEBANK_SENSOR_TYPE_ID,
                                        MB_TIMESTAMP_START_STR, MB_TIMESTAMP_END_STR,
                                        AOI_BBOX, MOVEBANK_OUTPUT_FILE)
    # username=MOVEBANK_USERNAME, password=MOVEBANK_PASSWORD) # Uncomment for your own data
    if movebank_file:
        MOVEBANK_STUDY_ID = 497138661
        MOVEBANK_SENSOR_TYPE_ID = 653  # GPS (likely)
        MB_TIMESTAMP_START_STR = "2018-07-08 02:40:08.000"
        MB_TIMESTAMP_END_STR = "2021-11-03 12:59:35.000"
        # Example Bounding Box (adjust as needed or set to None for the entire study area)
        AOI_BBOX_KAZAKHSTAN = [70.0, 40.0, 80.0, 45.0]
        # AOI_BBOX_KAZAKHSTAN = None # To get all data

        fetched_file = fetch_movebank_data_events(MOVEBANK_STUDY_ID, MOVEBANK_SENSOR_TYPE_ID,
                                                  MB_TIMESTAMP_START_STR, MB_TIMESTAMP_END_STR,
                                                  bbox=AOI_BBOX_KAZAKHSTAN,
                                                  username=MOVEBANK_USERNAME, password=MOVEBANK_PASSWORD)

        if fetched_file:
            print(f"\nEvent data successfully fetched and saved to: {fetched_file}")


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
