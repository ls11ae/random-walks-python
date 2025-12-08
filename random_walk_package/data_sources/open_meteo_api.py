import os
import time

import numpy as np
import pandas as pd
import requests


def _fetch_single_weather(lat: float, lon: float, timestamp_str: str) -> dict | None:
    """Helper for individual weather fetches with retry logic for a specific hour."""
    timestamp = pd.to_datetime(timestamp_str)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": timestamp.strftime("%Y-%m-%d"),
        "end_date": timestamp.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,snowfall,weather_code,cloud_cover",
        "timezone": "UTC"
    }

    for attempt in range(3):
        try:
            response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
            response.raise_for_status()  # Raise HTTPError for bad responses (4XX or 5XX)
            data = response.json()

            hourly_data = data.get("hourly", {})
            times = hourly_data.get("time", [])

            if not times:
                print(
                    f"Warning: No hourly data returned for {lat}, {lon} on {timestamp.strftime('%Y-%m-%d')} in API response: {data}")
                return None  # Or handle as an error

            target_hour = timestamp.hour
            for idx, time_str_from_api in enumerate(hourly_data["time"]):
                if pd.to_datetime(time_str_from_api).hour == target_hour:
                    entry_data = {
                        "timestamp": timestamp_str,  # Original requested timestamp
                        "latitude": lat,
                        "longitude": lon
                    }
                    for k in ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m',
                              'wind_direction_10m', 'snowfall', 'weather_code', 'cloud_cover']:
                        value_list = hourly_data.get(k)
                        if value_list is not None and idx < len(value_list):
                            entry_data[k] = value_list[idx]
                        else:
                            entry_data[k] = 0.0  # Handle missing data for a variable
                    return entry_data

            print(f"Warning: Target hour {target_hour} not found in API response for {timestamp_str}.")
            return None

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} (RequestException) failed for {timestamp_str}: {str(e)}")
            time.sleep(2 ** attempt)
        except Exception as e:  # Catch other potential errors like JSON parsing
            print(f"Attempt {attempt + 1} (Exception) failed for {timestamp_str}: {str(e)}")
            time.sleep(2 ** attempt)

    print(f"Failed to fetch weather for {timestamp_str} at {lat},{lon} after multiple attempts.")
    return None


def _fetch_hourly_data_for_period_at_point(lat: float, lon: float, start_date_str: str, end_date_str: str,
                                           fetch_hourly=False) -> list[dict]:
    """
    Helper for fetching all hourly weather data for a date range at a specific lat/lon.
    Returns a list of dictionaries, each dictionary representing one hourly record.
    """

    if fetch_hourly:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,snowfall,weather_code,cloud_cover",
            "timezone": "UTC"
        }
    else:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "daily": "weather_code,temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,snowfall_sum,wind_speed_10m_max,wind_direction_10m_dominant,cloud_cover_mean",
            "timezone": "UTC"
        }

    weather_records_for_point = []

    for attempt in range(3):  # Retry logic
        try:
            response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
            response.raise_for_status()
            data = response.json()

            hourly_data = data.get("hourly" if fetch_hourly else "daily", {})
            times = hourly_data.get("time", [])

            if not times:
                print(
                    f"Warning: No data returned for Lat: {lat:.4f}, Lon: {lon:.4f} between {start_date_str} and {end_date_str}. API Response: {data.get('reason', '')}")
                return []  # Return empty list if no time entries

            # Prepare series data, ensuring all expected keys exist and have correct length
            series_data = {}
            if fetch_hourly:
                expected_vars = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m',
                                 'wind_direction_10m', 'snowfall', 'weather_code', 'cloud_cover']
            else:
                # Fixed: Use the same variable names as in the API request
                expected_vars = ['weather_code', 'temperature_2m_mean', 'relative_humidity_2m_mean',
                                 'precipitation_sum', 'snowfall_sum', 'wind_speed_10m_max',
                                 'wind_direction_10m_dominant', 'cloud_cover_mean']

            num_timestamps = len(times)

            for var_name in expected_vars:
                raw_var_data = hourly_data.get(var_name, [None] * num_timestamps)
                # Ensure the list has the same length as 'times'
                if len(raw_var_data) < num_timestamps:
                    print(
                        f"Warning: Data for '{var_name}' at {lat},{lon} is shorter than time series. Padding with None.")
                    raw_var_data.extend([None] * (num_timestamps - len(raw_var_data)))
                elif len(raw_var_data) > num_timestamps:
                    print(f"Warning: Data for '{var_name}' at {lat},{lon} is longer than time series. Truncating.")
                    raw_var_data = raw_var_data[:num_timestamps]
                series_data[var_name] = raw_var_data

            for i, time_str in enumerate(times):
                if fetch_hourly:
                    record = {
                        "latitude": lat,
                        "longitude": lon,
                        "timestamp": time_str,
                        "temperature_2m": series_data['temperature_2m'][i],
                        "relative_humidity_2m": series_data['relative_humidity_2m'][i],
                        "precipitation": series_data['precipitation'][i],
                        "wind_speed_10m": series_data['wind_speed_10m'][i],
                        "wind_direction_10m": series_data['wind_direction_10m'][i],
                        "snowfall": series_data['snowfall'][i],
                        "weather_code": series_data['weather_code'][i],
                        "cloud_cover": series_data['cloud_cover'][i],
                    }
                else:
                    record = {
                        "latitude": lat,
                        "longitude": lon,
                        "timestamp": time_str,
                        "weather_code": series_data['weather_code'][i],
                        "temperature_2m_mean": series_data['temperature_2m_mean'][i],
                        "relative_humidity_2m_mean": series_data['relative_humidity_2m_mean'][i],
                        "precipitation_sum": series_data['precipitation_sum'][i],
                        "snowfall_sum": series_data['snowfall_sum'][i],
                        "wind_direction_10m_dominant": series_data['wind_direction_10m_dominant'][i],
                        "wind_speed_10m_max": series_data['wind_speed_10m_max'][i],  # Fixed: use max instead of mean
                        "cloud_cover_mean": series_data['cloud_cover_mean'][i],
                    }
                weather_records_for_point.append(record)
            return weather_records_for_point  # Success

        except requests.exceptions.HTTPError as e:
            print(
                f"Attempt {attempt + 1} (HTTPError: {e.response.status_code}) for {lat},{lon} ({start_date_str}-{end_date_str}): {e.response.text}")
            if e.response.status_code == 400:  # Bad request, likely won't succeed on retry
                print("Bad request, not retrying for this point.")
                break
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            print(
                f"Attempt {attempt + 1} (RequestException) for {lat},{lon} ({start_date_str}-{end_date_str}): {str(e)}")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(
                f"Attempt {attempt + 1} (Unexpected Error) for {lat},{lon} ({start_date_str}-{end_date_str}): {str(e)}")
            time.sleep(2 ** attempt)

    print(f"Failed to fetch weather data for Lat: {lat:.4f}, Lon: {lon:.4f} after multiple attempts.")
    return []  # Return empty list on failure after retries


def create_weather_csvs(bbox, interval, animal_id,
                        animal_dir, grid_points_per_edge=10, fetch_hourly=False, merged_csv_path=None,
                        results_map: dict[str, str] = None):
    start_date, end_date = interval
    min_lon, min_lat, max_lon, max_lat = bbox
    # Otherwise, generate grid coordinates and fetch
    if max_lon == min_lon or max_lat == min_lat:
        lon_coords = np.array([min_lon])
        lat_coords = np.array([min_lat])
        if grid_points_per_edge > 1:
            print(f"Warning: BBox for animal {animal_id} is point/line. Using 1 grid point.")
    else:
        lon_coords = np.linspace(min_lon + (max_lon - min_lon) / (2 * grid_points_per_edge),
                                 max_lon - (max_lon - min_lon) / (2 * grid_points_per_edge),
                                 num=grid_points_per_edge, endpoint=True)
        lat_coords = np.linspace(min_lat + (max_lat - min_lat) / (2 * grid_points_per_edge),
                                 max_lat - (max_lat - min_lat) / (2 * grid_points_per_edge),
                                 num=grid_points_per_edge, endpoint=True)

    grid_points_to_query = []
    for lat_val in lat_coords:
        for lon_val in lon_coords:
            grid_points_to_query.append((lat_val, lon_val))
    print(f"[{animal_id}] Generated {len(grid_points_to_query)} grid points for weather fetching.")

    total_points_to_fetch = len(grid_points_to_query)

    for i, (lat, lon) in enumerate(grid_points_to_query):
        print(
            f"[{animal_id}] Fetching weather for grid point {i + 1}/{total_points_to_fetch} (Lat: {lat:.4f}, Lon: {lon:.4f})")
        point_weather_data_list = _fetch_hourly_data_for_period_at_point(
            lat, lon, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), fetch_hourly
        )
        # Save per-grid-point CSV
        y_idx = i // grid_points_per_edge
        x_idx = i % grid_points_per_edge
        csv_name = f"weather_grid_y{y_idx}_x{x_idx}.csv"
        csv_path = os.path.join(animal_dir, csv_name)
        df_point = pd.DataFrame(point_weather_data_list)
        columns_order = (['latitude', 'longitude', 'timestamp', 'temperature_2m', 'relative_humidity_2m',
                          'precipitation', 'wind_speed_10m', 'wind_direction_10m', 'snowfall', 'weather_code',
                          'cloud_cover'] if fetch_hourly else
                         ['latitude', 'longitude', 'timestamp', 'temperature_2m_mean',
                          'relative_humidity_2m_mean',
                          'precipitation_sum', 'wind_speed_10m_max', 'wind_direction_10m_dominant',
                          'snowfall_sum',
                          'weather_code', 'cloud_cover_mean'])

        df_point = df_point.reindex(columns=columns_order)
        df_point.to_csv(csv_path, index=False)
        print(f"[{animal_id}] Saved grid point CSV: {csv_path}")
        results_map[str(animal_id)] = animal_dir

        if i < total_points_to_fetch - 1:
            time.sleep(0.2)

        results_map[str(animal_id)] = merged_csv_path
