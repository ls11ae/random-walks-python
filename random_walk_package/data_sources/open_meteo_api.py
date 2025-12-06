import time

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


def weather_tuples(weather_data):
    """Generate weather tuples grouped by movement steps"""
    if not weather_data:  # Check if it's None or empty
        # Changed condition from hasattr to direct check for None or empty list
        raise ValueError(
            "Weather data not loaded or generated. Call create_movement_data() and then fetch_trajectory_weather() or load_weather_data() first.")

    sampled_weather = []
    for i in range(len(weather_data)):
        entry = weather_data[i]
        sampled_weather.append((
            entry[0],  # timestamp
            entry[1],  # latitude
            entry[2],  # longitude
            entry[3],  # temperature_2m
            entry[4],  # relative_humidity_2m
            entry[5],  # precipitation
            entry[6],  # wind_speed_10m
            entry[7],  # wind_direction_10m
            entry[8],  # snowfall
            entry[9],  # weather_code
            entry[10]  # cloud_cover
        ))
    return sampled_weather
