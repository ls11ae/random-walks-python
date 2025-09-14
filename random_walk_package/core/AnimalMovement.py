import os
import time

import numpy as np

# Assuming these imports are correctly resolved from your project structure
from random_walk_package.bindings.data_processing.movebank_parser import *
from random_walk_package.bindings.data_processing.weather_parser import weather_entry_new, \
    weather_timeline
from random_walk_package.data_sources.geo_fetcher import *
from random_walk_package.data_sources.land_cover_adapter import landcover_to_discrete_txt
from random_walk_package.data_sources.movebank_adapter import get_start_end_dates, \
    get_unique_animal_ids, get_animal_coordinates, get_bounding_boxes_per_animal, \
    bbox_dict_to_discrete_space


# landcover_to_discrete_txt is imported in the original but not used, can be kept or removed.

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


def _fetch_hourly_data_for_period_at_point(lat: float, lon: float, start_date_str: str, end_date_str: str) -> \
        list[dict]:
    """
    Helper for fetching all hourly weather data for a date range at a specific lat/lon.
    Returns a list of dictionaries, each dictionary representing one hourly record.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,snowfall,weather_code,cloud_cover",
        "timezone": "UTC"
    }
    weather_records_for_point = []

    for attempt in range(3):  # Retry logic
        try:
            response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
            response.raise_for_status()
            data = response.json()

            hourly_data = data.get("hourly", {})
            times = hourly_data.get("time", [])

            if not times:
                print(
                    f"Warning: No hourly data returned for Lat: {lat:.4f}, Lon: {lon:.4f} between {start_date_str} and {end_date_str}. API Response: {data.get('reason', '')}")
                return []  # Return empty list if no time entries

            # Prepare series data, ensuring all expected keys exist and have correct length
            series_data = {}
            expected_vars = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m',
                             'wind_direction_10m', 'snowfall', 'weather_code', 'cloud_cover']
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


class AnimalMovementProcessor:
    def __init__(self, data_file):
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        base_project_dir = os.path.join(self.script_dir, '..')  # Adjust if script is not in a subdir of project root
        self.resources_dir = os.path.join(base_project_dir, 'resources')

        # Ensure data_file path is constructed correctly relative to resources_dir
        if not os.path.isabs(data_file):
            self.data_file = os.path.join(self.resources_dir, data_file)
        else:
            self.data_file = data_file

        self.df = None
        self.bbox: dict[str, tuple[float, float, float, float]] = {}
        self.bbox_utm: dict[str, tuple[float, float, float, float]] = {}
        self.aid_espg_map: dict[str, str] = {}
        self.discrete_params = None
        self.center_lat = None
        self.center_lon = None
        self.timeline = None
        self.grid_coords = None
        self.geo_coords = None
        self._weather_data = None  # For trajectory weather
        # self._gridded_weather_data = None # Optionally store gridded weather if needed

        # Initialize common resources
        self._load_data()
        self._compute_bbox()

    def _load_data(self):
        if self.df is None:
            try:
                self.df = pd.read_csv(self.data_file)
            except FileNotFoundError:
                print(f"Error: Data file not found at {self.data_file}")
                # Potentially raise an error or handle appropriately
                raise

    def _compute_bbox(self):
        """Compute bounding box of interest with 10% padding on each side."""
        print("Computing bounding boxes ...")
        self.bbox = get_bounding_boxes_per_animal(self.df)

    def _compute_discrete_params(self, resolution=200):
        """Convert lon lat bbox to array coordinates bb"""
        if self.discrete_params is None and self.bbox:
            self.discrete_params = bbox_dict_to_discrete_space(self.bbox, resolution)

    def bbox_utm_of(self, animal_id: str):
        return self.bbox_utm.get(str(animal_id))

    def create_landcover_data_txt(self, resolution=200, out_directory=None) -> dict[str, str]:
        """Generate per-animal landcover data (TIFF + TXT), named with animal_id and bbox.

        Returns:
            dict[str, str]: { animal_id: txt_path }
        """
        if not self.bbox:
            print("Error: Bounding boxes not computed. Load data first.")
            return {}

        # Compute per-animal discrete params once
        self._compute_discrete_params(resolution)
        if not self.discrete_params:
            print("Error: Discrete parameters not computed.")
            return {}

        # Target directory for outputs
        if out_directory:
            target_dir = out_directory
        else:
            target_dir = self.resources_dir
        os.makedirs(target_dir, exist_ok=True)

        results: dict[str, str] = {}
        # Expecting self.bbox as {animal_id: (min_lon, min_lat, max_lon, max_lat)}
        for animal_id, bbox in self.bbox.items():
            min_lon, min_lat, max_lon, max_lat = bbox
            base = f"landcover_{animal_id}_{min_lon:.1f}_{min_lat:.1f}_{max_lon:.1f}_{max_lat:.1f}"
            tif_path = os.path.join(target_dir, f"{base}.tif")
            txt_path = os.path.join(target_dir, f"{base}_{resolution}.txt")

            # Fetch TIFF if needed
            if not os.path.exists(tif_path):
                print(f"Fetching landcover data to {tif_path}")
                fetch_landcover_data(bbox, tif_path)
            else:
                print(f"TIFF file already exists at {tif_path}, skipping fetch.")

            # Look up per-animal discrete resolution; expected (0, 0, x_res, y_res)
            if isinstance(self.discrete_params, dict):
                dp = self.discrete_params.get(str(animal_id)) or self.discrete_params.get(animal_id)
            else:
                dp = None
            if not dp:
                print(f"Warning: No discrete params for animal_id={animal_id}. Skipping.")
                continue
            _, _, x_res, y_res = dp

            # Always (re)write TXT from TIFF for consistency

            espg_code, bbox_utm = landcover_to_discrete_txt(
                tif_path,
                x_res,  # width_discrete
                y_res,  # height_discrete
                min_lon, max_lat, max_lon, min_lat,
                txt_path
            )
            self.aid_espg_map[str(animal_id)] = str(espg_code)
            self.bbox_utm[str(animal_id)] = bbox_utm
            print(f"Wrote landcover TXT: {txt_path}")
            results[str(animal_id)] = txt_path

        return results

    def create_weather_tuples(self) -> list[tuple]:
        """Generate weather tuples grouped by movement steps"""
        if not self._weather_data:  # Check if it's None or empty
            # Changed condition from hasattr to direct check for None or empty list
            raise ValueError(
                "Weather data not loaded or generated. Call create_movement_data() and then fetch_trajectory_weather() or load_weather_data() first.")

        sampled_weather = []
        for i in range(len(self._weather_data)):
            entry = self._weather_data[i]
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

    def create_movement_data(self, samples=10) -> tuple[
        dict[str, list[tuple[int, int]]], dict[str, list[tuple[int, int]]]]:
        if self.df is None:
            self._load_data()

        utm_coords_by_animal: dict[str, list[tuple[int, int]]] = {}
        geo_coords_by_animal: dict[str, list[tuple[int, int]]] = {}
        animal_ids = get_unique_animal_ids(self.df)

        for aid in animal_ids:
            # Resolve this animal's UTM bbox
            bbox_utm = None
            if isinstance(self.bbox_utm, dict):
                bbox_utm = self.bbox_utm.get(aid) or self.bbox_utm.get(str(aid))
            if bbox_utm is None:
                print(f"No UTM bbox found for animal {aid}")
                continue

            # Resolve width/height from per-animal discrete params
            x_res = 100
            y_res = 100
            if isinstance(self.discrete_params, dict):
                dp = self.discrete_params.get(aid) or self.discrete_params.get(str(aid))
                if dp and len(dp) == 4:
                    _, _, x_res, y_res = dp

            # Get coordinates for current animal ID using its UTM bbox
            coords_data, geo_coords = get_animal_coordinates(
                df=self.df,
                animal_id=aid,
                epsg_code=self.aid_espg_map[str(aid)],
                samples=samples,
                width=x_res,
                height=y_res,
                bbox_utm=bbox_utm
            )
            utm_coords_by_animal[str(aid)] = coords_data
            geo_coords_by_animal[str(aid)] = geo_coords

        self.grid_coords = utm_coords_by_animal
        self.geo_coords = geo_coords_by_animal
        return utm_coords_by_animal, geo_coords_by_animal

    def grid_coordinates_to_geodetic(self, coord: list[tuple[int, int]], animal_id: str) -> list[tuple[float, float]]:
        """
        Convert grid coordinates (x, y) back to lon/lat (WGS84).

        Parameters
        ----------
        coord : list of (int, int)
            Grid coordinates (x, y).
        animal_id : str
            Animal identifier.

        Returns
        -------
        list of (lon, lat)
            Geographic coordinates (WGS84).
        """

        result: list[tuple[float, float]] = []

        epsg_code = self.aid_espg_map.get(str(animal_id))
        if epsg_code is None:
            raise ValueError(f"No EPSG code for animal {animal_id}")

        bbox_utm = self.bbox_utm.get(str(animal_id))
        if bbox_utm is None:
            raise ValueError(f"No bbox_utm for animal {animal_id}")

        _, _, width, height = self.discrete_params.get(str(animal_id))
        if width is None or height is None:
            raise ValueError(f"No discrete params for animal {animal_id}")

        min_x, min_y, max_x, max_y = bbox_utm

        for x, y in coord:
            # Grid → UTM
            utm_x = min_x + (x / (width - 1)) * (max_x - min_x)
            utm_y = max_y - (y / (height - 1)) * (max_y - min_y)

            # UTM → Geodetic
            lon, lat = utm_to_lonlat(utm_x, utm_y, epsg_code)
            result.append((lat, lon))

        return result

    def fetch_trajectory_weather(self, output_filename="weather_trajectory.csv") -> list[tuple]:
        """Fetch weather for pre-loaded trajectory coordinates and timestamps"""
        if not self.grid_coords or not self.timeline:
            raise ValueError("Call create_movement_data() first to generate coordinates and timeline.")

        # Assuming self.grid_coords has a .points list/array and .length or similar
        # And self.timeline is a list of timestamp strings
        num_points = 0
        if hasattr(self.grid_coords, 'points') and self.grid_coords.points is not None:  # For Point2DArray from C
            num_points = self.grid_coords.length if hasattr(self.grid_coords, 'length') else len(
                self.grid_coords.points)
        elif hasattr(self.grid_coords, 'coordinates'):  # For a simple list of point objects
            num_points = len(self.grid_coords.coordinates)

        if num_points != len(self.timeline):
            raise ValueError(
                f"Mismatch between number of coordinates ({num_points}) and timeline entries ({len(self.timeline)}).")

        weather_data_collected = []  # Stores dicts first

        for i in range(num_points):
            print(f"Fetching trajectory weather {i + 1}/{num_points}")
            # Adjust access to lat/lon based on self.grid_coords structure
            if hasattr(self.grid_coords, 'points') and self.grid_coords.points is not None:  # C-style Point2DArray
                lat = self.grid_coords.points[i].y
                lon = self.grid_coords.points[i].x
            elif hasattr(self.grid_coords, 'coordinates'):  # Python list of point objects
                lat = self.grid_coords.coordinates[i].y  # Assuming .y attribute
                lon = self.grid_coords.coordinates[i].x  # Assuming .x attribute
            else:
                raise AttributeError("Coordinates structure not recognized in self.grid_coords")

            timestamp_str = self.timeline[i]
            entry_dict = _fetch_single_weather(lat, lon, timestamp_str)
            if entry_dict:
                weather_data_collected.append(entry_dict)
            time.sleep(0.15)

        if not weather_data_collected:
            print("No trajectory weather data was fetched.")
            self._weather_data = []  # Ensure it's an empty list
            return []

        # Convert list of dicts to list of tuples for self._weather_data and DataFrame
        weather_data_tuples = []
        for entry in weather_data_collected:
            weather_data_tuples.append((
                entry['timestamp'],
                entry['latitude'],
                entry['longitude'],
                entry.get('temperature_2m'),
                entry.get('relative_humidity_2m'),
                entry.get('precipitation'),
                entry.get('wind_speed_10m'),
                entry.get('wind_direction_10m'),
                entry.get('snowfall'),
                entry.get('weather_code'),
                entry.get('cloud_cover')
            ))

        self._weather_data = weather_data_tuples

        df_columns = [
            'timestamp', 'latitude', 'longitude', 'temperature_2m',
            'relative_humidity_2m', 'precipitation', 'wind_speed_10m',
            'wind_direction_10m', 'snowfall', 'weather_code', 'cloud_cover'
        ]
        pd.DataFrame(weather_data_tuples, columns=df_columns).to_csv(output_filename, index=False)
        print(f"Trajectory weather data saved to {output_filename}")
        return weather_data_tuples

    def load_weather_data(self, input_filename="weather_trajectory.csv") -> None:
        """Load previously fetched weather data from CSV"""
        filepath = input_filename
        if not os.path.isabs(input_filename):
            # Assuming it might be in resources or current dir.
            # For consistency with how data_file is handled, or how output_filename in fetch_trajectory_weather is handled.
            # Let's assume it's a direct path or relative to CWD for now.
            pass

        try:
            df = pd.read_csv(filepath)
            # Convert DataFrame to list of tuples in expected order
            self._weather_data = [
                (
                    row['timestamp'],
                    float(row['latitude']),
                    float(row['longitude']),
                    float(row['temperature_2m']) if pd.notna(row['temperature_2m']) else None,
                    int(row['relative_humidity_2m']) if pd.notna(row['relative_humidity_2m']) else None,
                    float(row['precipitation']) if pd.notna(row['precipitation']) else None,
                    float(row['wind_speed_10m']) if pd.notna(row['wind_speed_10m']) else None,
                    int(row['wind_direction_10m']) if pd.notna(row['wind_direction_10m']) else None,
                    float(row['snowfall']) if pd.notna(row['snowfall']) else None,
                    int(row['weather_code']) if pd.notna(row['weather_code']) else None,
                    int(row['cloud_cover']) if pd.notna(row['cloud_cover']) else None
                ) for _, row in df.iterrows()
            ]
            print(f"Loaded {len(self._weather_data)} weather records from {filepath}")
        except FileNotFoundError:
            print(f"Warning: Weather file {filepath} not found. No data loaded.")
            self._weather_data = None  # Or []
        except KeyError as e:
            print(f"Error: Missing required column in weather file {filepath} - {str(e)}")
            self._weather_data = None  # Or []
        except Exception as e:
            print(f"Error loading weather data from {filepath}: {str(e)}")
            self._weather_data = None  # Or []

    def create_weather_data(self, output_file="weather_baboons.csv"):
        """Fetches trajectory weather using start/end dates from the entire dataset.
        Note: This might be confusing. fetch_trajectory_weather fetches for specific points.
        This method name suggests creating a weather dataset, perhaps gridded or for the whole timeframe.
        The original implementation calls fetch_trajectory_weather.
        This will fetch weather for the *currently loaded trajectory* in self.grid_coords and self.timeline.
        """
        if self.df is None or self.df.empty:
            raise ValueError("DataFrame not loaded. Cannot determine date range.")
        if self.grid_coords is None or self.timeline is None:
            raise ValueError("Movement data (coords and timeline) not created. Call create_movement_data() first.")

        start_date_df, end_date_df = get_start_end_dates(self.df)  # These are from the whole dataset
        print(f"Dataset date range: OM_START_DATE = {start_date_df}, OM_END_DATE = {end_date_df}")
        print(f"Fetching weather for the current trajectory within this general timeframe.")

        # output_file path handling
        output_path = output_file
        if not os.path.isabs(output_file):
            output_path = os.path.join(self.resources_dir, output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        return self.fetch_trajectory_weather(output_path)

    def create_weather_tuples_ctypes(self) -> POINTER(WeatherTimeline):  # type: ignore
        """Convert trajectory weather data to C-compatible WeatherTimeline structure"""
        if not self._weather_data:  # Check if it's None or empty
            raise ValueError(
                "Weather data not available. Call fetch_trajectory_weather() or load_weather_data() first.")

        num_entries = len(self._weather_data)
        if num_entries == 0:
            # Return an empty or minimally initialized WeatherTimeline
            # Assuming weather_timeline can handle num_entries = 0
            return weather_timeline(0)

        # Create array of WeatherEntry structures
        # Assuming weather_timeline creates the timeline structure and allocates memory for entries
        c_timeline = weather_timeline(num_entries)  # capacity, size

        for i, data_tuple in enumerate(self._weather_data):
            # data_tuple: (timestamp, lat, lon, temp, hum, precip, wind_s, wind_d, snow, code, cloud)
            # weather_entry_new expects: temp, hum, precip, wind_s, wind_d, snow, code, cloud
            # Need to handle None values if C structure cannot take them (e.g. use a default like -9999 or 0)

            def nan_to_default(val, default=0):
                return default if val is None or pd.isna(val) else val

            entry = weather_entry_new(
                float(nan_to_default(data_tuple[3], -9999)),  # temperature_2m
                int(nan_to_default(data_tuple[4], -1)),  # relative_humidity_2m
                float(nan_to_default(data_tuple[5], -1)),  # precipitation
                float(nan_to_default(data_tuple[6], -1)),  # wind_speed_10m
                int(nan_to_default(data_tuple[7], -1)),  # wind_direction_10m
                float(nan_to_default(data_tuple[8], -1)),  # snowfall
                int(nan_to_default(data_tuple[9], -1)),  # weather_code
                int(nan_to_default(data_tuple[10], -1))  # cloud_cover
            )
            c_timeline.contents.entries[i] = entry
        return c_timeline

    # --- New methods for gridded weather data ---

    def fetch_gridded_weather_data(self, output_folder: str,
                                   start_date_override: str = None, days_to_fetch: int = 7,
                                   grid_points_per_edge: int = 10) -> str:
        """
        Fetches and saves hourly weather data for a grid of equidistant points within the bounding box.
        If the output_folder exists and contains the expected number of CSVs, does nothing.
        Otherwise, fetches and saves the data as CSVs in output_folder.

        Args:
            output_folder (str): Folder where the grid CSVs should be stored.
            start_date_override (str, optional): 'YYYY-MM-DD' string to override the start date.
            days_to_fetch (int): Number of days for hourly weather data.
            grid_points_per_edge (int): Number of points along each edge of the bounding box.

        Returns:
            str: Path to the folder where the grid CSVs are stored.
        """
        if self.bbox is None:
            self._compute_bbox()
            if self.bbox is None:
                raise ValueError("Bounding box (self.bbox) is not set. Load data first.")

        if self.df is None or self.df.empty:
            if not start_date_override:
                raise ValueError(
                    "Animal movement data (self.df) is not loaded. Needed to determine start date, or provide 'start_date_override'.")
        if start_date_override:
            try:
                start_date = pd.to_datetime(start_date_override)
            except ValueError:
                raise ValueError("Invalid format for 'start_date_override'. Please use 'YYYY-MM-DD'.")
        elif self.df is not None and not self.df.empty:
            min_date_str, _ = get_start_end_dates(self.df)
            if not min_date_str:
                raise ValueError(
                    "Could not determine start date from animal movement data. Provide 'start_date_override'.")
            start_date = pd.to_datetime(min_date_str)
        else:
            raise ValueError(
                "Cannot determine start date: no animal data loaded and no 'start_date_override' provided.")
        end_date = start_date + pd.Timedelta(days=days_to_fetch - 1)
        print(f"Fetching gridded weather from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")

        min_lon, min_lat, max_lon, max_lat = self.bbox

        output_dir = output_folder
        os.makedirs(output_dir, exist_ok=True)
        expected_csv_count = grid_points_per_edge * grid_points_per_edge
        existing_csvs = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        if len(existing_csvs) >= expected_csv_count:
            print(f"Grid CSV folder {output_dir} exists and contains {len(existing_csvs)} CSVs. Skipping fetch.")
            return output_dir

        # Otherwise, fetch and save as before
        if max_lon == min_lon or max_lat == min_lat:
            lon_coords = np.array([min_lon])
            lat_coords = np.array([min_lat])
            if grid_points_per_edge > 1:
                print("Warning: BBox is point/line. Using 1 grid point.")
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
        print(f"Generated {len(grid_points_to_query)} grid points for weather fetching.")

        total_points_to_fetch = len(grid_points_to_query)
        for i, (lat, lon) in enumerate(grid_points_to_query):
            print(f"Fetching weather for grid point {i + 1}/{total_points_to_fetch} (Lat: {lat:.4f}, Lon: {lon:.4f})")
            point_weather_data_list = _fetch_hourly_data_for_period_at_point(
                lat, lon, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )
            # Save per-grid-point CSV
            y_idx = i // grid_points_per_edge
            x_idx = i % grid_points_per_edge
            csv_name = f"weather_grid_y{y_idx}_x{x_idx}.csv"
            csv_path = os.path.join(output_dir, csv_name)
            df_point = pd.DataFrame(point_weather_data_list)
            columns_order = ['latitude', 'longitude', 'timestamp', 'temperature_2m', 'relative_humidity_2m',
                             'precipitation', 'wind_speed_10m', 'wind_direction_10m', 'snowfall', 'weather_code',
                             'cloud_cover']
            df_point = df_point.reindex(columns=columns_order)
            df_point.to_csv(csv_path, index=False)
            print(f"Saved grid point CSV: {csv_path}")
            if i < total_points_to_fetch - 1:
                time.sleep(0.2)

        print(f"Gridded weather data saved to folder: {output_dir}")
        return output_dir
