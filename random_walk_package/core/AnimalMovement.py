import os
import os
import time

import numpy as np

# Assuming these imports are correctly resolved from your project structure
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
        self.bbox: dict[str, tuple[float, float, float, float]] = {}       # geo bbox per animal_id
        self.bbox_utm: dict[str, tuple[float, float, float, float]] = {}   # utm bbox per animal_id
        self.aid_espg_map: dict[str, str] = {}                             # EPSG code per animal_id
        self.discrete_params = None                                        # grid parameters (0,0,width, height) per animal_id
        self.grid_coords = None
        self.geo_coords = None
        self._weather_data = None  # For trajectory weather

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

    def create_movement_data(self, samples=10, time_stamped=False) -> tuple[
        dict[str, list[tuple[int, int]]], dict[str, list[tuple[int, int]]], dict[str, str] | None]:
        if self.df is None:
            self._load_data()

        grid_coords_per_animal: dict[str, list[tuple[int, int]]] = {}
        geo_coords_by_animal: dict[str, list[tuple[int, int]]] = {}
        time_stamps: dict[str, str] = {}
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
            coords_data, geo_coords, times = get_animal_coordinates(
                df=self.df,
                animal_id=aid,
                epsg_code=self.aid_espg_map[str(aid)],
                samples=samples,
                width=x_res,
                height=y_res,
                bbox_utm=bbox_utm,
                time_stamped=time_stamped
            )
            grid_coords_per_animal[str(aid)] = coords_data
            geo_coords_by_animal[str(aid)] = geo_coords
            if time_stamped:
                time_stamps[str(aid)] = times

        self.grid_coords = grid_coords_per_animal
        self.geo_coords = geo_coords_by_animal
        return grid_coords_per_animal, geo_coords_by_animal, time_stamps

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

    def fetch_gridded_weather_data(self, output_folder: str,
                                   grid_points_per_edge: int = 10) -> dict[str, str]:
        """
        Fetches and saves hourly weather data for a grid of equidistant points within the bounding box
        of each animal. For each animal, per-grid-point CSVs are written into a dedicated subfolder,
        and a merged CSV (weather_grid_all.csv) is created that aggregates all grid points.

        Args:
            output_folder (str): Root folder where per-animal grid CSVs should be stored.
            grid_points_per_edge (int): Number of points along each edge of the bounding box.

        Returns:
            dict[str, str]: Mapping {animal_id: path_to_merged_csv} for each processed animal.
        """
        # Ensure bounding boxes are available
        if not self.bbox:
            self._compute_bbox()
            if not self.bbox:
                raise ValueError("Bounding boxes are not set. Load data first.")

        # Prepare root output directory
        base_output_dir = output_folder
        os.makedirs(base_output_dir, exist_ok=True)

        expected_csv_count = grid_points_per_edge * grid_points_per_edge
        results_map: dict[str, str] = {}

        # Iterate over each animal's bounding box
        for animal_id, bbox in self.bbox.items():
            min_lon, min_lat, max_lon, max_lat = bbox

            animal_dir = os.path.join(base_output_dir, str(animal_id))
            os.makedirs(animal_dir, exist_ok=True)

            min_date_str, end_date_str = get_start_end_dates(self.df, animal_id)
            if not min_date_str:
                raise ValueError(
                    "Could not determine start date from animal movement data. Provide 'start_date_override'.")
            start_date = pd.to_datetime(min_date_str)
            end_date = pd.to_datetime(end_date_str)
            delta = end_date - start_date
            days = delta.days  # integer days
            exact_days = delta / pd.Timedelta(days=1)
            fetch_hourly: bool = exact_days < 20
            merged_csv_path = animal_dir

            # Check if per-grid CSVs already exist
            existing_point_csvs = [f for f in os.listdir(animal_dir)
                                   if f.endswith('.csv') and f.startswith('weather_grid_y')]
            if len(existing_point_csvs) >= expected_csv_count:
                print(
                    f"Grid CSV folder {animal_dir} exists and contains {len(existing_point_csvs)} CSVs. Skipping fetch.")
                results_map[str(animal_id)] = merged_csv_path
                continue

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
            all_rows = []
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

        print(f"Gridded weather data stored under: {base_output_dir}")
        return results_map
