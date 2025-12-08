import datetime
import os
import time

import numpy as np
from pandas.core.interchange.dataframe_protocol import DataFrame

from random_walk_package.bindings import parse_terrain
from random_walk_package.bindings.data_processing.movebank_parser import df_add_properties
from random_walk_package.data_sources.geo_fetcher import *
from random_walk_package.data_sources.land_cover_adapter import landcover_to_discrete_txt
from random_walk_package.data_sources.movebank_adapter import get_start_end_dates, \
    get_unique_animal_ids, get_animal_coordinates, get_bounding_boxes_per_animal, \
    bbox_dict_to_discrete_space
from random_walk_package.data_sources.open_meteo_api import _fetch_hourly_data_for_period_at_point


# landcover_to_discrete_txt is imported in the original but not used, can be kept or removed.

class AnimalMovementProcessor:
    def __init__(self, data_file=None, df=None, environment_grid_size: int = 5):
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        base_project_dir = os.path.join(self.script_dir, '..')  # Adjust if script is not in a subdir of project root
        self.resources_dir = os.path.join(base_project_dir, 'resources')

        # Ensure data_file path is constructed correctly relative to resources_dir
        if data_file is not None:
            if not os.path.isabs(data_file):
                self.data_file = os.path.join(self.resources_dir, data_file)
            else:
                self.data_file = data_file

        self.df = df
        self.bbox: dict[str, tuple[float, float, float, float]] = {}  # geo bbox per animal_id
        self.bbox_utm: dict[str, tuple[float, float, float, float]] = {}  # utm bbox per animal_id
        self.aid_espg_map: dict[str, str] = {}  # EPSG code per animal_id
        self.terrain_paths: dict[str, str] = {}  # terrain txt path per animal_id
        self.discrete_params: dict[
            str, tuple[int, int, int, int]] = None  # grid parameters (0,0,width, height) per animal_id
        self.grid_coords = None
        self.geo_coords = None
        self._weather_data = None  # For trajectory weather
        self.grid_points_per_edge = environment_grid_size

        # Initialize common resources
        if data_file is not None or self.df is None:
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
        self.terrain_paths = results
        return results

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
            # Grid -> UTM
            utm_x = min_x + (x / (width - 1)) * (max_x - min_x)
            utm_y = max_y - (y / (height - 1)) * (max_y - min_y)

            # UTM -> Geodetic
            lon, lat = utm_to_lonlat(utm_x, utm_y, epsg_code)
            result.append((lat, lon))

        return result

    def fetch_gridded_weather_data(self, output_folder: str,
                                   grid_points_per_edge: int = 5) -> dict[str, str]:
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
        self.grid_points_per_edge = grid_points_per_edge
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

    def kernel_params_per_animal_csv(
            self,
            df: DataFrame,
            aid: str,
            kernel_resolver,  # function (landmark, row) -> KernelParametersPtr
            start_date: datetime,
            end_date: datetime,
            time_stamp='timestamp',
            lon='location-long',
            lat='location-lat'
    ):
        out_dir = os.path.join(self.resources_dir, 'kernel_data')
        os.makedirs(out_dir, exist_ok=True)

        results = {}
        bbox = self.bbox[aid]
        _, _, width, height = self.discrete_params.get(aid)
        print(f"[KERNEL PARAMETERS] Processing {aid} with bbox {width} x {height}")
        terrain_pth = self.terrain_paths.get(aid)
        terrain_map = parse_terrain(file=terrain_pth, delim=' ')
        df_proc, times = df_add_properties(
            df=df,
            kernel_resolver=kernel_resolver,
            terrain=terrain_map,
            bbox_geo=bbox,
            grid_width=width,
            grid_height=height,
            utm_code=self.aid_espg_map[str(aid)],
            start_date=start_date,
            end_date=end_date,
            time_stamp=time_stamp,
            grid_points_per_edge=self.grid_points_per_edge,
            lon=lon,
            lat=lat,
        )

        # Save CSV
        out_path = os.path.join(out_dir, f"{aid}_kernel_data.csv")
        df_proc.to_csv(out_path, index=False)
        results[str(aid)] = out_path

        print(f"KernelData Saved: {out_path}")

        return results, times
