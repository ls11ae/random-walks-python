import os
from pathlib import Path

import geopandas as gpd
import movingpandas as mpd
from pyproj import CRS

from random_walk_package.data_sources.geo_fetcher import *
from random_walk_package.data_sources.land_cover_adapter import landcover_to_discrete_txt
from random_walk_package.data_sources.movebank_adapter import padded_bbox, clamp_lonlat_bbox
from random_walk_package.data_sources.open_meteo_api import create_weather_csvs


class AnimalMovementProcessor:
    def __init__(self,
                 data,
                 time_col="timestamp",
                 lon_col="longitude",
                 lat_col="latitude",
                 id_col="animal_id",
                 crs="EPSG:4326"):
        """
        Initializes an instance of a class to process and manage trajectory data.

        Parameters
        ----------
        data : pandas.DataFrame or geopandas.GeoDataFrame or mpd.TrajectoryCollection
            Input dataset containing trajectory data. If it is not an instance of
            `mpd.TrajectoryCollection`, it is converted into one.
        time_col : str, optional
            The name of the column containing time data, by default "timestamp".
        lon_col : str, optional
            The name of the column containing longitude data, by default "longitude".
        lat_col : str, optional
            The name of the column containing latitude data, by default "latitude".
        id_col : str, optional
            The name of the column containing unique animal identifiers, by default "animal_id".
        crs : str, optional
            The coordinate reference system to assign to the data if it is not already a
            GeoDataFrame, by default "EPSG:4326".

        Attributes
        ----------
        traj : mpd.TrajectoryCollection
            A TrajectoryCollection object containing processed trajectories.
        terrain_paths : dict[str, str]
            A dictionary storing terrain text paths for each unique animal identifier.
        resolution : None
            the number of cells in the regular grid along the longer axis of the bounding box.
        """
        if isinstance(data, mpd.TrajectoryCollection):
            self.traj = data

        else:
            if not isinstance(data, gpd.GeoDataFrame):
                gdf = gpd.GeoDataFrame(
                    data,
                    geometry=gpd.points_from_xy(
                        data[lon_col],
                        data[lat_col],
                    ),
                    crs=crs,
                )
            else:
                gdf = data

            self.traj = mpd.TrajectoryCollection(
                gdf,
                traj_id_col=id_col,
                t=time_col,
            )
        self.terrain_paths: dict[str, str] = {}  # terrain txt path per animal_id
        self.resolution = None
        self.weather_samples = 5

    def traj_utm(self, traj_id):
        # we dont save utm bboxes anymore, we compute them on the fly
        traj = self.traj.get_trajectory(traj_id)

        lon, lat = traj.df.geometry.iloc[0].coords[0]
        utm_crs = CRS.from_epsg(32600 + int((lon + 180) // 6) + 1)

        return traj.to_crs(utm_crs)

    def bbox_geo(self, traj_id):
        # we dont save utm geo bboxes anymore, we compute them on the fly
        min_lon, min_lat, max_lon, max_lat = self.traj.get_trajectory(traj_id).df.total_bounds
        return clamp_lonlat_bbox(padded_bbox(min_lon, min_lat, max_lon, max_lat, padding=0.2))

    def bbox_utm(self, traj_id):
        min_x, min_y, max_x, max_y = self.traj_utm(traj_id).df.total_bounds
        return padded_bbox(min_x, min_y, max_x, max_y, padding=0.2)

    @staticmethod
    def _grid_shape_from_bbox(bbox_utm, resolution):
        """Compute regular grid shape (width, height) from utm bounding box and resolution."""
        xmin, ymin, xmax, ymax = bbox_utm
        width_m = xmax - xmin
        height_m = ymax - ymin

        if width_m >= height_m:
            nx = resolution
            ny = max(1, int(resolution * height_m / width_m))
        else:
            ny = resolution
            nx = max(1, int(resolution * width_m / height_m))

        return nx, ny

    def create_landcover_data_txt(self, resolution: int = 200, out_directory: str | None = None) -> dict[str, str]:
        """Generate per-animal landcover data (TIFF + TXT), named with animal_id and bbox.

        Returns:
            dict[str, str]: { animal_id: txt_path }
        """
        self.resolution = resolution
        if out_directory is None:
            out_directory = "landcover"
        out_directory = Path(out_directory)
        out_directory.mkdir(exist_ok=True, parents=True)

        results = {}
        for traj in self.traj.trajectories:
            traj_id = traj.id
            # PADDED GEO BBOX (lon/lat)
            min_lon, min_lat, max_lon, max_lat = self.bbox_geo(traj_id)
            # PADDED UTM BBOX (x/y)
            min_x, min_y, max_x, max_y = self.bbox_utm(traj_id)
            # REGULAR GRID SHAPE (x/y)
            nx, ny = self._grid_shape_from_bbox([min_x, min_y, max_x, max_y], resolution)

            # Output paths
            base_name = (
                f"landcover_{traj_id}_"
                f"{min_lon:.2f}_{min_lat:.2f}_{max_lon:.2f}_{max_lat:.2f}"
            )
            tif_path = out_directory / f"{base_name}.tif"
            txt_path = out_directory / f"{base_name}_{resolution}.txt"

            # only fetch TIFF if it doesn't exist yet
            if not tif_path.exists():
                fetch_landcover_data(
                    (min_lon, min_lat, max_lon, max_lat),
                    str(tif_path),
                )
            # always reconstruct terrain txt files
            landcover_to_discrete_txt(
                str(tif_path),
                nx, ny,
                min_lon, max_lat, max_lon, min_lat,
                str(txt_path),
            )

            results[str(traj_id)] = str(txt_path)

        self.terrain_paths = results
        return results

    def create_movement_data(self, traj_id: str):
        traj_geo = self.traj.get_trajectory(traj_id)
        traj_utm = self.traj_utm(traj_id)

        xmin, ymin, xmax, ymax = traj_utm.df.total_bounds

        nx, ny = AnimalMovementProcessor._grid_shape_from_bbox(traj_utm.df.total_bounds, self.resolution)
        # ADD GRID COORDINATES
        df = traj_utm.df.copy()
        df["gx"] = ((df.geometry.x - xmin) / (xmax - xmin) * (nx - 1)).astype(int)
        df["gy"] = ((df.geometry.y - ymin) / (ymax - ymin) * (ny - 1)).astype(int)

        return {
            "grid": df[["gx", "gy"]],
            "geo": traj_geo.df.geometry,
            "time": traj_geo.df.index
        }

    def create_movement_data_dict(self):
        results = {}
        for traj in self.traj.trajectories:
            traj_id = traj.id
            results[str(traj_id)] = self.create_movement_data(str(traj_id))
        return results

    @staticmethod
    def grid_to_geo(x, y, min_x, min_y, max_x, max_y, width, height, epsg) -> tuple[float, float]:
        utm_x = min_x + (x / (width - 1)) * (max_x - min_x)
        utm_y = max_y - (y / (height - 1)) * (max_y - min_y)

        # UTM -> Geodetic
        lon, lat = utm_to_lonlat(utm_x, utm_y, epsg)
        return lon, lat

    def grid_coordinates_to_geodetic(self, coord: list[tuple[int, int]], animal_id: str) -> pd.DataFrame:
        utm_traj = self.traj_utm(animal_id)
        epsg = utm_traj.crs
        min_x, min_y, max_x, max_y = utm_traj.df.total_bounds
        # todo: rewrite padded bbox to accept tuple
        min_x, min_y, max_x, max_y = padded_bbox(min_x, min_y, max_x, max_y, padding=0.2)
        width, height = AnimalMovementProcessor._grid_shape_from_bbox((min_x, min_y, max_x, max_y), self.resolution)
        df = pd.DataFrame(coord, columns=["gx", "gy"])
        df["lon"], df["lat"] = df.apply(lambda row: self.grid_to_geo(row["gx"], row["gy"],
                                                                     min_x, min_y, max_x, max_y, width, height,
                                                                     epsg), axis=1)
        # todo: add timestamps
        return df

    def fetch_open_meteo_weather(self, output_folder: str, samples_per_dimension: int = 5):
        self.weather_samples = samples_per_dimension
        if output_folder is None:
            output_folder = "weather"
        out_directory = Path(output_folder)
        out_directory.mkdir(exist_ok=True, parents=True)

        expected_csv_count = self.weather_samples * self.weather_samples
        results_map: dict[str, str] = {}
        for traj in self.traj.trajectories:
            traj_id = traj.id
            min_lon, min_lat, max_lon, max_lat = self.bbox_geo(traj_id)
            animal_dir = os.path.join(output_folder, str(traj_id))
            os.makedirs(animal_dir, exist_ok=True)

            start_date = self.traj.get_trajectory(traj_id).get_start_time()
            end_date = self.traj.get_trajectory(traj_id).get_end_time()
            delta = end_date - start_date
            exact_days = delta / pd.Timedelta(days=1)
            fetch_hourly: bool = exact_days < 20
            merged_csv_path = animal_dir

            # Check if per-grid CSVs already exist
            existing_point_csvs = [f for f in os.listdir(animal_dir)
                                   if f.endswith('.csv') and f.startswith('weather_grid_y')]
            if len(existing_point_csvs) >= expected_csv_count:
                print(
                    f"Grid CSV folder {animal_dir} exists and contains {len(existing_point_csvs)} CSVs. Skipping fetch.")
                results_map[str(traj_id)] = merged_csv_path
                continue

            create_weather_csvs(bbox=[min_lon, min_lat, max_lon, max_lat],
                                interval=(start_date, end_date),
                                animal_id=traj_id,
                                animal_dir=animal_dir,
                                grid_points_per_edge=self.weather_samples,
                                fetch_hourly=fetch_hourly,
                                merged_csv_path=merged_csv_path,
                                results_map=results_map)
        return results_map
