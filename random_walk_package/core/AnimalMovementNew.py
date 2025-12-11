import datetime
import os
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import movingpandas as mpd
from pandas import DataFrame
from pyproj import CRS

from random_walk_package.bindings import parse_terrain
from random_walk_package.bindings.data_processing.movebank_parser import df_add_properties
from random_walk_package.data_sources.geo_fetcher import *
from random_walk_package.data_sources.land_cover_adapter import landcover_to_discrete_txt
from random_walk_package.data_sources.movebank_adapter import padded_bbox, clamp_lonlat_bbox
from random_walk_package.data_sources.open_meteo_api import create_weather_csvs


@dataclass
class MovementTrajectory:
    traj_id: str
    df: pd.DataFrame

    # df columns: ["grid_x", "grid_y", "geo_x", "geo_y", "time"]

    def grid_steps(self) -> list[tuple[int, int]]:
        return list(zip(self.df.grid_x, self.df.grid_y))

    def geo_path(self) -> list[tuple[float, float]]:
        return list(zip(self.df.geo_x, self.df.geo_y))

    def __len__(self):
        return len(self.df)


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
        terrain_paths : dict[id, str]
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
        self.terrain_paths = {}  # terrain txt path per animal_id
        self.resolution = None
        self.env_samples = 5
        self.longitude_col = lon_col
        self.latitude_col = lat_col

    @property
    def terrain_path(self):
        return self.terrain_paths

    def traj_utm(self, traj_id):
        # we dont save utm bboxes anymore, we compute them on the fly
        traj = self.traj.get_trajectory(traj_id)

        lon, lat = traj.df.geometry.iloc[0].coords[0]
        utm_crs = CRS.from_epsg(32600 + int((lon + 180) // 6) + 1)

        return traj.to_crs(utm_crs)

    def bbox_geo(self, traj_id):
        # we dont save utm geo bboxes anymore, we compute them on the fly
        min_lon, min_lat, max_lon, max_lat = self.traj.get_trajectory(traj_id).df.total_bounds
        return clamp_lonlat_bbox(padded_bbox(min_lon, min_lat, max_lon, max_lat, padding=0.1))

    def bbox_utm(self, traj_id):
        utm_traj = self.traj_utm(traj_id)
        min_x, min_y, max_x, max_y = utm_traj.df.total_bounds
        return padded_bbox(min_x, min_y, max_x, max_y, padding=0.1), utm_traj.crs.to_epsg()

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

        out_directory = Path(out_directory, "landcover")
        out_directory.mkdir(exist_ok=True, parents=True)

        results = {}
        for traj in self.traj.trajectories:
            traj_id = traj.id
            # PADDED GEO BBOX (lon/lat)
            min_lon, min_lat, max_lon, max_lat = self.bbox_geo(traj_id)
            # PADDED UTM BBOX (x/y)
            utm_bbox, _ = self.bbox_utm(traj_id)
            # REGULAR GRID SHAPE (x/y)
            nx, ny = self._grid_shape_from_bbox(utm_bbox, resolution)

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

            results[traj_id] = str(txt_path)

        self.terrain_paths = results
        return results

    def create_movement_data(self, traj_id):
        traj_utm = self.traj_utm(traj_id)
        utm_bbox, _ = self.bbox_utm(traj_id)
        xmin, ymin, xmax, ymax = utm_bbox

        nx, ny = self._grid_shape_from_bbox(utm_bbox, self.resolution)
        df = traj_utm.df.copy()

        df["grid_x"] = ((df.geometry.x - xmin) / (xmax - xmin) * (nx - 1)).astype(int)
        df["grid_y"] = ((df.geometry.y - ymin) / (ymax - ymin) * (ny - 1)).astype(int)
        df["geo_x"] = self.traj.get_trajectory(traj_id).df[self.longitude_col]
        df["geo_y"] = self.traj.get_trajectory(traj_id).df[self.latitude_col]
        df["time"] = df.index

        return MovementTrajectory(traj_id=traj_id, df=df)

    def create_movement_data_dict(self):
        return {
            traj.id: self.create_movement_data(traj.id)
            for traj in self.traj.trajectories
        }

    @staticmethod
    def grid_to_geo(x, y, utm_bbox, width, height, epsg) -> tuple[float, float]:
        min_x, min_y, max_x, max_y = utm_bbox
        utm_x = min_x + (x / (width - 1)) * (max_x - min_x)
        utm_y = max_y - (y / (height - 1)) * (max_y - min_y)

        # UTM -> Geodetic
        lon, lat = utm_to_lonlat(utm_x, utm_y, epsg)
        return lon, lat

    def make_timestamped_geodetic_trajectory(self, full_path, movement_traj: MovementTrajectory):
        """
        full_path: list of (x, y) grid coords
        movement_traj: original MovementTrajectory with timestamps
        """

        # Convert all grid coords → lon/lat
        geo_df = self.grid_to_geo_path(full_path, movement_traj.traj_id)

        # track segment boundaries
        steps = movement_traj.df
        indices = steps.index
        boundaries = movement_traj.segment_boundaries  # collected during walk building

        rows = []
        for i in range(len(indices) - 1):
            t0 = steps.loc[indices[i], "time"]
            t1 = steps.loc[indices[i + 1], "time"]

            a = boundaries[i]
            b = boundaries[i + 1]
            seg = geo_df.iloc[a:b].copy()

            seg["time"] = self.interpolate_timestamps(t0, t1, len(seg))
            seg["traj_id"] = movement_traj.traj_id

            rows.append(seg)

        df = pd.concat(rows, ignore_index=True)
        df["geometry"] = gpd.points_from_xy(df.longitude, df.latitude)

        return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    @staticmethod
    def interpolate_timestamps(start_t, end_t, n_points):
        """Returns a list of timestamps of length n_points, inclusive."""
        return pd.date_range(start=start_t, end=end_t, periods=n_points).to_list()

    def grid_to_geo_path(self, path, traj_id):
        utm_bounds, epsg = self.bbox_utm(traj_id)
        width, height = self._grid_shape_from_bbox(utm_bounds, self.resolution)
        geo = [self.grid_to_geo(x, y, utm_bounds, width, height, epsg) for x, y in path]
        df = pd.DataFrame(geo, columns=["longitude", "latitude"])
        return df

    def fetch_open_meteo_weather(self, output_folder: str, samples_per_dimension: int = 5):
        self.env_samples = samples_per_dimension
        if output_folder is None:
            output_folder = "weather"
        out_directory = Path(output_folder)
        out_directory.mkdir(exist_ok=True, parents=True)

        expected_csv_count = self.env_samples * self.env_samples
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
                                grid_points_per_edge=self.env_samples,
                                fetch_hourly=fetch_hourly,
                                merged_csv_path=merged_csv_path,
                                results_map=results_map)
        return results_map

    def kernel_params_per_animal_csv(
            self,
            df: DataFrame,
            animal_id: str,
            kernel_resolver,  # function (landmark, row) -> KernelParametersPtr
            start_date: datetime.datetime,
            end_date: datetime.datetime,
            time_stamp='timestamp',
            lon='location-long',
            lat='location-lat',
            out_directory: str | None = None
    ):
        if out_directory is None:
            out_directory = "kernels"
        out_directory = Path(out_directory)
        out_directory.mkdir(exist_ok=True, parents=True)

        results = {}
        for traj in self.traj.trajectories:
            aid = traj.id
            if str(aid) != animal_id:
                continue
            bbox = self.bbox_geo(aid)
            utm_bbox, epsg = self.bbox_utm(aid)
            width, height = self._grid_shape_from_bbox(utm_bbox, self.resolution)
            print(f"[KERNEL PARAMETERS] Processing {aid} with bbox {width} x {height}")
            terrain_pth = self.terrain_paths.get(aid)
            terrain_map = parse_terrain(file=terrain_pth, delim=' ')
            df_proc, _ = df_add_properties(
                df=df,
                kernel_resolver=kernel_resolver,
                terrain=terrain_map,
                bbox_geo=bbox,
                grid_width=width,
                grid_height=height,
                utm_code=epsg,
                start_date=start_date,
                end_date=end_date,
                time_stamp=time_stamp,
                grid_points_per_edge=5,
                lon=lon,
                lat=lat,
            )

            # Save CSV
            out_path = os.path.join(out_directory, f"{aid}_kernel_data.csv")
            df_proc.to_csv(out_path, index=False)
            results[str(aid)] = out_path
        print(f"KernelData Saved: {out_directory}")
        return results
