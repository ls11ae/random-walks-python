import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import geopandas as gpd
import movingpandas as mpd
from pandas import DataFrame
from pyproj import CRS

from random_walk_package.bindings import parse_terrain
from random_walk_package.bindings.data_processing.movebank_parser import df_add_properties
from random_walk_package.core.hmm import preprocess_for_hmm, apply_hmm
from random_walk_package.data_sources.geo_fetcher import *
from random_walk_package.data_sources.land_cover_adapter import landcover_to_discrete_txt
from random_walk_package.data_sources.movebank_adapter import padded_bbox, clamp_lonlat_bbox
from random_walk_package.data_sources.ocean_cover import fetch_ocean_cover_tif
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
                 lon_col="location-long",
                 lat_col="location-lat",
                 id_col="tag-local-identifier",
                 crs="EPSG:4326",
                 env_samples=5):
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
        # these columns must not be NaN
        required_cols = [time_col, lon_col, lat_col, id_col]

        # TrajectoryCollection
        if isinstance(data, mpd.TrajectoryCollection):
            self.traj = data
            return

        # GeoDataFrame
        if isinstance(data, gpd.GeoDataFrame):
            gdf = data.copy()

            # in case 'geometry' is missing
            if gdf.geometry is None:
                gdf = gdf.set_geometry(
                    gpd.points_from_xy(gdf[lon_col], gdf[lat_col]),
                    crs=crs
                )

        # Normal df
        else:
            gdf = gpd.GeoDataFrame(
                data.copy(),
                geometry=gpd.points_from_xy(data[lon_col], data[lat_col]),
                crs=crs,
            )

        gdf = gdf.dropna(subset=required_cols)
        self.traj = mpd.TrajectoryCollection(
            gdf,
            traj_id_col=id_col,
            t=time_col,
        )
        self.terrain_paths = {}  # terrain txt path per animal_id
        self.resolution = None
        self.env_samples = env_samples
        self.longitude_col = lon_col
        self.latitude_col = lat_col
        self.start_dt = {str(traj.id): traj.get_start_time() for traj in self.traj.trajectories}
        self.end_dt = {str(traj.id): traj.get_end_time() for traj in self.traj.trajectories}

    @property
    def terrain_path(self):
        return self.terrain_paths

    def time_period(self):
        return self.start_dt, self.end_dt

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

    def create_landcover_data_txt(self, is_marine: bool = False, resolution: int = 200,
                                  out_directory: str | None = None) -> dict[str, str]:
        """
        Generate per-animal landcover data (TIFF + TXT), named with animal_id and bbox.
        
        Parameters
        ----------
        resolution : int, optional
            Grid resolution identifier for file(default: 200)
        out_directory : str, optional
            Output directory path
            
        is_marine : bool, optional
            If True, generate ocean/land cover from shapefile instead of ESA WorldCover.
            Requires shapefile_path to be provided. (default: False)

        Returns:
            dict[str, str]: { animal_id: txt_path }
        """
        self.resolution = resolution
        if out_directory is None:
            out_directory = "landcover"

        out_directory = Path(out_directory, "landcover")
        out_directory.mkdir(exist_ok=True, parents=True)

        shapefile_path = resources.files("random_walk_package.resources.marine_cover") / "ne_10m_land.shp"

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
                if is_marine is True:
                    fetch_ocean_cover_tif(
                        shapefile_path,
                        (min_lon, min_lat, max_lon, max_lat),
                        str(tif_path),
                    )
                else:
                    fetch_landcover_data(
                        (min_lon, min_lat, max_lon, max_lat),
                        str(tif_path),
                    )

            landcover_to_discrete_txt(
                str(tif_path),
                nx, ny,
                min_lon, max_lat, max_lon, min_lat,
                str(txt_path),
            )
            if is_marine is True:
                with open(txt_path, 'r') as file:
                    data = file.read()
                OCEAN_VALUE = 0
                LAND_VALUE = 1
                OCEAN_VALUE_MAPPED = 80
                LAND_VALUE_MAPPED = 10
                # Use temporary placeholder to avoid conflicts
                data = data.replace(str(OCEAN_VALUE), str(OCEAN_VALUE_MAPPED))
                data = data.replace(str(LAND_VALUE), str(LAND_VALUE_MAPPED))
                data = data.replace("255", "0")

                with open(txt_path, 'w') as file:
                    file.write(data)

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
            kernel_resolver,  # function (landmark, row) -> KernelParametersPtr
            time_stamp='timestamp',
            lon='location-long',
            lat='location-lat',
            out_directory: str | None = None
    ):
        """
        This function defines the spatial grid where kernels will be evaluated, loads terrain information for the animal’s area,
        calls a custom kernel resolver for each row in the DataFrame to generate movement kernels
        
        
        :param df: Description
        :type df: DataFrame with the environment parameters
        :param kernel_resolver: your function that returns kernel parameters
        :param time_stamp: the name of your time instance column
        :param lon: the name of your longitude instance 
        :param lat: the name of your latitude instance 
    
        """
        if out_directory is None:
            out_directory = "kernels"
        out_directory = Path(out_directory)
        out_directory.mkdir(exist_ok=True, parents=True)
        results = {}
        times = {}
        for traj in self.traj.trajectories:
            aid = traj.id
            bbox = self.bbox_geo(aid)
            utm_bbox, epsg = self.bbox_utm(aid)
            width, height = self._grid_shape_from_bbox(utm_bbox, self.resolution)
            print(f"[KERNEL PARAMETERS] Processing {aid} with bbox {width} x {height}")
            terrain_pth = self.terrain_paths.get(aid)
            terrain_map = parse_terrain(file=terrain_pth, delim=' ')
            df_proc, t = df_add_properties(
                df=df,
                kernel_resolver=kernel_resolver,
                terrain=terrain_map,
                bbox_geo=bbox,
                grid_width=width,
                grid_height=height,
                utm_code=epsg,
                start_date=self.start_dt[str(aid)],
                end_date=self.end_dt[str(aid)],
                time_stamp=time_stamp,
                grid_points_per_edge=self.env_samples,
                lon=lon,
                lat=lat,
            )
            times[aid] = t

            # Save CSV
            out_path = os.path.join(out_directory, f"{aid}_kernel_data.csv")
            df_proc.to_csv(out_path, index=False)
            results[str(aid)] = out_path
        print(f"KernelData Saved: {out_directory}")
        return results

    def add_states(self):
        def utm_crs_from_geometry(geom):
            lon, lat = geom.coords[0]
            zone = int((lon + 180) // 6) + 1
            epsg = 32600 + zone if lat >= 0 else 32700 + zone
            return CRS.from_epsg(epsg)

        self.traj.add_speed()
        self.traj.add_direction()
        self.traj.add_angular_difference()
        self.traj.add_distance()
        data_gdf = self.traj.to_point_gdf()
        data_gdf = data_gdf.copy()
        # UTM per individual animal
        utm_gdfs = []
        for traj_id, sub in data_gdf.groupby('individual-local-identifier'):
            utm_crs = utm_crs_from_geometry(sub.geometry.iloc[0])
            utm_gdfs.append(sub.to_crs(utm_crs))

        data_gdf_utm = gpd.GeoDataFrame(pd.concat(utm_gdfs), crs=utm_gdfs[0].crs)

        arrays, scaler, seq_dfs = preprocess_for_hmm(data_gdf_utm)
        apply_hmm(arrays, seq_dfs, plot=False)
