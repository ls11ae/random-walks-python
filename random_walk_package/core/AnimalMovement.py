import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import geopandas as gpd
import movingpandas as mpd
import numpy as np
from pandas import DataFrame
from pyproj import CRS
import utm

from random_walk_package.bindings import parse_terrain, terrain_map_free, terrain_at
from random_walk_package.bindings.data_processing.movebank_parser import df_add_properties, df_add_properties2
from random_walk_package.core.KernelFactory import KernelFactory
from random_walk_package.data_sources.geo_fetcher import *
from random_walk_package.data_sources.land_cover_adapter import landcover_to_discrete_txt
from random_walk_package.data_sources.movebank_adapter import padded_bbox, clamp_lonlat_bbox
from random_walk_package.data_sources.ocean_cover import fetch_ocean_cover_tif
from random_walk_package.data_sources.open_meteo_api import create_weather_csvs
from random_walk_package.data_sources.walks_serialization import serialize_env_grid, serialize_kernel_paths_json


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

    def __init__(
        self,
        data,
        time_col="timestamp",
        lon_col="location-long",
        lat_col="location-lat",
        id_col="tag-local-identifier",
        target_crs="EPSG:4326",   # IMMER GEO
        env_samples=5,
        movement_policy=None,
        reference_speed=None,
    ):
        self.reference_speed = reference_speed
        self.terrain_paths = {}
        self.terrain_TIFFs = {}
        self.cell_sizes_m = {}
        self.resolution = None
        self.env_samples = env_samples
        self.movement_policy = movement_policy

        # TrajectoryCollection
        if isinstance(data, mpd.TrajectoryCollection):

            traj_col = data
            gdf = traj_col.to_point_gdf().copy()

            orig_time = traj_col.t
            orig_id = traj_col.get_traj_id_col()

            # Zeit auch als Spalte
            if orig_time not in gdf.columns:
                gdf[orig_time] = gdf.index


        # DataFrame / GeoDataFrame
        elif isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):

            gdf = gpd.GeoDataFrame(data.copy())

            if time_col not in gdf.columns:
                raise ValueError("time_col not found in dataframe")

            if id_col not in gdf.columns:
                raise ValueError("id_col not found in dataframe")

            orig_time = time_col
            orig_id = id_col

            if "geometry" not in gdf.columns:
                if lon_col not in gdf.columns or lat_col not in gdf.columns:
                    raise ValueError("Need geometry or lon/lat columns")

                gdf = gdf.set_geometry(
                    gpd.points_from_xy(gdf[lon_col], gdf[lat_col]),
                    crs="EPSG:4326"
                )

        else:
            raise ValueError("Input must be TrajectoryCollection or DataFrame")


        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")

        if str(gdf.crs) != target_crs:
            gdf = gdf.to_crs(target_crs)

        gdf["x"] = gdf.geometry.x
        gdf["y"] = gdf.geometry.y

        if orig_time not in gdf.columns:
            gdf[orig_time] = gdf.index

        gdf = gdf.dropna(subset=["x", "y", orig_time, orig_id])

        self.traj = mpd.TrajectoryCollection(
            gdf,
            traj_id_col=orig_id,
            t=orig_time,
            x="x",
            y="y",
            crs=target_crs,
        )

        self.time_col = orig_time
        self.id_col = orig_id
        self.crs = target_crs
        self.start_dt = {
            str(traj.id): traj.get_start_time()
            for traj in self.traj.trajectories
        }

        self.end_dt = {
            str(traj.id): traj.get_end_time()
            for traj in self.traj.trajectories
        }



    @property
    def cell_sizes(self):
        return self.cell_sizes_m

    @property
    def terrain_path(self):
        return self.terrain_paths

    def time_period(self):
        return self.start_dt, self.end_dt

    @staticmethod
    def geo_to_utm(lat, lon):
        easting, northing, zone_no, zone_let = utm.from_latlon(lat, lon)
        return easting, northing, zone_no, zone_let

    @staticmethod
    def utm_to_geo(utm_x, utm_y, zone_no, zone_let):
        return utm.to_latlon(utm_x, utm_y, zone_no, zone_let)

    def traj_utm(self, traj_id):
        # we dont save utm bboxes anymore, we compute them on the fly
        traj = self.traj.get_trajectory(traj_id)
        lon, lat = traj.df.geometry.iloc[0].x, traj.df.geometry.iloc[0].y

        print(f"{lon}, {lat}")
        print(traj.crs)

        zone = int((lon + 180) // 6) + 1
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        utm_crs = CRS.from_epsg(epsg)

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
    def grid_shape_from_bbox(bbox_utm, resolution):
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
            nx, ny = self.grid_shape_from_bbox(utm_bbox, resolution)
            # SIZE OF A (SQUARE) GRID CELL IN METERS
            self.cell_sizes_m[str(traj_id)] = (utm_bbox[2] - utm_bbox[0]) / nx

            # Output paths
            base_name = (
                f"landcover_{traj_id}_"
                f"{min_lon:.2f}_{min_lat:.2f}_{max_lon:.2f}_{max_lat:.2f}"
            )
            tif_path = out_directory / f"{base_name}.tif"
            txt_path = out_directory / f"{base_name}_{resolution}.txt"
            self.terrain_TIFFs[str(traj_id)] = tif_path
            # only fetch TIFF if it doesn't exist yet
            if not tif_path.exists():
                if is_marine:
                    fetch_ocean_cover_tif(
                        str(shapefile_path),
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
                res_x=nx, res_y=ny,
                min_lon=min_lon, max_lat=max_lat, max_lon=max_lon, min_lat=min_lat,
                output=str(txt_path),
            )
            if is_marine:
                with open(txt_path, 'r') as file:
                    data = file.read()
                OCEAN_VALUE = 0
                LAND_VALUE = 1
                OCEAN_VALUE_MAPPED = 80
                LAND_VALUE_MAPPED = 10
                # Use temporary placeholder to avoid conflicts
                data = data.replace(str(OCEAN_VALUE), str(OCEAN_VALUE_MAPPED))
                data = data.replace(str(LAND_VALUE), str(LAND_VALUE_MAPPED))
                data = data.replace("255", str(OCEAN_VALUE_MAPPED))

                with open(txt_path, 'w') as file:
                    file.write(data)

            results[traj_id] = str(txt_path)

        self.terrain_paths = results
        return results

    @staticmethod
    def utm_to_grid(nx, ny, xmin, ymin, xmax, ymax, utm_x, utm_y):
        x = np.round((utm_x - xmin) / (xmax - xmin) * (nx - 1)).astype(int)
        y = np.round((ymax - utm_y) / (ymax - ymin) * (ny - 1)).astype(int)
        return x, y

    def create_movement_data(self, traj_id, has_states):
        traj_utm = self.traj_utm(traj_id)
        utm_bbox, _ = self.bbox_utm(traj_id)
        xmin, ymin, xmax, ymax = utm_bbox

        nx, ny = self.grid_shape_from_bbox(utm_bbox, self.resolution)
        df = traj_utm.df.copy()

        gx, gy = self.utm_to_grid(
            nx, ny, xmin, ymin, xmax, ymax,
            df.geometry.x.values,
            df.geometry.y.values
        )

        df["grid_x"] = gx
        df["grid_y"] = gy
        traj = self.traj.get_trajectory(traj_id)
        df["geo_x"] = traj.df.geometry.x
        df["geo_y"] = traj.df.geometry.y
        utm_traj = self.traj_utm(traj_id)
        df["utm_x"] = utm_traj.df.geometry.x
        df["utm_y"] = utm_traj.df.geometry.y
        df["time"] = df.index
        if has_states:
            df["state"] = self.traj.get_trajectory(traj_id).df["state"]

        return MovementTrajectory(traj_id=traj_id, df=df)

    def create_movement_data_dict(self, has_states=False):
        return {
            traj.id: self.create_movement_data(traj.id, has_states)
            for traj in self.traj.trajectories
        }

    @staticmethod
    def grid_to_utm(x, y, utm_bbox, width, height):
        min_x, min_y, max_x, max_y = utm_bbox

        utm_x = min_x + x / (width - 1) * (max_x - min_x)
        utm_y = max_y - y / (height - 1) * (max_y - min_y)

        return utm_x, utm_y

    @staticmethod
    def grid_to_geo(x, y, utm_bbox, width, height, epsg):
        min_x, min_y, max_x, max_y = utm_bbox

        utm_x = min_x + x / (width - 1) * (max_x - min_x)
        utm_y = max_y - y / (height - 1) * (max_y - min_y)

        lon, lat = utm_to_lonlat(utm_x, utm_y, epsg)
        return lon, lat

    @staticmethod
    def grid_to_geo_walk(walk, utm_bbox, width, height, epsg):
        result = [(AnimalMovementProcessor.grid_to_geo(x, y, utm_bbox, width, height, epsg)) for x, y in walk]
        return result


    def grid_to_geo_path(self, path, traj_id):
        utm_bounds, epsg = self.bbox_utm(traj_id)
        width, height = self.grid_shape_from_bbox(utm_bounds, self.resolution)
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
            width, height = self.grid_shape_from_bbox(utm_bbox, self.resolution)
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

    @staticmethod
    def load_env_interval(t_start, t_end, parquet_dir, time_col):

        df = pd.read_parquet(
            parquet_dir,
            filters=[
                (time_col, ">=", t_start),
                (time_col, "<=", t_end),
            ],
            engine="pyarrow"
        )

        return df

    @staticmethod
    def convert_env_csv_to_parquet(env_csv, out_dir, time_col):
        print("Converting env csv to parquet...")
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)

        for chunk in pd.read_csv(
                env_csv,
                chunksize=1_000_000,
                parse_dates=[time_col]
        ):
            chunk["date"] = chunk[time_col].dt.strftime("%Y-%m-%d")
            chunk.to_parquet(
                out_dir,
                engine="pyarrow",
                partition_cols=["date"],
                compression="zstd"
            )
        print(f"Parquet Saved: {out_dir}")

    def kernel_params_per_animal_binary(
            self,
            env_path: str,
            kernel_resolver,  # function (df row) -> KernelParametersPtr
            time_stamp='timestamp',
            lon='location-long',
            lat='location-lat',
            out_directory: str | None = None
    ):
        """
        :param env_path: path to environment data CSV
        :param kernel_resolver: your function that returns kernel parameters from a row of your env dataframe
        :param time_stamp: the name of your time instance column
        :param lon: the name of your longitude instance
        :param lat: the name of your latitude instance
        :param out_directory: path to output directory
        """
        # prepare outer folder for kernels
        if out_directory is None:
            out_directory = "kernels"
        out_directory = Path(out_directory)
        out_directory.mkdir(exist_ok=True, parents=True)

        binary_paths: dict[tuple[str, str, str], str] = {}

        parquet_root = out_directory / "env_parquet"
        parquet_root.mkdir(exist_ok=True, parents=True)
        #AnimalMovementProcessor.convert_env_csv_to_parquet(env_path, parquet_root, time_col=time_stamp)
        if self.reference_speed is None:
            diffusivity = 1.5
        else:
            diffusivity = None

        # for each animal trajectory
        for traj in self.traj.trajectories:
            trajectories = self.create_movement_data(traj.id, has_states=False)
            times = traj.df.index
            points = traj.df.geometry           # (lon, lat)
            intervals = [(times[i], times[i + 1]) for i in range(len(times) - 1)]
            point_pairs = [(points[i], points[i + 1]) for i in range(len(points) - 1)]

            aid = traj.id
            bbox = self.bbox_geo(aid)
            utm_bbox, epsg = self.bbox_utm(aid)
            width, height = self.grid_shape_from_bbox(utm_bbox, self.resolution)
            print(f"[KERNEL PARAMETERS] Processing {aid} with bbox {width} x {height}")

            terrain_pth = self.terrain_paths.get(aid)
            terrain_map = parse_terrain(file=terrain_pth, delim=' ')

            aid_out = out_directory / str(aid)
            aid_out.mkdir(parents=True, exist_ok=True)

            for index, (t_start, t_end) in enumerate(intervals):
                ts = pd.Timestamp(t_start).strftime("%Y%m%dT%H")
                te = pd.Timestamp(t_end).strftime("%Y%m%dT%H")
                out_path_bin = aid_out / f"{aid}_kernels_{ts}-{te}.bin"
                out_path_csv = aid_out / f"{aid}_kernels_{ts}-{te}.csv"
                binary_paths[str(aid), ts, te] = str(out_path_bin)

                interval_df = AnimalMovementProcessor.load_env_interval(
                    t_start, t_end, parquet_root, time_col=time_stamp
                )

                if interval_df.empty:
                    continue

                start_x, start_y = point_pairs[index][0].x, point_pairs[index][0].y
                end_x, end_y = point_pairs[index][1].x, point_pairs[index][1].y

                sx = trajectories.df.iloc[index]["grid_x"]
                sy = trajectories.df.iloc[index]["grid_y"]

                ex = trajectories.df.iloc[index + 1]["grid_x"]
                ey = trajectories.df.iloc[index + 1]["grid_y"]

                print(f"[KERNEL PARAMETERS] Processing interval {index}\n")
                print(f"[KERNEL PARAMETERS] Start point {sx}, {sy}, End point {ex}, {ey}\n")
                print(f"[KERNEL PARAMETERS] Start Time {t_start}, End Time {t_end}\n")

                _, S = self.movement_policy.resolve([sx, sy], [ex, ey],
                                             start_time=t_start, end_time=t_end, reference_speed=self.reference_speed, movement_diffusivity=diffusivity)
                print(f"S: {S}\n")
                df_proc, T = df_add_properties2(
                    df=interval_df,
                    kernel_resolver=kernel_resolver,
                    terrain=terrain_map,
                    bbox_geo=bbox,
                    grid_width=width,
                    grid_height=height,
                    utm_code=epsg,
                    time_stamp=time_stamp,
                    grid_points_per_edge=self.env_samples,
                    lon=lon,
                    lat=lat,
                    start=(start_x, start_y),
                    end=(end_x, end_y),
                    S=S
                )

                # save binary
                serialize_env_grid(
                    binary_dir=str(out_path_bin),
                    kernel_df=df_proc,
                    time_col=time_stamp,
                    env_samples=self.env_samples,
                    T=T
                )
                # save csv
                df_proc.to_csv(out_path_csv, index=False)
                print(f"[KERNEL PARAMETERS] Saved CSV and Binary to {aid_out}")

            terrain_map_free(terrain_map)

        serialize_kernel_paths_json(binary_paths, out_directory)
        return binary_paths

    def get_hmm_kernels(self, dt_tolerance, rnge, out_dir=None, num_states=3):
        """Computes HMM kernels from trajectory data"""
        self.traj.add_speed(overwrite=True)
        self.traj.add_direction(overwrite=True)
        self.traj.add_angular_difference(overwrite=True)
        self.traj.add_distance(overwrite=True)
        data_gdf = self.traj.to_point_gdf()
        data_gdf = data_gdf.copy()
        # local mean utm zone
        mean_lon = data_gdf.geometry.x.mean()
        mean_lat = data_gdf.geometry.y.mean()
        zone = int((mean_lon + 180) // 6) + 1
        epsg = 32600 + zone if mean_lat >= 0 else 32700 + zone
        TARGET_CRS = f"EPSG:{epsg}"
        # UTM per individual animal
        utm_gdfs = []
        for traj_id, sub in data_gdf.groupby(self.id_col):
            sub = sub.copy()
            sub = gpd.GeoDataFrame(sub, geometry="geometry", crs=data_gdf.crs)
            # add terrain info
            grid_coords = self.create_movement_data(traj_id, False)
            terrain_map = parse_terrain(file=self.terrain_paths[traj_id], delim=' ')
            sub["terrain"] = [terrain_at(terrain_map, x, y) for x, y in grid_coords.grid_steps()]
            terrain_map_free(terrain_map)
            utm_gdfs.append(sub.to_crs(TARGET_CRS))

        data_gdf_utm = gpd.GeoDataFrame(
            pd.concat(utm_gdfs, ignore_index=True),
            crs=TARGET_CRS
        )
        data_gdf_utm.reset_index()
        # initialize HMM
        hmm_thingy = KernelFactory(data_gdf_utm, id_cols=self.id_col, num_states=num_states)
        # apply HMM to retrieve trajectories annotated with hidden states
        gdf = hmm_thingy.apply_hmm()
        # compute kernels from states
        [crwZ, brwZ] = hmm_thingy.get_state_kernels(dt_tolerance, rnge, 2 * rnge + 1, out_dir)
        original_gdf = self.traj.to_point_gdf()
        original_gdf["state"] = gdf["state"]
        self.traj = mpd.TrajectoryCollection(
            original_gdf,
            traj_id_col=self.id_col,
            t=self.time_col,
            crs=self.crs
        )
        return crwZ, brwZ
    
    def angular_diffusivity(self, t_prev, t_current, t_next):
        """
        Docstring for angular_diffusivity
        
        
        :param t_prev: timestamp for the start of a vector A
        :param t_current: timestamp for the end of a vector A, start of the vector B
        :param t_next: timestamp for the end of a vector B
        """
        # Rotate so that vector b->a aligns with x-axis
        pos_prev = self.traj.get_locations_at(t_prev)
        print(pos_prev)
        pos_current = self.traj.get_locations_at(t_current)
        pos_next = self.traj.get_locations_at(t_next)
        prev_piece = np.array([pos_current.geometry.iloc[0].x-pos_prev.geometry.iloc[0].x, pos_current.geometry.iloc[0].y-pos_prev.geometry.iloc[0].y]) #placeholder
        current_piece = np.array([pos_next.geometry.iloc[0].x-pos_current.geometry.iloc[0].x, pos_next.geometry.iloc[0].y-pos_current.geometry.iloc[0].y])
        angle_prev_to_xy = np.arctan2(prev_piece[1], prev_piece[0])
        cos_alpha = np.cos(-angle_prev_to_xy)
        sin_alpha = np.sin(-angle_prev_to_xy)
        rotation_matrix = np.array([
            [cos_alpha, -sin_alpha],
            [sin_alpha, cos_alpha]
        ])

        #Rotate next piece vector into the new coordinate system
        rotated_b = rotation_matrix @ current_piece

        #Calculate angle of rotated B with respect to new x-axis
        angle_rad = np.arctan2(rotated_b[1], rotated_b[0])
        angular_diffusivity = np.abs(np.sin(angle_rad))

        return angular_diffusivity
