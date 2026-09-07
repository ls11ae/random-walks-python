import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd
from environmentcma import (
    bbox_utm,
    clamp_lonlat_bbox,
    create_landcover_data_txt as env_create_landcover_data_txt,
    create_weather_csvs,
    grid_shape_from_bbox,
    grid_to_geo,
    padded_bbox,
    traj_utm,
    utm_to_grid,
)

from randomwalks.bindings.data_structures.Terrain import TerrainMapHandle
from randomwalks.bindings.movebank_parser import df_add_properties2
from randomwalks.bindings.walks_serialization import serialize_env_grid, serialize_kernel_paths_json
from randomwalks.core.KernelFactory import (
    StateAnnotationMethod,
    annotate_states,
    state_kernels,
)
from randomwalks.core.WalkerHelper import WalkerHelper


def coerce_trajectory_collection(
        data,
        time_col="timestamp",
        lon_col="location-long",
        lat_col="location-lat",
        id_col="tag-local-identifier",
        target_crs="EPSG:4326",
):
    if isinstance(data, (str, os.PathLike)):
        data = pd.read_csv(data)
        if time_col in data.columns:
            data[time_col] = pd.to_datetime(data[time_col], errors="coerce")

    if isinstance(data, mpd.TrajectoryCollection):
        traj_col = data
        gdf = traj_col.to_point_gdf().copy()
        orig_time = traj_col.t
        orig_id = traj_col.get_traj_id_col()

        if orig_time not in gdf.columns:
            gdf[orig_time] = gdf.index

        if gdf.crs is None:
            gdf = gdf.set_crs(target_crs)
        if str(gdf.crs) != target_crs:
            gdf = gdf.to_crs(target_crs)

        gdf["x"] = gdf.geometry.x
        gdf["y"] = gdf.geometry.y
        gdf = gdf.dropna(subset=["x", "y", orig_time, orig_id])

        return mpd.TrajectoryCollection(
            gdf,
            traj_id_col=orig_id,
            t=orig_time,
            x="x",
            y="y",
            crs=target_crs,
        )

    if not isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
        raise ValueError("Input must be a TrajectoryCollection, DataFrame, GeoDataFrame, or CSV path")

    gdf = gpd.GeoDataFrame(data.copy())

    if time_col not in gdf.columns:
        raise ValueError("time_col not found in dataframe")
    if id_col not in gdf.columns:
        raise ValueError(f"id_col: {id_col} not found in dataframe")

    if "geometry" not in gdf.columns:
        if lon_col not in gdf.columns or lat_col not in gdf.columns:
            raise ValueError("Need geometry or lon/lat columns")
        gdf = gdf.set_geometry(
            gpd.points_from_xy(gdf[lon_col], gdf[lat_col]),
            crs="EPSG:4326",
        )

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)

    gdf[time_col] = pd.to_datetime(gdf[time_col], errors="coerce")
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    gdf = gdf.dropna(subset=["x", "y", time_col, id_col])

    return mpd.TrajectoryCollection(
        gdf,
        traj_id_col=id_col,
        t=time_col,
        x="x",
        y="y",
        crs=target_crs,
    )


class AnimalMovementProcessor:

    def __init__(
            self,
            data,
            time_col="timestamp",
            lon_col="location-long",
            lat_col="location-lat",
            id_col="tag-local-identifier",
            target_crs="EPSG:4326",  # IMMER GEO
            env_samples=5,
            movement_policy=None,
            reference_speed=None,
            coerce_data=True,
    ):
        self.reference_speed = reference_speed
        self.terrain_paths = {}
        self.terrain_TIFFs = {}
        self.resolution = None
        self.env_samples = env_samples
        self.movement_policy = movement_policy
        if coerce_data:
            self.traj = coerce_trajectory_collection(
                data,
                time_col=time_col,
                lon_col=lon_col,
                lat_col=lat_col,
                id_col=id_col,
                target_crs=target_crs,
            )
        else:
            if not isinstance(data, mpd.TrajectoryCollection):
                raise ValueError("coerce_data=False requires a TrajectoryCollection")
            self.traj = data

        self.time_col = self.traj.t
        self.id_col = self.traj.get_traj_id_col()
        self.crs = target_crs
        self.start_dt = {
            str(traj.id): traj.get_start_time()
            for traj in self.traj.trajectories
        }

        self.end_dt = {
            str(traj.id): traj.get_end_time()
            for traj in self.traj.trajectories
        }
        self.annotation_result = None

    @property
    def traj_coll(self):
        return self.traj

    @property
    def terrain_path(self):
        return self.terrain_paths

    def bbox_geo(self, traj_id):
        # we dont save utm geo bboxes anymore, we compute them on the fly
        min_lon, min_lat, max_lon, max_lat = self.traj.get_trajectory(traj_id).df.total_bounds
        return clamp_lonlat_bbox(padded_bbox(min_lon, min_lat, max_lon, max_lat, padding=0.1))

    def bbox_utm(self, traj_id):
        return bbox_utm(self.traj.get_trajectory(traj_id))

    def create_landcover_data_txt(self, is_marine: bool = False, resolution: int = 200,
                                  out_directory: str | None = None) -> dict[Any, str]:
        self.resolution = resolution
        expected_paths = {
            trajectory.id: _landcover_txt_path(trajectory, resolution, out_directory)
            for trajectory in self.traj.trajectories
        }
        existing = {
            trajectory.id
            for trajectory in self.traj.trajectories
            if expected_paths[trajectory.id].is_file()
        }

        if not existing:
            self.terrain_paths = env_create_landcover_data_txt(
                traj=self.traj,
                is_marine=is_marine,
                resolution=resolution,
                out_directory=out_directory,
            )
        else:
            generated_paths = {}
            missing_trajectories = []
            for trajectory in self.traj.trajectories:
                if trajectory.id in existing:
                    _add_grid_coordinates(trajectory, resolution)
                    print(f"Landcover grid already exists, skipping: {expected_paths[trajectory.id]}")
                else:
                    missing_trajectories.append(trajectory)

            if missing_trajectories:
                generated_paths = env_create_landcover_data_txt(
                    traj=mpd.TrajectoryCollection(missing_trajectories),
                    is_marine=is_marine,
                    resolution=resolution,
                    out_directory=out_directory,
                )

            self.terrain_paths = {
                trajectory.id: (
                    str(expected_paths[trajectory.id])
                    if trajectory.id in existing
                    else generated_paths[trajectory.id]
                )
                for trajectory in self.traj.trajectories
            }
        self.terrain_TIFFs = {
            str(traj_id): _txt_to_tif_path(path, resolution)
            for traj_id, path in self.terrain_paths.items()
        }
        return self.terrain_paths

    def grid_to_geo_path(self, path, traj_id):
        utm_bounds, epsg = self.bbox_utm(traj_id)
        width, height = grid_shape_from_bbox(utm_bounds, self.resolution)
        geo = [grid_to_geo(x, y, utm_bounds, width, height, epsg) for x, y in path]
        df = pd.DataFrame(geo, columns=["longitude", "latitude"])
        return df

    def movebank_path_to_gdf(self, full_path, steps_df, animal_id, idx, segment_boundaries):
        if len(steps_df) > 0:
            last_row = steps_df.iloc[-1]
            last_grid = (int(last_row["grid_x"]), int(last_row["grid_y"]))
            if len(full_path) == 0 or tuple(full_path[-1]) != last_grid:
                full_path.append(last_grid)

        geodetic_path_df = self.grid_to_geo_path(full_path, animal_id)
        if not isinstance(geodetic_path_df, pd.DataFrame):
            geodetic_path_df = pd.DataFrame(geodetic_path_df, columns=["longitude", "latitude"])

        rows = WalkerHelper.create_timed_df(
            steps_df,
            geodetic_path_df,
            animal_id,
            idx,
            segment_boundaries,
            traj_id_col=self.id_col,
        )
        if not rows:
            return None

        final_df = pd.concat(rows, ignore_index=True)
        final_df["geometry"] = gpd.points_from_xy(final_df.longitude, final_df.latitude)
        return gpd.GeoDataFrame(final_df, geometry="geometry", crs="EPSG:4326")

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
        AnimalMovementProcessor.convert_env_csv_to_parquet(env_path, parquet_root, time_col=time_stamp)
        if self.reference_speed is None:
            diffusivity = 1.5
        else:
            diffusivity = None

        # for each animal trajectory
        for traj in self.traj.trajectories:
            trajectory_df = traj.df
            times = traj.df.index
            points = traj.df.geometry  # (lon, lat)
            intervals = [(times[i], times[i + 1]) for i in range(len(times) - 1)]
            point_pairs = [(points[i], points[i + 1]) for i in range(len(points) - 1)]

            aid = traj.id
            bbox = self.bbox_geo(aid)
            utm_bbox, epsg = self.bbox_utm(aid)
            width, height = grid_shape_from_bbox(utm_bbox, self.resolution)
            print(f"[KERNEL PARAMETERS] Processing {aid} with bbox {width} x {height}")

            terrain_pth = self.terrain_paths.get(aid)
            terrain_map = TerrainMapHandle.from_file(file=terrain_pth, delim=' ')

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

                sx = trajectory_df.iloc[index]["grid_x"]
                sy = trajectory_df.iloc[index]["grid_y"]

                ex = trajectory_df.iloc[index + 1]["grid_x"]
                ey = trajectory_df.iloc[index + 1]["grid_y"]

                print(f"[KERNEL PARAMETERS] Processing interval {index}\n")
                print(f"[KERNEL PARAMETERS] Start point {sx}, {sy}, End point {ex}, {ey}\n")
                print(f"[KERNEL PARAMETERS] Start Time {t_start}, End Time {t_end}\n")

                _, S = self.movement_policy.resolve([sx, sy], [ex, ey],
                                                    start_time=t_start, end_time=t_end,
                                                    reference_speed=self.reference_speed,
                                                    movement_diffusivity=diffusivity)
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

        serialize_kernel_paths_json(binary_paths, out_directory)
        return binary_paths

    def annotate_behavior(
            self,
            method: StateAnnotationMethod | str = StateAnnotationMethod.HMM,
            features=None,
            num_states=3,
            penalty=10.0,
            plot_path=None,
    ):
        result = annotate_states(
            self.traj,
            method=method,
            features=features,
            num_states=num_states,
            penalty=penalty,
            plot_path=plot_path,
        )

        self.annotation_result = result
        self.traj = result.trajectory_collection
        return result

    def generate_state_kernels(
            self,
            *,
            state_col="state",
            dt_tolerance=1.2,
            rnge=1000,
            reso=None,
            out_dir=None,
            mass_percentile=0.99,
            dt_model_s=None,
            time_factor=None,
            density_config=None,
            density_preset=None,
            density_method=None,
            density_model=None,
            n_components=None,
            covariance_type=None,
            reg_covar=None,
            reg_covariance=None,
            is_brownian=False,
    ):
        if self.annotation_result is None:
            raise ValueError("Call annotate_behavior() before generate_state_kernels().")
        return state_kernels(
            self.traj,
            state_col=state_col,
            dt_tolerance=dt_tolerance,
            rnge=rnge,
            reso=reso,
            out=out_dir,
            dt_model_s=dt_model_s,
            time_factor=time_factor,
            mass_percentile=mass_percentile,
            density_config=density_config,
            density_preset=density_preset,
            density_method=density_method,
            density_model=density_model,
            n_components=n_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            reg_covariance=reg_covariance,
            is_brownian=is_brownian,
        )


def _landcover_txt_path(trajectory, resolution, out_directory):
    min_lon, min_lat, max_lon, max_lat = trajectory.df.total_bounds
    root = Path("landcover" if out_directory is None else out_directory) / "landcover"
    filename = (
        f"landcover_{trajectory.id}_"
        f"{min_lon:.2f}_{min_lat:.2f}_{max_lon:.2f}_{max_lat:.2f}_{resolution}.txt"
    )
    return root / filename


def _add_grid_coordinates(trajectory, resolution):
    utm_bounds, _ = bbox_utm(trajectory)
    width, height = grid_shape_from_bbox(utm_bounds, resolution)
    projected = traj_utm(trajectory)
    grid_x, grid_y = utm_to_grid(
        width,
        height,
        utm_bounds,
        projected.df.geometry.x.values,
        projected.df.geometry.y.values,
    )
    trajectory.df["grid_x"] = grid_x
    trajectory.df["grid_y"] = grid_y


def _txt_to_tif_path(txt_path, resolution):
    path = Path(txt_path)
    suffix = f"_{resolution}.txt"
    if path.name.endswith(suffix):
        return path.with_name(path.name[:-len(suffix)] + ".tif")
    return path.with_suffix(".tif")
