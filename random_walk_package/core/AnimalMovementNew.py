from pathlib import Path

import geopandas as gpd
import movingpandas as mpd
from pyproj import CRS

from random_walk_package.data_sources.geo_fetcher import *
from random_walk_package.data_sources.land_cover_adapter import landcover_to_discrete_txt
from random_walk_package.data_sources.movebank_adapter import padded_bbox


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

    def traj_utm(self, traj_id):
        # we dont save utm bboxes anymore, we compute them on the fly
        traj = self.traj.get_trajectory(traj_id)

        lon, lat = traj.df.geometry.iloc[0].coords[0]
        utm_crs = CRS.from_epsg(32600 + int((lon + 180) // 6) + 1)

        return traj.to_crs(utm_crs)

    def bbox_geo(self, traj_id):
        # we dont save utm geo bboxes anymore, we compute them on the fly
        return self.traj.get_trajectory(traj_id).df.total_bounds

    def bbox_utm(self, traj_id):
        return self.traj_utm(traj_id).df.total_bounds

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
        traj_id_col = self.traj.get_traj_id_col()
        results = {}
        for traj_id in self.traj.to_point_gdf()[traj_id_col].unique():
            # RAW GEO BBOX (lon/lat)
            min_lon, min_lat, max_lon, max_lat = self.bbox_geo(traj_id)
            # Padded GEO BBOX (lon/lat) padding between 0-1.0 (0% to 100% of bbox size)
            min_lon, min_lat, max_lon, max_lat = padded_bbox(min_lon, min_lat, max_lon, max_lat, padding=0.2)
            # UTM BBOX (x/y)
            traj_utm = self.traj_utm(traj_id)
            # REGULAR GRID SHAPE (x/y)
            nx, ny = self._grid_shape_from_bbox(traj_utm.df.total_bounds, resolution)

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
                nx,
                ny,
                min_lon,
                max_lat,
                max_lon,
                min_lat,
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
        traj_id_col = self.traj.get_traj_id_col()
        for traj_id in self.traj.to_point_gdf()[traj_id_col].unique():
            results[str(traj_id)] = self.create_movement_data(traj_id)
        return results
