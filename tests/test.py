# debugging: gdb --args python -m tests.test
import gzip
import pickle
import random
import pandas as pd
import numpy as np
from random_walk_package import create_correlated_kernel_parameters
from random_walk_package.bindings.data_structures.kernel_terrain_mapping import marine_kernels_baseline_crw, \
    update_kernels_mapping
from random_walk_package.bindings.plotter import plot_walk_from_json
from random_walk_package.core.MixedWalker import *
from random_walk_package.core.MovementPolicy import TimeStepPolicy
from random_walk_package import StateDependentWalker
from random_walk_package.data_sources.walk_visualization import save_trajectory_collection_timed
from tests.mixed_walk_test import test_marine_walker, test_time_walker, test_mixed_walk
from random_walk_package.core.MarineMovement import shark_data_filter


def weather_terrain_params(row):
    S = min(15, max(1, np.round(float(row["wind_speed_10m_max"]) / 2.0).astype(int)))
    D = min(16, max(1, np.round(int(row["wind_direction_10m_dominant"] // 45)).astype(int)))
    is_brownian = D == 1
    diffusity = float(row["cloud_cover_mean"]) / 100.0
    bias_x = int(row["precipitation_sum"] > 0.1)
    bias_y = int(row["terrain"] in (50, 60))
    return [is_brownian, S, D, diffusity, bias_x, bias_y]

def filter_bbox(traj_collection, bbox):
    gdf = traj_collection.to_point_gdf().copy()
    lon_min, lat_min, lon_max, lat_max = bbox

    keep = (
        (gdf.geometry.x >= lon_min) &
        (gdf.geometry.x <= lon_max) &
        (gdf.geometry.y >= lat_min) &
        (gdf.geometry.y <= lat_max)
    )

    gdf = gdf[keep]

    return mpd.TrajectoryCollection(
        gdf,
        traj_id_col="traj_id",
        t="time"
    )





if __name__ == "__main__":
    study = "random_walk_package/resources/Boars_Austria/boar_study_austria.csv"
    df = pd.read_csv(study)  # or a traj collection in case of MoveApps
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["location-long"], df["location-lat"]), crs="EPSG:4326")
    f = open("random_walk_package/resources/move_apps/input4_LatLon.pickle", "rb")
    traj_col = pickle.load(f)#mpd.TrajectoryCollection(gdf, traj_id_col="individual-local-identifier", t="timestamp")
    short_trajs = []

    for traj in traj_col:
        short_df = traj.df.iloc[:20].copy()
        short_traj = mpd.Trajectory(short_df, traj.id)
        print(short_df.head())
        short_trajs.append(short_traj)

    traj_col = mpd.TrajectoryCollection(short_trajs)
    gdf2 = traj_col.to_point_gdf()
    print(gdf2["main_location"].tolist())

    out_dir ="random_walk_package/resources/move_apps"

    # animal type must be set Choice (Terrestrial, aerial or some better name for birds, Marine)
    # also for terrestrial: set behaviour towards water: 1. completely avoids water, cant cross water bodies 2. water is avoided but some points may be in water in the original dataset, if start in water
    # or must cross water (two points on either sides of a river for example, then it is possible) 3. water is like any other terrain
    # instead of resolution: user can set how fine-grained the walks should be. one step from one grid cell to another as the shortest unit. grid cell size (50m x 50x per cell for example)
    walker = StateDependentWalker(data=traj_col, animal_type=Animal.AIRBORNE, resolution=500,
                                  out_directory=out_dir, n_hmm_states=2)  # data can also be a traj collection (MoveApp's input)
    # 3 options to determine number of steps and step size in grid: 1. specify 1 step every x seconds 2. fixed number of steps 3. automatic calculation but reference speed of animal must be provided
    mvm_pol = TimeStepPolicy(60*5) # this would be option 1
    walk_dir = os.path.join(out_dir, "walks")
    traj_coll = walker.generate_walks(out_dir=walk_dir,
                                      dt_tolerance=3.0,
                                      rnge=200,
                                      movement_policy=mvm_pol,
                                      max_cell_size=10, water_mode=WaterMode.ALLOW,
                                      is_brownian=True)  # dt tolerance is a threshold to determine if two records belong to the same trajectory. 2 means deviation in time up to double the median delta t are allowed (depends on regularity of dataset, maybe allow automatic detection)
    os.makedirs(walk_dir, exist_ok=True)
    save_trajectory_collection_timed(traj_coll, str(walk_dir))  # creates leaflet html with TimestampedGeoJson
    pickle_path = os.path.join(walk_dir, "state_walks.pickle")
    with gzip.open(pickle_path, 'wb') as f:
        pickle.dump(traj_coll, f, protocol=pickle.HIGHEST_PROTOCOL)   # this gets passed to the next MoveApp
    exit()
    # test_marine_walker()
    plot_walk_from_json(
        "/home/omar/PycharmProjects/random-walks-python/random_walk_package/resources/tiger_sharks/kernels/204413/.json")

    test_marine_walker()
    exit()

    study_path = 'random_walk_package/resources/tiger_sharks/shark_13_filtered.csv'
    study_df = pd.read_csv(study_path)
    env_samples = 5
    # i took the original csv but this also works for your processed csv with additional data, just adapt the kernel resolver
    env_path = '/home/omar/Downloads/current_filename.csv'
    # env_path = 'random_walk_package/resources/movebank_test/weather/weather_data_full.csv'
    processor = AnimalMovementProcessor(study_df, env_samples=env_samples)
    processor.create_landcover_data_txt(is_marine=True, resolution=1000, out_directory=os.path.dirname(study_path))
    processor.kernel_params_per_animal_binary(env_path=env_path,
                                              kernel_resolver=marine_params,
                                              time_stamp="time", lon="longitude", lat="latitude",
                                              out_directory=os.path.dirname(study_path))
    # example mapping for marine animals. Water is the only allowed landmark, motion is always correlated
    kernels_mapping = marine_kernels_baseline_crw(step_size=5, directions=8, angle_diffusity=0.3, len_diffusivity=1)
    # update the mapping parameters
    update_kernels_mapping(kernels_mapping, landmark=WATER, stepsize=7, directions=6, diffusity=1.5)

    exit()
    test_marine_walker()
