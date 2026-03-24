# debugging: gdb --args python -m tests.test
import os.path
import pickle

from random_walk_package import StateDependentWalker, FixedStepsPolicy, TimeStepPolicy
from random_walk_package.bindings.data_structures.kernel_terrain_mapping import marine_kernels_baseline_crw, \
    update_kernels_mapping
from random_walk_package.core.MixedWalker import *
from random_walk_package.data_sources.walk_visualization import save_trajectory_collection_timed
from tests.mixed_walk_test import test_marine_walker


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


def examine_traj_coll_pickle(file, out):
    with open(file, "rb") as ff:
        traj_col: mpd.TrajectoryCollection = pickle.load(ff)

    print("\n=== OUTPUT PICKLE ===")
    print(type(traj_col))
    print(traj_col.to_point_gdf().head(20))
    print(traj_col.to_point_gdf().columns)
    print(traj_col.to_point_gdf().dtypes)

    out_dir = "random_walk_package/resources/move_apps"

    walk_dir = os.path.join(out_dir, out)
    save_trajectory_collection_timed(traj_col, str(walk_dir))





if __name__ == "__main__":
    import pandas as pd
    """output_file = "/home/omar/PycharmProjects/random-walks-python/tests/cats.pickle"
    examine_traj_coll_pickle(output_file, "output")
    examine_traj_coll_pickle(input_file, "input")
"""


    import movingpandas as mpd
    import movingpandas.trajectory_collection as mpd_tc

    print(mpd.TrajectoryCollection.to_point_gdf.__name__)
    print(mpd_tc.TrajectoryCollection.to_point_gdf.__name__)

    input_file = "/home/omar/PycharmProjects/random-walks-python/tests/rubythroat.pickle"
    out_dir = "random_walk_package/resources/move_apps"

    with open(input_file, "rb") as f:
        traj_col:mpd.TrajectoryCollection = pickle.load(f)
        print(traj_col.to_point_gdf().dtypes)

    data = pd.read_csv("random_walk_package/resources/leap_of_the_cat/The Leap of the Cat.csv")
    # animal type must be set Choice (Terrestrial, aerial or some better name for birds, Marine)
    # also for terrestrial: set behaviour towards water: 1. completely avoids water, cant cross water bodies 2. water is avoided but some points may be in water in the original dataset, if start in water
    # or must cross water (two points on either sides of a river for example, then it is possible) 3. water is like any other terrain
    # instead of resolution: user can set how fine-grained the walks should be. one step from one grid cell to another as the shortest unit. grid cell size (50m x 50x per cell for example)
    walker = StateDependentWalker(data=data,
                                  animal_type=Animal.TERRESTRIAL,
                                  resolution=600,
                                  out_directory=out_dir,
                                  n_hmm_states=2)  # data can also be a traj collection (MoveApp's input)
    # 3 options to determine number of steps and step size in grid: 1. specify 1 step every x seconds 2. fixed number of steps 3. automatic calculation but reference speed of animal must be provided
    mvm_pol = TimeStepPolicy(60*60*4) # this would be option 1
    walk_dir = os.path.join(out_dir, "walks")
    traj_coll = walker.generate_walks(out_dir=walk_dir,
                                      dt_tolerance=100.0,
                                      rnge=50,
                                      movement_policy=mvm_pol,
                                      max_cell_size=5, water_mode=WaterMode.FORBID,
                                      is_brownian=True)  # dt tolerance is a threshold to determine if two records belong to the same trajectory. 2 means deviation in time up to double the median delta t are allowed (depends on regularity of dataset, maybe allow automatic detection)

    id_col = traj_coll.get_traj_id_col()
    print(traj_coll.trajectories[0].df[id_col].dtype)  # MUSS object sein
    print(traj_coll.trajectories[0].df[id_col].map(type).value_counts().head())

    os.makedirs(walk_dir, exist_ok=True)
    save_trajectory_collection_timed(traj_coll, str(os.path.join(walk_dir, "timed.html")))  # creates leaflet html with TimestampedGeoJson
    pickle_path = os.path.join(walk_dir, "cat_walks.pickle")



    with open(pickle_path, 'wb') as f:
        pickle.dump(traj_coll, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(pickle_path, 'rb') as f:
        traj_collection = pickle.load(f)
        id_col = traj_collection.get_traj_id_col()
        print(traj_collection.to_point_gdf().dtypes)
        print(traj_collection.trajectories[0].df[id_col].dtype)  # MUSS object sein
        print(traj_collection.trajectories[0].df[id_col].map(type).value_counts().head())
        s = traj_collection.to_point_gdf()[id_col]
        print("dtype:", s.dtype, "array:", type(s.array))

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